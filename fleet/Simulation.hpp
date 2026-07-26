#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "City.hpp"
#include "Request.hpp"
#include "Vehicle.hpp"

struct SimConfig {
    int numVehicles = 20;
    int horizonSec = 3600;          // one-hour shift
    int numRequests = 160;          // total demand over the horizon
    double vehicleSpeed = 1.0;      // distance units / second
    int vehicleCapacity = 1;
    int seed = 42;
    bool verbose = false;
    // When false, dispatch pauses for an external policy (PPO) instead of the
    // built-in greedy heuristic.
    bool useHeuristic = true;
};

struct SimMetrics {
    int requestsSpawned = 0;
    int requestsCompleted = 0;
    int requestsCancelled = 0;
    int assignments = 0;

    double waitSum = 0.0;
    double tripSum = 0.0;
    double busyVehicleSec = 0.0;

    double meanWait() const {
        return requestsCompleted > 0 ? waitSum / requestsCompleted : 0.0;
    }

    double meanTrip() const {
        return requestsCompleted > 0 ? tripSum / requestsCompleted : 0.0;
    }

    double completionRate() const {
        return requestsSpawned > 0
                   ? static_cast<double>(requestsCompleted) / requestsSpawned
                   : 0.0;
    }

    double utilization(int numVehicles, int horizonSec) const {
        const double denom =
            static_cast<double>(std::max(1, numVehicles)) * std::max(1, horizonSec);
        return busyVehicleSec / denom;
    }
};

#include "Recording.hpp"

enum class EventType : uint8_t {
    RequestSpawn = 0,
    ArrivePickup = 1,
    ArriveDropoff = 2,
    ArriveIdle = 3,
};

struct SimEvent {
    int time = 0;
    EventType type = EventType::RequestSpawn;
    int vehicleId = -1;
    int requestId = -1;
    int seq = 0;  // tie-break for stable ordering

    bool operator>(const SimEvent& other) const {
        if (time != other.time) return time > other.time;
        return seq > other.seq;
    }
};

class Simulation {
public:
    Simulation(City& city, SimConfig config = {})
        : city_(city), config_(std::move(config)), rng_(config_.seed) {
        reset();
    }

    void reset() {
        now_ = 0;
        nextSeq_ = 0;
        vehicles_.clear();
        requests_.clear();
        busySince_.clear();
        metrics_ = {};
        while (!events_.empty()) events_.pop();

        const int n = static_cast<int>(city_.intersections.size());
        if (n <= 0) return;

        std::uniform_int_distribution<int> nodeDist(0, n - 1);
        vehicles_.reserve(static_cast<size_t>(config_.numVehicles));
        for (int i = 0; i < config_.numVehicles; ++i) {
            Vehicle v;
            v.id = i;
            v.node = nodeDist(rng_);
            v.fromNode = v.node;
            v.toNode = v.node;
            v.capacity = config_.vehicleCapacity;
            v.status = VehicleStatus::Idle;
            vehicles_.push_back(v);
        }
        busySince_.assign(vehicles_.size(), -1);

        // Pre-schedule request spawns uniformly across the horizon.
        std::uniform_int_distribution<int> timeDist(0, std::max(0, config_.horizonSec - 1));
        std::vector<int> spawnTimes;
        spawnTimes.reserve(static_cast<size_t>(config_.numRequests));
        for (int i = 0; i < config_.numRequests; ++i) {
            spawnTimes.push_back(timeDist(rng_));
        }
        std::sort(spawnTimes.begin(), spawnTimes.end());

        requests_.reserve(static_cast<size_t>(config_.numRequests));
        for (int i = 0; i < config_.numRequests; ++i) {
            Request r;
            r.id = i;
            r.origin = nodeDist(rng_);
            do {
                r.destination = nodeDist(rng_);
            } while (r.destination == r.origin && n > 1);
            r.spawnTime = spawnTimes[static_cast<size_t>(i)];
            r.size = 1;
            r.status = RequestStatus::Scheduled;
            requests_.push_back(r);
            schedule(r.spawnTime, EventType::RequestSpawn, -1, r.id);
        }

        if (recording_) {
            initRecordingSkeleton();
        }
    }

    void set_recording(DayRecording* out, int sample_interval_sec = 60) {
        recording_ = out;
        sampleInterval_ = std::max(1, sample_interval_sec);
        nextSampleSec_ = 0;
        if (recording_) {
            initRecordingSkeleton();
        }
    }

    SimMetrics run() {
        if (recording_) {
            maybeRecordSamples();
        }
        while (advanceOneEvent()) {
        }
        const SimMetrics m = finalizeHorizon();
        finishRecording(m);
        return m;
    }

    // Flush request/sample/metadata into the active DayRecording after finalize.
    void finishRecording(const SimMetrics& m) {
        if (!recording_) return;
        syncRequestRecordsFinal();
        const int savedNow = now_;
        now_ = config_.horizonSec;
        maybeRecordSamples();
        if (recording_->samples.empty() ||
            recording_->samples.back().sec < config_.horizonSec) {
            recording_->samples.push_back(buildSample(config_.horizonSec));
        }
        now_ = savedNow;
        recording_->metrics = m;
        recording_->numVehicles = config_.numVehicles;
        recording_->numRequests = config_.numRequests;
        recording_->horizonSec = config_.horizonSec;
        recording_->vehicleSpeed = config_.vehicleSpeed;
        recording_->numIntersections = static_cast<int>(city_.intersections.size());
    }

    // Process the next event at or before horizon. Returns false when the
    // episode can no longer advance (no events, or next event past horizon).
    bool advanceOneEvent() {
        if (events_.empty()) return false;
        const SimEvent ev = events_.top();
        if (ev.time > config_.horizonSec) return false;
        events_.pop();
        now_ = ev.time;
        dispatchEvent(ev);
        if (recording_) {
            maybeRecordSamples();
        }
        return true;
    }

    // Cancel unfinished requests at horizon and settle busy vehicles.
    SimMetrics finalizeHorizon() {
        for (Request& r : requests_) {
            if (r.spawnTime > config_.horizonSec) continue;
            if (r.status != RequestStatus::Pending && r.status != RequestStatus::Assigned &&
                r.status != RequestStatus::PickedUp) {
                continue;
            }
            if (r.assignedVehicle >= 0) {
                Vehicle& v = vehicles_[static_cast<size_t>(r.assignedVehicle)];
                if (v.assignedRequest == r.id) {
                    settleBusy(v);
                    v.status = VehicleStatus::Idle;
                    v.node = v.toNode;
                    v.assignedRequest = -1;
                }
            }
            r.status = RequestStatus::Cancelled;
            r.assignedVehicle = -1;
            ++metrics_.requestsCancelled;
        }
        return metrics_;
    }

    int now() const { return now_; }
    const std::vector<Vehicle>& vehicles() const { return vehicles_; }
    const std::vector<Request>& requests() const { return requests_; }
    const SimMetrics& metrics() const { return metrics_; }
    const SimConfig& config() const { return config_; }
    bool eventsPending() const {
        return !events_.empty() && events_.top().time <= config_.horizonSec;
    }

    // Assign free vehicle → open request (used by heuristic and policy).
    void assignPickup(int vehicleId, int requestId) {
        Vehicle& v = vehicles_[static_cast<size_t>(vehicleId)];
        Request& r = requests_[static_cast<size_t>(requestId)];
        if (!v.isFree() || !r.isOpen()) return;

        v.assignedRequest = requestId;
        r.assignedVehicle = vehicleId;
        r.assignTime = now_;
        r.status = RequestStatus::Assigned;
        ++metrics_.assignments;
        recordRequestAssign(requestId);

        if (config_.verbose) {
            std::cout << "[" << now_ << "] assign vehicle " << vehicleId
                      << " -> request " << requestId
                      << " dist=" << city_.distance(v.node, r.origin) << "\n";
        }

        if (v.node == r.origin) {
            // Already at pickup — serve immediately via a zero-delay event.
            schedule(now_, EventType::ArrivePickup, vehicleId, requestId);
            v.status = VehicleStatus::EnRoutePickup;
            v.fromNode = v.node;
            v.toNode = v.node;
            v.departTime = now_;
            v.arriveTime = now_;
            markBusy(v);
            recordTrip(v, requestId, TripKind::Pickup);
        } else {
            beginTrip(v, r.origin, VehicleStatus::EnRoutePickup, EventType::ArrivePickup,
                      requestId);
        }
    }

    // Reposition an idle vehicle toward a destination node (IDLE action).
    void assignIdle(int vehicleId, int destNode) {
        Vehicle& v = vehicles_[static_cast<size_t>(vehicleId)];
        if (!v.isFree()) return;
        if (destNode < 0 || destNode >= static_cast<int>(city_.intersections.size())) return;
        if (v.node == destNode) return;
        if (!city_.reachable(v.node, destNode)) return;
        beginTrip(v, destNode, VehicleStatus::EnRouteIdle, EventType::ArriveIdle, -1);
    }

private:
    City& city_;
    SimConfig config_;
    std::mt19937 rng_;

    int now_ = 0;
    int nextSeq_ = 0;
    std::vector<Vehicle> vehicles_;
    std::vector<Request> requests_;
    SimMetrics metrics_;
    std::priority_queue<SimEvent, std::vector<SimEvent>, std::greater<SimEvent>> events_;

    // Per-vehicle busy accounting: time when current busy segment started (-1 if idle).
    std::vector<int> busySince_;

    DayRecording* recording_ = nullptr;
    int sampleInterval_ = 60;
    int nextSampleSec_ = 0;

    void initRecordingSkeleton() {
        if (!recording_) return;
        recording_->trips.clear();
        recording_->samples.clear();
        recording_->metrics = {};
        recording_->vehicleStartNodes.clear();
        recording_->vehicleStartNodes.reserve(vehicles_.size());
        for (const Vehicle& v : vehicles_) {
            recording_->vehicleStartNodes.push_back(v.node);
        }
        fillCityGeometry();
        initRequestRecords();
        nextSampleSec_ = 0;
    }

    void fillCityGeometry() {
        if (!recording_) return;
        CityGeometry& g = recording_->city;
        g.width = city_.width;
        g.height = city_.height;
        g.nodes.clear();
        g.nodes.reserve(city_.intersections.size());
        for (const Intersection* inter : city_.intersections) {
            g.nodes.emplace_back(inter->pos.x, inter->pos.y);
        }
        g.edges = city_.undirectedEdges;
    }

    void initRequestRecords() {
        if (!recording_) return;
        recording_->requests.clear();
        recording_->requests.reserve(requests_.size());
        for (const Request& r : requests_) {
            RequestRecord rr;
            rr.id = r.id;
            rr.origin = r.origin;
            rr.dest = r.destination;
            rr.spawnSec = r.spawnTime;
            rr.assignSec = r.assignTime;
            rr.pickupSec = r.pickupTime;
            rr.dropoffSec = r.dropoffTime;
            rr.status = static_cast<uint8_t>(r.status);
            recording_->requests.push_back(rr);
        }
    }

    void syncRequestRecordsFinal() {
        if (!recording_) return;
        if (recording_->requests.size() != requests_.size()) {
            initRequestRecords();
            return;
        }
        for (size_t i = 0; i < requests_.size(); ++i) {
            const Request& r = requests_[i];
            RequestRecord& rr = recording_->requests[i];
            rr.assignSec = r.assignTime;
            rr.pickupSec = r.pickupTime;
            rr.dropoffSec = r.dropoffTime;
            rr.status = static_cast<uint8_t>(r.status);
        }
    }

    RequestRecord* requestRecord(int requestId) {
        if (!recording_ || requestId < 0 ||
            requestId >= static_cast<int>(recording_->requests.size())) {
            return nullptr;
        }
        return &recording_->requests[static_cast<size_t>(requestId)];
    }

    void recordRequestAssign(int requestId) {
        if (RequestRecord* rr = requestRecord(requestId)) {
            rr->assignSec = now_;
            rr->status = static_cast<uint8_t>(RequestStatus::Assigned);
        }
    }

    void recordTrip(const Vehicle& v, int requestId, TripKind kind) {
        if (!recording_) return;
        TripRecord t;
        t.vehicleId = v.id;
        t.fromNode = v.fromNode;
        t.toNode = v.toNode;
        t.startSec = v.departTime;
        t.endSec = v.arriveTime;
        t.requestId = requestId;
        t.kind = kind;
        recording_->trips.push_back(t);
    }

    static TripKind tripKindFromStatus(VehicleStatus status) {
        switch (status) {
            case VehicleStatus::EnRouteDropoff:
                return TripKind::Dropoff;
            case VehicleStatus::EnRouteIdle:
                return TripKind::Idle;
            case VehicleStatus::EnRoutePickup:
            default:
                return TripKind::Pickup;
        }
    }

    MetricSample buildSample(int sec) const {
        MetricSample s;
        s.sec = sec;
        int pending = 0;
        for (const Request& r : requests_) {
            if (r.status == RequestStatus::Pending) ++pending;
        }
        int freeCount = 0;
        int busyCount = 0;
        for (const Vehicle& v : vehicles_) {
            if (v.isFree()) {
                ++freeCount;
            } else {
                ++busyCount;
            }
        }
        s.pending = pending;
        s.freeVehicles = freeCount;
        s.busyVehicles = busyCount;
        s.completed = metrics_.requestsCompleted;
        s.spawned = metrics_.requestsSpawned;
        s.meanWait = static_cast<float>(metrics_.meanWait());
        return s;
    }

    void maybeRecordSamples() {
        if (!recording_ || sampleInterval_ <= 0) return;
        while (nextSampleSec_ <= now_ && nextSampleSec_ <= config_.horizonSec) {
            recording_->samples.push_back(buildSample(nextSampleSec_));
            nextSampleSec_ += sampleInterval_;
        }
    }

    void schedule(int time, EventType type, int vehicleId, int requestId) {
        if (time < now_) time = now_;
        events_.push(SimEvent{time, type, vehicleId, requestId, nextSeq_++});
    }

    void markBusy(Vehicle& v) {
        if (busySince_.size() < vehicles_.size()) {
            busySince_.assign(vehicles_.size(), -1);
        }
        if (busySince_[static_cast<size_t>(v.id)] < 0) {
            busySince_[static_cast<size_t>(v.id)] = now_;
        }
    }

    void settleBusy(Vehicle& v) {
        if (busySince_.size() < vehicles_.size()) {
            busySince_.assign(vehicles_.size(), -1);
        }
        const int start = busySince_[static_cast<size_t>(v.id)];
        if (start >= 0) {
            metrics_.busyVehicleSec += static_cast<double>(now_ - start);
            busySince_[static_cast<size_t>(v.id)] = -1;
        }
    }

    void beginTrip(Vehicle& v, int dest, VehicleStatus status, EventType arriveType,
                   int requestId) {
        const int travel = city_.travelTime(v.node, dest, config_.vehicleSpeed);
        v.fromNode = v.node;
        v.toNode = dest;
        v.departTime = now_;
        v.arriveTime = now_ + (travel >= City::kUnreachable ? 1 : travel);
        v.status = status;
        markBusy(v);
        schedule(v.arriveTime, arriveType, v.id, requestId);
        recordTrip(v, requestId, tripKindFromStatus(status));
    }

    void dispatchEvent(const SimEvent& ev) {
        switch (ev.type) {
            case EventType::RequestSpawn:
                onRequestSpawn(ev.requestId);
                break;
            case EventType::ArrivePickup:
                onArrivePickup(ev.vehicleId, ev.requestId);
                break;
            case EventType::ArriveDropoff:
                onArriveDropoff(ev.vehicleId, ev.requestId);
                break;
            case EventType::ArriveIdle:
                onArriveIdle(ev.vehicleId);
                break;
        }
    }

    void onRequestSpawn(int requestId) {
        if (requestId < 0 || requestId >= static_cast<int>(requests_.size())) return;
        Request& r = requests_[static_cast<size_t>(requestId)];
        if (r.status != RequestStatus::Scheduled) return;
        r.status = RequestStatus::Pending;
        ++metrics_.requestsSpawned;
        if (RequestRecord* rr = requestRecord(requestId)) {
            rr->status = static_cast<uint8_t>(RequestStatus::Pending);
        }
        if (config_.verbose) {
            std::cout << "[" << now_ << "] spawn request " << requestId
                      << " " << r.origin << "->" << r.destination << "\n";
        }
        maybeDispatch();
    }

    void onArrivePickup(int vehicleId, int requestId) {
        if (vehicleId < 0 || vehicleId >= static_cast<int>(vehicles_.size())) return;
        if (requestId < 0 || requestId >= static_cast<int>(requests_.size())) return;

        Vehicle& v = vehicles_[static_cast<size_t>(vehicleId)];
        Request& r = requests_[static_cast<size_t>(requestId)];
        if (v.assignedRequest != requestId || r.assignedVehicle != vehicleId) return;
        if (r.status != RequestStatus::Assigned) return;

        v.node = r.origin;
        r.status = RequestStatus::PickedUp;
        r.pickupTime = now_;
        if (RequestRecord* rr = requestRecord(requestId)) {
            rr->pickupSec = now_;
            rr->status = static_cast<uint8_t>(RequestStatus::PickedUp);
        }
        if (config_.verbose) {
            std::cout << "[" << now_ << "] vehicle " << vehicleId
                      << " pickup request " << requestId << "\n";
        }
        beginTrip(v, r.destination, VehicleStatus::EnRouteDropoff,
                  EventType::ArriveDropoff, requestId);
    }

    void onArriveDropoff(int vehicleId, int requestId) {
        if (vehicleId < 0 || vehicleId >= static_cast<int>(vehicles_.size())) return;
        if (requestId < 0 || requestId >= static_cast<int>(requests_.size())) return;

        Vehicle& v = vehicles_[static_cast<size_t>(vehicleId)];
        Request& r = requests_[static_cast<size_t>(requestId)];
        if (v.assignedRequest != requestId || r.assignedVehicle != vehicleId) return;
        if (r.status != RequestStatus::PickedUp) return;

        v.node = r.destination;
        v.toNode = r.destination;
        v.assignedRequest = -1;
        settleBusy(v);
        v.status = VehicleStatus::Idle;

        r.status = RequestStatus::Completed;
        r.dropoffTime = now_;
        metrics_.requestsCompleted += 1;
        metrics_.waitSum += static_cast<double>(r.waitTime());
        metrics_.tripSum += static_cast<double>(r.tripTime());
        if (RequestRecord* rr = requestRecord(requestId)) {
            rr->dropoffSec = now_;
            rr->status = static_cast<uint8_t>(RequestStatus::Completed);
        }

        if (config_.verbose) {
            std::cout << "[" << now_ << "] vehicle " << vehicleId
                      << " dropoff request " << requestId
                      << " wait=" << r.waitTime() << "\n";
        }
        maybeDispatch();
    }

    void onArriveIdle(int vehicleId) {
        if (vehicleId < 0 || vehicleId >= static_cast<int>(vehicles_.size())) return;
        Vehicle& v = vehicles_[static_cast<size_t>(vehicleId)];
        if (v.status != VehicleStatus::EnRouteIdle) return;
        v.node = v.toNode;
        settleBusy(v);
        v.status = VehicleStatus::Idle;
        maybeDispatch();
    }

    void maybeDispatch() {
        if (config_.useHeuristic) {
            heuristicDispatch();
        }
        // Policy mode: FleetEnv pauses at decision points; no auto-dispatch.
    }

    // Exclusive greedy dispatch: oldest pending request ← nearest free vehicle.
    void heuristicDispatch() {
        std::vector<int> pending;
        pending.reserve(requests_.size());
        for (const Request& r : requests_) {
            if (r.isOpen()) pending.push_back(r.id);
        }
        std::sort(pending.begin(), pending.end(), [&](int a, int b) {
            const Request& ra = requests_[static_cast<size_t>(a)];
            const Request& rb = requests_[static_cast<size_t>(b)];
            if (ra.spawnTime != rb.spawnTime) return ra.spawnTime < rb.spawnTime;
            return a < b;
        });

        std::vector<int> freeVehicles;
        freeVehicles.reserve(vehicles_.size());
        for (const Vehicle& v : vehicles_) {
            if (v.isFree()) freeVehicles.push_back(v.id);
        }
        if (pending.empty() || freeVehicles.empty()) return;

        std::vector<char> vehicleTaken(vehicles_.size(), 0);

        for (int reqId : pending) {
            Request& r = requests_[static_cast<size_t>(reqId)];
            if (!r.isOpen()) continue;

            int bestVehicle = -1;
            int bestDist = City::kUnreachable;
            for (int vid : freeVehicles) {
                if (vehicleTaken[static_cast<size_t>(vid)]) continue;
                const Vehicle& v = vehicles_[static_cast<size_t>(vid)];
                if (v.capacity < r.size) continue;
                const int d = city_.distance(v.node, r.origin);
                if (d < bestDist) {
                    bestDist = d;
                    bestVehicle = vid;
                }
            }
            if (bestVehicle < 0 || bestDist >= City::kUnreachable) continue;

            assignPickup(bestVehicle, reqId);
            vehicleTaken[static_cast<size_t>(bestVehicle)] = 1;
        }
    }
};

inline std::string formatDuration(double seconds) {
    if (seconds < 0.0 || !std::isfinite(seconds)) return "n/a";
    const int total = static_cast<int>(std::lround(seconds));
    const int h = total / 3600;
    const int m = (total % 3600) / 60;
    const int s = total % 60;
    std::ostringstream out;
    if (h > 0) {
        out << h << "h " << m << "m " << s << "s";
    } else if (m > 0) {
        out << m << "m " << s << "s";
    } else {
        out << s << "s";
    }
    out << " (" << total << "s)";
    return out.str();
}

inline std::string formatPercent(double fraction) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << (100.0 * fraction) << "%";
    return out.str();
}

inline void printMetrics(const SimMetrics& m, const SimConfig& cfg) {
    const int unfinished = m.requestsSpawned - m.requestsCompleted;
    const double util = m.utilization(cfg.numVehicles, cfg.horizonSec);

    std::cout << "=== Fleet simulation results ===\n"
              << "\n"
              << "Setup\n"
              << "  Fleet size:              " << cfg.numVehicles << " vehicles\n"
              << "  Shift length:            " << formatDuration(cfg.horizonSec) << "\n"
              << "  Planned demand:          " << cfg.numRequests
              << " ride requests over the shift\n"
              << "  Vehicle speed:           " << cfg.vehicleSpeed
              << " distance units / second\n"
              << "\n"
              << "Request outcomes\n"
              << "  Appeared (spawned):      " << m.requestsSpawned
              << "  - guests who asked for a ride during the shift\n"
              << "  Completed:               " << m.requestsCompleted
              << "  - picked up and dropped off before the shift ended\n"
              << "  Unfinished at end:       " << unfinished
              << "  - still waiting, en route, or mid-trip when time ran out\n"
              << "  Cancelled at horizon:    " << m.requestsCancelled
              << "  - unfinished requests force-closed at shift end\n"
              << "  Completion rate:         " << formatPercent(m.completionRate())
              << "  - completed / appeared\n"
              << "\n"
              << "Service quality (completed rides only)\n"
              << "  Avg guest wait:          " << formatDuration(m.meanWait())
              << "  - spawn -> pickup\n"
              << "  Avg onboard trip:        " << formatDuration(m.meanTrip())
              << "  - pickup -> dropoff\n"
              << "\n"
              << "Dispatch / fleet\n"
              << "  Assignments made:        " << m.assignments
              << "  - times a free vehicle was matched to a request\n"
              << "  Fleet busy time:         " << formatPercent(util)
              << "  - share of (vehicles x shift) spent driving or serving\n";
}

/** Run a seeded heuristic episode and return a scrubbable recording. */
inline DayRecording record_day(uint64_t seed, RecordConfig cfg = {},
                               int sample_interval_sec = 60) {
    City city(cfg.cityWidth, cfg.cityHeight, cfg.numIntersections,
              static_cast<int>(seed & 0x7fffffffULL));

    SimConfig simCfg;
    simCfg.numVehicles = cfg.numVehicles;
    simCfg.numRequests = cfg.numRequests;
    simCfg.horizonSec = cfg.horizonSec;
    simCfg.vehicleSpeed = cfg.vehicleSpeed;
    simCfg.vehicleCapacity = cfg.vehicleCapacity;
    simCfg.seed = static_cast<int>(seed);
    simCfg.useHeuristic = true;
    simCfg.verbose = false;

    Simulation sim(city, simCfg);
    DayRecording recording;
    sim.set_recording(&recording, sample_interval_sec);
    sim.run();
    return recording;
}

#endif
