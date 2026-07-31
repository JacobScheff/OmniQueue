#ifndef FLEET_ENV_HPP
#define FLEET_ENV_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "City.hpp"
#include "Simulation.hpp"

// Must match fleet/config.py action / obs contract.
namespace fleet_env {

constexpr int kMaxRequests = 128;
constexpr int kNumSpecialActions = 2;
constexpr int kActionStay = kMaxRequests;
constexpr int kActionIdle = kMaxRequests + 1;
constexpr int kNumActions = kMaxRequests + kNumSpecialActions;

constexpr int kVehicleFeatDim = 8;
constexpr int kRequestFeatDim = 8;
constexpr int kPairwiseFeatDim = 4;
constexpr int kEnvFeatDim = 4;

// Flat layout (single deciding vehicle, V=1):
//   vehicle[Fv] | request[R*Fr] | pairwise[R*Fp] | env[Fe]
//   | request_mask[R] | action_mask[A]
//   | vehicle_node[1] | request_origin[R] | request_dest[R]
constexpr int kFlatObsDim =
    kVehicleFeatDim + kMaxRequests * kRequestFeatDim + kMaxRequests * kPairwiseFeatDim +
    kEnvFeatDim + kMaxRequests + kNumActions + 1 + kMaxRequests + kMaxRequests;

struct Observation {
    std::array<float, kFlatObsDim> flat{};
    int vehicleId = -1;
    int nRequests = 0;
};

struct EnvStepResult {
    Observation obs;
    float reward = 0.0f;
    bool done = false;
    bool has_obs = false;
    SimMetrics metrics;
};

struct EnvConfig {
    int cityWidth = 1200;
    int cityHeight = 1200;
    int numIntersections = 80;
    int numVehicles = 30;
    int numRequests = 120;
    int horizonSec = 3600;
    double vehicleSpeed = 2.0;
    int vehicleCapacity = 1;
    // Target average degree for city street generation (≈ 2 * streets / intersections).
    int avgStreetsPerIntersection = 5;
};

inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

// Reward weights. Immediate assignment terms give action-differentiated credit;
// pending/completed wait keep pressure on guest delay.
constexpr float kRewardCompletion = 1.0f;
constexpr float kRewardCompletedWait = 4.0f;   // * (wait_sec / horizon) per completed ride
constexpr float kRewardPendingWait = 12.0f;    // * pendingWaitSum / (horizon * numRequests) / step
constexpr float kRewardAssignEta = 10.0f;      // * pickup_eta / horizon (prefer nearer)
constexpr float kRewardAssignUrgency = 3.0f;   // * current_wait / horizon (prefer older)
constexpr float kRewardUnfinished = 2.0f;      // * unfinished / numRequests at horizon

struct AssignInfo {
    bool assigned = false;
    float pickupEtaSec = 0.0f;
    float requestWaitSec = 0.0f;
};

class FleetEnv {
public:
    explicit FleetEnv(uint64_t seed = 0, EnvConfig envConfig = {})
        : envConfig_(envConfig) {
        rebuild(seed);
    }

    Observation reset(uint64_t seed) {
        rebuild(seed);
        stayedVehicleIds_.clear();

        if (recordEnabled_) {
            sim_->set_recording(&recording_, sampleInterval_);
        }

        if (!pumpToDecision()) {
            // Degenerate episode (no decisions). Still return a zero obs.
            if (recordEnabled_) {
                const SimMetrics m = sim_->finalizeHorizon();
                sim_->finishRecording(m);
            }
            Observation obs{};
            return obs;
        }
        return buildObservation();
    }

    // Arm DayRecording for the next reset/episode (policy or heuristic trips).
    void enable_recording(int sample_interval_sec = 60) {
        recordEnabled_ = true;
        sampleInterval_ = std::max(1, sample_interval_sec);
        if (sim_) {
            sim_->set_recording(&recording_, sampleInterval_);
        }
    }

    const DayRecording& recording() const { return recording_; }

    EnvStepResult step(int action) {
        EnvStepResult result{};
        if (currentVehicleId_ < 0) {
            result.done = true;
            result.metrics = sim_->finalizeHorizon();
            if (recordEnabled_) {
                sim_->finishRecording(result.metrics);
            }
            return result;
        }

        const int completedBefore = sim_->metrics().requestsCompleted;
        const double waitSumBefore = sim_->metrics().waitSum;
        const float horizon =
            static_cast<float>(std::max(1, envConfig_.horizonSec));

        const AssignInfo assign = applyAction(action);

        const auto accumulateReward = [&]() -> float {
            const int newCompletions =
                sim_->metrics().requestsCompleted - completedBefore;
            const double newWaitSum = sim_->metrics().waitSum - waitSumBefore;
            const float nReq =
                static_cast<float>(std::max(1, envConfig_.numRequests));

            float reward = 0.0f;
            // Immediate, action-conditional: nearer + older pickups score better.
            if (assign.assigned) {
                reward -= kRewardAssignEta * assign.pickupEtaSec / horizon;
                reward += kRewardAssignUrgency * assign.requestWaitSec / horizon;
            }
            // Completions: bonus minus realized guest wait.
            reward += kRewardCompletion * static_cast<float>(newCompletions);
            if (newCompletions > 0) {
                reward -= kRewardCompletedWait *
                          static_cast<float>(newWaitSum) / horizon;
            }
            // Continuous pressure from everyone still waiting.
            reward -= kRewardPendingWait * pendingWaitSum() / (horizon * nReq);
            return reward;
        };

        if (!pumpToDecision()) {
            // Score pending wait before finalize cancels open requests.
            float reward = accumulateReward();
            result.metrics = sim_->finalizeHorizon();
            if (recordEnabled_) {
                sim_->finishRecording(result.metrics);
            }
            const float nReq =
                static_cast<float>(std::max(1, envConfig_.numRequests));
            const int unfinished = result.metrics.requestsSpawned -
                                   result.metrics.requestsCompleted;
            reward -= kRewardUnfinished *
                      static_cast<float>(std::max(0, unfinished)) / nReq;
            result.done = true;
            result.has_obs = false;
            result.reward = reward;
            return result;
        }

        result.reward = accumulateReward();
        result.has_obs = true;
        result.obs = buildObservation();
        result.metrics = sim_->metrics();
        return result;
    }

    const SimMetrics& metrics() const { return sim_->metrics(); }

private:
    EnvConfig envConfig_;
    std::unique_ptr<City> city_;
    std::unique_ptr<Simulation> sim_;
    int currentVehicleId_ = -1;
    // Vehicles that chose STAY at the current sim time; skipped until time moves.
    std::vector<int> stayedVehicleIds_;
    int stayedAtTime_ = -1;

    bool recordEnabled_ = false;
    int sampleInterval_ = 60;
    DayRecording recording_{};

    void rebuild(uint64_t seed) {
        city_ = std::make_unique<City>(envConfig_.cityWidth, envConfig_.cityHeight,
                                       envConfig_.numIntersections,
                                       static_cast<int>(seed & 0x7fffffffULL),
                                       envConfig_.avgStreetsPerIntersection);
        SimConfig cfg;
        cfg.numVehicles = envConfig_.numVehicles;
        cfg.numRequests = envConfig_.numRequests;
        cfg.horizonSec = envConfig_.horizonSec;
        cfg.vehicleSpeed = envConfig_.vehicleSpeed;
        cfg.vehicleCapacity = envConfig_.vehicleCapacity;
        cfg.seed = static_cast<int>(seed);
        cfg.useHeuristic = false;
        cfg.verbose = false;
        sim_ = std::make_unique<Simulation>(*city_, cfg);
        currentVehicleId_ = -1;
        stayedVehicleIds_.clear();
        stayedAtTime_ = -1;
    }

    float pendingWaitSum() const {
        float total = 0.0f;
        const int now = sim_->now();
        for (const Request& r : sim_->requests()) {
            if (!r.isOpen()) continue;
            total += static_cast<float>(std::max(0, now - r.spawnTime));
        }
        return total;
    }

    bool isReachablePickup(int vehicleId, int requestId) const {
        const Vehicle& v = sim_->vehicles()[static_cast<size_t>(vehicleId)];
        const Request& r = sim_->requests()[static_cast<size_t>(requestId)];
        if (!v.isFree() || !r.isOpen()) return false;
        if (v.capacity < r.size) return false;
        return city_->reachable(v.node, r.origin) &&
               city_->reachable(r.origin, r.destination);
    }

    bool vehicleHasLegalPickup(int vehicleId) const {
        for (const Request& r : sim_->requests()) {
            if (r.isOpen() && isReachablePickup(vehicleId, r.id)) return true;
        }
        return false;
    }

    bool vehicleStayedThisTick(int vehicleId) const {
        if (stayedAtTime_ != sim_->now()) return false;
        return std::find(stayedVehicleIds_.begin(), stayedVehicleIds_.end(), vehicleId) !=
               stayedVehicleIds_.end();
    }

    // Pick the next free vehicle that still has work, or return -1.
    int selectDecisionVehicle() const {
        for (const Vehicle& v : sim_->vehicles()) {
            if (!v.isFree()) continue;
            if (vehicleStayedThisTick(v.id)) continue;
            if (vehicleHasLegalPickup(v.id)) return v.id;
        }
        return -1;
    }

    // Advance DES until a policy decision is needed, or the episode ends.
    bool pumpToDecision() {
        while (true) {
            currentVehicleId_ = selectDecisionVehicle();
            if (currentVehicleId_ >= 0) return true;

            // All free vehicles stayed this tick, or nobody has legal work.
            // Advance time so STAY does not spin forever.
            if (!sim_->eventsPending()) return false;
            const int t0 = sim_->now();
            if (!sim_->advanceOneEvent()) return false;
            if (sim_->now() != t0) {
                stayedVehicleIds_.clear();
                stayedAtTime_ = -1;
            }
        }
    }

    AssignInfo applyAction(int action) {
        AssignInfo info{};
        const int vid = currentVehicleId_;
        if (vid < 0) return info;

        if (action >= 0 && action < kMaxRequests) {
            // Pointer into the padded pending-request list in the observation.
            const std::vector<int> pending = pendingRequestIds();
            if (action < static_cast<int>(pending.size())) {
                const int reqId = pending[static_cast<size_t>(action)];
                if (isReachablePickup(vid, reqId)) {
                    const Vehicle& v = sim_->vehicles()[static_cast<size_t>(vid)];
                    const Request& r = sim_->requests()[static_cast<size_t>(reqId)];
                    const int eta =
                        city_->travelTime(v.node, r.origin, envConfig_.vehicleSpeed);
                    info.assigned = true;
                    info.pickupEtaSec =
                        eta >= City::kUnreachable ? 0.0f : static_cast<float>(eta);
                    info.requestWaitSec =
                        static_cast<float>(std::max(0, sim_->now() - r.spawnTime));
                    sim_->assignPickup(vid, reqId);
                    return info;
                }
            }
            // Illegal pointer → treat as STAY.
            markStay(vid);
            return info;
        }

        if (action == kActionStay) {
            markStay(vid);
            return info;
        }

        if (action == kActionIdle) {
            // Simple IDLE: reposition toward the oldest pending request's origin
            // (or STAY if none). Not a separate anchor set in v1.
            const std::vector<int> pending = pendingRequestIds();
            if (!pending.empty()) {
                const Request& r =
                    sim_->requests()[static_cast<size_t>(pending.front())];
                const Vehicle& v = sim_->vehicles()[static_cast<size_t>(vid)];
                if (city_->reachable(v.node, r.origin) && v.node != r.origin) {
                    sim_->assignIdle(vid, r.origin);
                    return info;
                }
            }
            markStay(vid);
            return info;
        }

        markStay(vid);
        return info;
    }

    void markStay(int vehicleId) {
        if (stayedAtTime_ != sim_->now()) {
            stayedVehicleIds_.clear();
            stayedAtTime_ = sim_->now();
        }
        stayedVehicleIds_.push_back(vehicleId);
    }

    std::vector<int> pendingRequestIds() const {
        std::vector<int> pending;
        pending.reserve(sim_->requests().size());
        for (const Request& r : sim_->requests()) {
            if (r.isOpen()) pending.push_back(r.id);
        }
        std::sort(pending.begin(), pending.end(), [&](int a, int b) {
            const Request& ra = sim_->requests()[static_cast<size_t>(a)];
            const Request& rb = sim_->requests()[static_cast<size_t>(b)];
            if (ra.spawnTime != rb.spawnTime) return ra.spawnTime < rb.spawnTime;
            return a < b;
        });
        if (static_cast<int>(pending.size()) > kMaxRequests) {
            pending.resize(static_cast<size_t>(kMaxRequests));
        }
        return pending;
    }

    Observation buildObservation() const {
        Observation obs{};
        obs.vehicleId = currentVehicleId_;
        if (currentVehicleId_ < 0) return obs;

        const Vehicle& v = sim_->vehicles()[static_cast<size_t>(currentVehicleId_)];
        const auto pending = pendingRequestIds();
        obs.nRequests = static_cast<int>(pending.size());

        const float horizon = static_cast<float>(std::max(1, envConfig_.horizonSec));
        const float speed = static_cast<float>(std::max(1e-3, envConfig_.vehicleSpeed));
        const int nNodes = static_cast<int>(city_->intersections.size());
        const float invNodes = 1.0f / static_cast<float>(std::max(1, nNodes));

        float* out = obs.flat.data();
        int o = 0;

        // --- vehicle features (8) ---
        out[o++] = static_cast<float>(v.node) * invNodes;
        out[o++] = v.soc;
        out[o++] = static_cast<float>(v.capacity) / 4.0f;
        out[o++] = v.isFree() ? 1.0f : 0.0f;
        out[o++] = 0.0f;  // time-to-free (free now)
        out[o++] = static_cast<float>(sim_->now()) / horizon;
        out[o++] = static_cast<float>(sim_->metrics().assignments) /
                   static_cast<float>(std::max(1, envConfig_.numRequests));
        out[o++] = 1.0f;  // padding / reserved

        // --- request features (R, 8) ---
        for (int i = 0; i < kMaxRequests; ++i) {
            if (i < obs.nRequests) {
                const Request& r = sim_->requests()[static_cast<size_t>(pending[static_cast<size_t>(i)])];
                const float wait =
                    static_cast<float>(std::max(0, sim_->now() - r.spawnTime)) / horizon;
                const int tripDist = city_->distance(r.origin, r.destination);
                const float tripNorm =
                    tripDist >= City::kUnreachable
                        ? 1.0f
                        : clampf(static_cast<float>(tripDist) / (speed * horizon), 0.0f, 1.0f);
                out[o++] = static_cast<float>(r.origin) * invNodes;
                out[o++] = static_cast<float>(r.destination) * invNodes;
                out[o++] = wait;
                out[o++] = static_cast<float>(r.size) / 4.0f;
                out[o++] = tripNorm;
                out[o++] = 1.0f;  // open
                out[o++] = static_cast<float>(r.spawnTime) / horizon;
                out[o++] = 0.0f;  // reserved
            } else {
                for (int f = 0; f < kRequestFeatDim; ++f) out[o++] = 0.0f;
            }
        }

        // --- pairwise vehicle↔request (R, 4) ---
        for (int i = 0; i < kMaxRequests; ++i) {
            if (i < obs.nRequests) {
                const Request& r = sim_->requests()[static_cast<size_t>(pending[static_cast<size_t>(i)])];
                const int dPickup = city_->distance(v.node, r.origin);
                const bool reach = isReachablePickup(v.id, r.id);
                const float drive = dPickup >= City::kUnreachable
                                        ? 1.0f
                                        : clampf(static_cast<float>(dPickup) / (speed * horizon),
                                                 0.0f, 1.0f);
                const float distNorm =
                    dPickup >= City::kUnreachable
                        ? 1.0f
                        : clampf(static_cast<float>(dPickup) /
                                     static_cast<float>(std::max(1, envConfig_.cityWidth +
                                                                        envConfig_.cityHeight)),
                                 0.0f, 1.0f);
                out[o++] = drive;
                out[o++] = distNorm;
                out[o++] = drive;  // energy proxy
                out[o++] = reach ? 1.0f : 0.0f;
            } else {
                for (int f = 0; f < kPairwiseFeatDim; ++f) out[o++] = 0.0f;
            }
        }

        // --- env (4) ---
        int nPending = 0;
        int nFree = 0;
        float socSum = 0.0f;
        for (const Request& r : sim_->requests()) {
            if (r.isOpen()) ++nPending;
        }
        for (const Vehicle& veh : sim_->vehicles()) {
            if (veh.isFree()) ++nFree;
            socSum += veh.soc;
        }
        out[o++] = static_cast<float>(sim_->now()) / horizon;
        out[o++] = static_cast<float>(nPending) /
                   static_cast<float>(std::max(1, envConfig_.numRequests));
        out[o++] = socSum / static_cast<float>(std::max(1, envConfig_.numVehicles));
        out[o++] = static_cast<float>(nFree) /
                   static_cast<float>(std::max(1, envConfig_.numVehicles));

        // --- request mask ---
        for (int i = 0; i < kMaxRequests; ++i) {
            out[o++] = (i < obs.nRequests) ? 1.0f : 0.0f;
        }

        // --- action mask ---
        // STAY is masked whenever a legal pickup exists so the policy cannot
        // collapse into STAY-spam (many decisions, no progress).
        bool anyLegalPickup = false;
        for (int i = 0; i < obs.nRequests; ++i) {
            if (isReachablePickup(v.id, pending[static_cast<size_t>(i)])) {
                anyLegalPickup = true;
                break;
            }
        }
        for (int a = 0; a < kNumActions; ++a) {
            bool ok = false;
            if (a < obs.nRequests) {
                ok = isReachablePickup(v.id, pending[static_cast<size_t>(a)]);
            } else if (a == kActionStay) {
                ok = !anyLegalPickup;
            } else if (a == kActionIdle) {
                // Reposition only when there is demand but no legal pickup
                // (e.g. capacity / reachability) — otherwise prefer pickup.
                ok = obs.nRequests > 0 && !anyLegalPickup;
            }
            out[o++] = ok ? 1.0f : 0.0f;
        }

        // --- node indices (as floats; cast in Python) ---
        out[o++] = static_cast<float>(v.node);
        for (int i = 0; i < kMaxRequests; ++i) {
            if (i < obs.nRequests) {
                out[o++] = static_cast<float>(
                    sim_->requests()[static_cast<size_t>(pending[static_cast<size_t>(i)])]
                        .origin);
            } else {
                out[o++] = 0.0f;
            }
        }
        for (int i = 0; i < kMaxRequests; ++i) {
            if (i < obs.nRequests) {
                out[o++] = static_cast<float>(
                    sim_->requests()[static_cast<size_t>(pending[static_cast<size_t>(i)])]
                        .destination);
            } else {
                out[o++] = 0.0f;
            }
        }

        return obs;
    }
};

}  // namespace fleet_env

#endif
