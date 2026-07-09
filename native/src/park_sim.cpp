#include "park_sim.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include "graph_data.hpp"

namespace park {

namespace detail {

namespace gd = graph_data;

class Rng {
public:
    explicit Rng(uint64_t seed) : gen_(seed) {}

    double normal(double mean, double stddev) {
        return std::normal_distribution<double>(mean, stddev)(gen_);
    }

    double uniform01() { return std::uniform_real_distribution<double>(0.0, 1.0)(gen_); }

    int randint(int lo_inclusive, int hi_exclusive) {
        return std::uniform_int_distribution<int>(lo_inclusive, hi_exclusive - 1)(gen_);
    }

    double lognormal(double mu, double sigma) {
        return std::lognormal_distribution<double>(mu, sigma)(gen_);
    }

    std::mt19937_64& engine() { return gen_; }

private:
    std::mt19937_64 gen_;
};

struct Ride {
    RideStatus status = RideStatus::Open;
    int broken_until = 0;
    int generation = 0;
    double next_board_sec = 0.0;
    double current_wait = 0.0;
    int incoming = 0;
    std::unordered_map<int, int> pending_board;
    std::vector<int> on_ride;
    std::vector<int> evacuation;
    std::vector<int> evacuating_on_ride;
    bool evacuation_active = false;
};

class TimingWheel {
public:
    TimingWheel() : buckets_(kDaySeconds + 1) {}

    void schedule(int at_second, Event event) {
        if (at_second < current_sec_) {
            at_second = current_sec_;
        }
        if (at_second > kDaySeconds) {
            at_second = kDaySeconds;
        }
        buckets_[at_second].push_back(event);
        if (at_second > max_scheduled_) {
            max_scheduled_ = at_second;
        }
    }

    bool empty() const {
        if (max_scheduled_ < 0) {
            return true;
        }
        advance_cursor();
        return cursor_ > max_scheduled_;
    }

    std::pair<int, std::vector<Event>> pop_next() {
        advance_cursor();
        if (cursor_ > max_scheduled_) {
            return {current_sec_, {}};
        }
        current_sec_ = cursor_;
        auto events = std::move(buckets_[cursor_]);
        buckets_[cursor_] = {};
        const int sec = cursor_;
        ++cursor_;
        return {sec, events};
    }

private:
    void advance_cursor() const {
        while (cursor_ <= max_scheduled_ && buckets_[cursor_].empty()) {
            ++cursor_;
        }
    }

    mutable int cursor_ = 0;
    mutable int max_scheduled_ = -1;
    int current_sec_ = 0;
    std::vector<std::vector<Event>> buckets_;
};

struct PartyArrays {
    int count = 0;
    std::vector<int32_t> party_size;
    std::vector<int32_t> spawn_sec;
    std::vector<int32_t> leave_sec;
    std::vector<int32_t> location_node_idx;
    std::vector<float> effective_speed;
    std::vector<int8_t> state;
    std::vector<int32_t> target_ride_id;
    std::vector<int32_t> target_node_idx;
    std::vector<int32_t> walk_target_ride;
    std::vector<std::array<int16_t, kNumRides>> preference_order;
    std::vector<std::array<float, kNumRides>> balk_sec;
    std::vector<std::array<float, kNumRides>> preferences;
    std::vector<std::array<uint8_t, kNumRides>> must_do_remaining;
    std::vector<std::array<int16_t, kNumRides>> ride_history;
    std::vector<int32_t> rides_completed;
};

int party_walk_sec(int from_idx, int to_idx, float speed) {
    const int base = gd::kBaseWalkMatrix[from_idx][to_idx];
    const double scale = kBaseWalkingSpeed / std::max(0.1f, speed);
    return std::max(1, static_cast<int>(std::ceil(base * scale)));
}

int party_walk_to_ride_sec(int from_idx, int ride_id, float speed) {
    const int base = gd::kBaseWalkToRides[from_idx][ride_id];
    const double scale = kBaseWalkingSpeed / std::max(0.1f, speed);
    return std::max(1, static_cast<int>(std::ceil(base * scale)));
}

void compute_preference_order(
    const std::array<float, kNumRides>& prefs,
    const std::array<uint8_t, kNumRides>& must_do,
    std::array<int16_t, kNumRides>& order) {
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const bool must_a = must_do[a] != 0;
        const bool must_b = must_do[b] != 0;
        if (must_a != must_b) {
            return must_a > must_b;
        }
        if (prefs[a] != prefs[b]) {
            return prefs[a] > prefs[b];
        }
        return a < b;
    });
}

void compute_balk_sec(const std::array<float, kNumRides>& prefs, std::array<float, kNumRides>& out) {
    for (int i = 0; i < kNumRides; ++i) {
        out[i] = static_cast<float>(kBaseBalkSec + kBalkScale * std::pow(prefs[i], kBalkPrefExp));
    }
}

float party_effective_speed(Rng& rng, int party_size) {
    float min_speed = std::numeric_limits<float>::max();
    for (int i = 0; i < party_size; ++i) {
        min_speed = std::min(min_speed, static_cast<float>(rng.lognormal(kMemberSpeedLogMu, kMemberSpeedLogSigma)));
    }
    return min_speed;
}

int random_idle_node_idx(Rng& rng, int from_idx) {
    const auto& candidates = gd::kIdleNeighborNodeIdx[from_idx];
    if (candidates.empty()) {
        return from_idx;
    }
    return candidates[rng.randint(0, static_cast<int>(candidates.size()))];
}

int route_one(
    int party_id,
    int now_sec,
    const PartyArrays& parties,
    const std::array<bool, kNumRides>& open_mask,
    const std::array<float, kNumRides>& wait_times,
    const std::array<int, kNumRides>& durations,
    double rand_u01) {
    if (now_sec >= parties.leave_sec[party_id]) {
        return kExitRideId;
    }

    const int remaining = parties.leave_sec[party_id] - now_sec;
    const int node_idx = parties.location_node_idx[party_id];
    float speed = parties.effective_speed[party_id];
    if (speed < 0.1f) {
        speed = 0.1f;
    }
    const double scale = kBaseWalkingSpeed / speed;
    const int current_ride = gd::kNodeIdxToRide[node_idx];

    for (int k = 0; k < kNumRides; ++k) {
        const int ride_id = parties.preference_order[party_id][k];
        if (current_ride >= 0 && ride_id == current_ride) {
            continue;
        }
        if (!open_mask[ride_id]) {
            continue;
        }
        const int walk = std::max(1, static_cast<int>(std::ceil(gd::kBaseWalkToRides[node_idx][ride_id] * scale)));
        if (walk + static_cast<int>(wait_times[ride_id]) + durations[ride_id] > remaining) {
            continue;
        }
        if (wait_times[ride_id] <= parties.balk_sec[party_id][ride_id]) {
            return ride_id;
        }
    }

    if (rand_u01 < kIdleWalkProb) {
        return kRouteIdleCode;
    }

    for (int k = 0; k < kNumRides; ++k) {
        const int ride_id = parties.preference_order[party_id][k];
        if (current_ride >= 0 && ride_id == current_ride) {
            continue;
        }
        if (!open_mask[ride_id]) {
            continue;
        }
        const int walk = std::max(1, static_cast<int>(std::ceil(gd::kBaseWalkToRides[node_idx][ride_id] * scale)));
        if (walk + static_cast<int>(wait_times[ride_id]) + durations[ride_id] > remaining) {
            continue;
        }
        return ride_id;
    }

    return kExitRideId;
}

void route_batch(
    const std::vector<int32_t>& party_ids,
    int now_sec,
    PartyArrays& parties,
    const std::array<bool, kNumRides>& open_mask,
    const std::array<float, kNumRides>& wait_times,
    const std::array<int, kNumRides>& durations,
    Rng& rng,
    const std::function<void(int, int)>& assign_route) {
    for (size_t start = 0; start < party_ids.size(); start += kMaxRouteBatch) {
        const size_t end = std::min(party_ids.size(), start + kMaxRouteBatch);
        for (size_t i = start; i < end; ++i) {
            const int pid = party_ids[i];
            const int target = route_one(pid, now_sec, parties, open_mask, wait_times, durations, rng.uniform01());
            assign_route(pid, target);
        }
    }
}

class Simulator {
public:
    explicit Simulator(uint64_t seed) : rng_(seed) {
        for (int i = 0; i < kNumRides; ++i) {
            rides_[i].status = RideStatus::Open;
        }
    }

    DayMetricsResult run() {
        const auto t0 = std::chrono::steady_clock::now();
        reset();
        spawn_day();
        metrics_.total_parties = parties_.count;
        metrics_.total_guests = total_guests_;

        for (const auto& [spawn_sec, party_id] : spawn_schedule_) {
            wheel_.schedule(spawn_sec, Event{EventType::PartySpawn, party_id, -1, 0});
        }

        while (!wheel_.empty()) {
            auto [now_sec, events] = wheel_.pop_next();
            if (now_sec > kDaySeconds) {
                break;
            }

            update_wait_estimates(now_sec);
            maybe_sample(now_sec);
            maybe_record_ride_sample(now_sec);

            for (int ride_id = 0; ride_id < kNumRides; ++ride_id) {
                auto route_now = maybe_breakdown(ride_id, now_sec);
                if (route_now.has_value()) {
                    metrics_.breakdown_count += 1;
                    on_breakdown(ride_id, *route_now, now_sec);
                }
            }

            std::vector<int32_t> deciding;
            deciding.reserve(events.size());

            for (const auto& event : events) {
                switch (event.type) {
                    case EventType::PartySpawn:
                        parties_.location_node_idx[event.party_id] = gd::kEntranceNodeIdx;
                        deciding.push_back(event.party_id);
                        break;
                    case EventType::ArriveAtDestination:
                        append_deciding(handle_arrive(event.party_id, now_sec), deciding);
                        break;
                    case EventType::RideStart:
                        handle_ride_start(event, now_sec);
                        break;
                    case EventType::RideComplete:
                        deciding.push_back(event.party_id);
                        handle_ride_complete(event.party_id, event.ride_id, now_sec);
                        break;
                    case EventType::BreakdownEnd:
                        on_breakdown_end(event.ride_id, event.ride_generation, now_sec);
                        break;
                    case EventType::EvacuateParty:
                        append_deciding(handle_evacuate(event, now_sec), deciding);
                        break;
                }
            }

            if (!deciding.empty()) {
                route_parties(deciding, now_sec);
            }
        }

        finalize_recording();

        const auto t1 = std::chrono::steady_clock::now();
        metrics_.wall_time_sec =
            std::chrono::duration<double>(t1 - t0).count();
        if (recording_ != nullptr) {
            recording_->metrics = metrics_;
        }
        return metrics_;
    }

    void set_bc_recorder(std::vector<BCSample>* out) { bc_out_ = out; }

    void set_recording(DayRecording* out, int sample_interval_sec = 60) {
        recording_ = out;
        viz_sample_interval_sec_ = std::max(1, sample_interval_sec);
        next_viz_sample_sec_ = 0;
    }

    void env_begin(uint64_t seed) {
        hold_routing_ = true;
        env_done_ = false;
        env_now_sec_ = 0;
        env_queue_.clear();
        env_queue_pos_ = 0;
        last_var_sample_count_ = 0;
        rng_ = Rng(seed);
        reset();
        spawn_day();
        metrics_.total_parties = parties_.count;
        metrics_.total_guests = total_guests_;
        for (const auto& [spawn_sec, party_id] : spawn_schedule_) {
            wheel_.schedule(spawn_sec, Event{EventType::PartySpawn, party_id, -1, 0});
        }
    }

    bool env_pump() {
        while (env_queue_pos_ >= env_queue_.size()) {
            env_queue_.clear();
            env_queue_pos_ = 0;
            if (env_done_ || wheel_.empty()) {
                env_done_ = true;
                return false;
            }
            if (!env_tick()) {
                env_done_ = true;
                return false;
            }
        }
        return true;
    }

    int env_current_party() const {
        return env_queue_[env_queue_pos_];
    }

    int env_now_sec() const { return env_now_sec_; }

    Observation env_build_obs(int party_id) const {
        return build_observation(party_id, env_now_sec_);
    }

    void env_apply_action(int action) {
        assign_route(env_current_party(), target_from_action(action), env_now_sec_);
        ++env_queue_pos_;
    }

    float env_reward_delta() {
        const size_t n = metrics_.wait_variance_samples.size();
        if (n <= last_var_sample_count_) {
            return -0.001f;
        }
        const double var = metrics_.wait_variance_samples.back();
        last_var_sample_count_ = n;
        return static_cast<float>(-var / 1'000'000.0);
    }

    DayMetricsResult env_finalize() {
        metrics_.wall_time_sec = 0.0;
        return metrics_;
    }

    bool env_is_done() const { return env_done_ && env_queue_pos_ >= env_queue_.size(); }

    bool env_ensure_routing_ready() {
        while (env_queue_pos_ >= env_queue_.size()) {
            env_queue_.clear();
            env_queue_pos_ = 0;
            if (env_done_ || wheel_.empty()) {
                env_done_ = true;
                return false;
            }
            if (!env_tick()) {
                env_done_ = true;
                return false;
            }
        }
        return true;
    }

    int env_peek_obs_batch(const int max_batch, std::vector<float>& flat_out) {
        if (max_batch <= 0 || env_queue_pos_ >= env_queue_.size()) {
            return 0;
        }
        const int available = static_cast<int>(env_queue_.size() - env_queue_pos_);
        const int n = std::min(max_batch, available);
        flat_out.reserve(flat_out.size() + static_cast<size_t>(n) * kFlatObsDim);
        for (int i = 0; i < n; ++i) {
            const int party_id = env_queue_[env_queue_pos_ + static_cast<size_t>(i)];
            const auto flat = build_observation(party_id, env_now_sec_).flat();
            flat_out.insert(flat_out.end(), flat.begin(), flat.end());
        }
        return n;
    }

    bool env_apply_actions(
        const std::vector<int>& actions,
        std::vector<float>& rewards,
        bool& done,
        DayMetricsResult& metrics) {
        done = false;
        for (int action : actions) {
            rewards.push_back(env_reward_delta());
            env_apply_action(action);
        }
        if (!env_pump()) {
            done = true;
            metrics = env_finalize();
            if (!rewards.empty()) {
                rewards.back() += static_cast<float>(-metrics.avg_wait_variance() / 1000.0);
            }
            return false;
        }
        return true;
    }

private:
    static void append_deciding(const std::vector<int32_t>& src, std::vector<int32_t>& dst) {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    void reset() {
        parties_ = PartyArrays{};
        for (auto& ride : rides_) {
            ride = Ride{};
            ride.status = RideStatus::Open;
        }
        wheel_ = TimingWheel{};
        metrics_ = DayMetricsResult{};
        spawn_schedule_.clear();
        total_guests_ = 0;
        active_walk_idx_.clear();
        next_viz_sample_sec_ = 0;
    }

    void spawn_day() {
        const int total_guests = std::max(1000, static_cast<int>(std::round(rng_.normal(kTotalGuestsMean, kTotalGuestsStd))));

        std::vector<int32_t> sizes;
        std::vector<int32_t> spawns;
        std::vector<int32_t> leaves;
        std::vector<float> speeds;
        std::vector<std::array<float, kNumRides>> pref_rows;
        std::vector<std::array<uint8_t, kNumRides>> must_do_rows;

        int party_id = 0;
        int guests_assigned = 0;

        while (guests_assigned < total_guests) {
            const int size = std::max(1, static_cast<int>(std::round(rng_.normal(kPartySizeMean, kPartySizeStd))));
            guests_assigned += size;

            int spawn_sec = static_cast<int>(std::round(rng_.normal(kSpawnMeanSec, kSpawnStdSec)));
            spawn_sec = std::clamp(spawn_sec, 0, kDaySeconds - kMinDwellSec);

            int dwell = static_cast<int>(std::round(rng_.normal(kDwellMeanSec, kDwellStdSec)));
            dwell = std::max(kMinDwellSec, dwell);
            const int leave_sec = std::min(kDaySeconds, spawn_sec + dwell);

            const int must_do_count = rng_.randint(0, 5);
            std::array<uint8_t, kNumRides> must_do{};
            if (must_do_count > 0) {
                std::array<int, kNumRides> rides{};
                std::iota(rides.begin(), rides.end(), 0);
                std::shuffle(rides.begin(), rides.end(), rng_.engine());
                for (int i = 0; i < must_do_count; ++i) {
                    must_do[rides[i]] = 1;
                }
            }

            std::array<float, kNumRides> prefs{};
            for (int i = 0; i < kNumRides; ++i) {
                prefs[i] = static_cast<float>(rng_.uniform01() * 0.9 + 0.1);
                if (must_do[i]) {
                    prefs[i] *= static_cast<float>(kMustDoPrefBoost);
                }
            }
            float sum = 0.0f;
            for (float p : prefs) {
                sum += p;
            }
            if (sum > 0.0f) {
                for (float& p : prefs) {
                    p /= sum;
                }
            }

            sizes.push_back(size);
            spawns.push_back(spawn_sec);
            leaves.push_back(leave_sec);
            speeds.push_back(party_effective_speed(rng_, size));
            pref_rows.push_back(prefs);
            must_do_rows.push_back(must_do);
            spawn_schedule_.emplace_back(spawn_sec, party_id);
            ++party_id;
        }

        const int n = party_id;
        parties_.count = n;
        parties_.party_size = std::move(sizes);
        parties_.spawn_sec = std::move(spawns);
        parties_.leave_sec = std::move(leaves);
        parties_.location_node_idx.assign(n, gd::kEntranceNodeIdx);
        parties_.effective_speed = std::move(speeds);
        parties_.state.assign(n, static_cast<int8_t>(PartyState::Walking));
        parties_.target_ride_id.assign(n, kRouteIdleCode);
        parties_.target_node_idx.assign(n, gd::kEntranceNodeIdx);
        parties_.walk_target_ride.assign(n, -1);
        parties_.preferences = std::move(pref_rows);
        parties_.must_do_remaining = std::move(must_do_rows);
        parties_.ride_history.assign(n, {});
        parties_.rides_completed.assign(n, 0);
        parties_.preference_order.resize(n);
        parties_.balk_sec.resize(n);
        active_walk_idx_.assign(n, -1);

        for (int i = 0; i < n; ++i) {
            compute_preference_order(parties_.preferences[i], parties_.must_do_remaining[i], parties_.preference_order[i]);
            compute_balk_sec(parties_.preferences[i], parties_.balk_sec[i]);
        }

        total_guests_ = guests_assigned;
    }

    void update_wait_estimates(int now_sec) {
        for (int i = 0; i < kNumRides; ++i) {
            auto& ride = rides_[i];
            if (ride.status == RideStatus::Broken) {
                ride.current_wait = 9999.0;
            } else if (gd::kRideCapacityPerSec[i] <= 0.0) {
                ride.current_wait = 9999.0;
            } else {
                const int ahead = static_cast<int>(ride.pending_board.size()) + static_cast<int>(ride.on_ride.size());
                const double until_board = std::max(0.0, ride.next_board_sec - now_sec);
                ride.current_wait = until_board + ahead / gd::kRideCapacityPerSec[i];
            }
            open_mask_[i] = ride.status == RideStatus::Open;
            wait_arr_[i] = static_cast<float>(ride.current_wait);
        }
    }

    void maybe_sample(int now_sec) {
        if (now_sec < next_sample_sec_) {
            return;
        }
        if (now_sec > kDaySeconds) {
            return;
        }

        std::vector<double> valid;
        valid.reserve(kNumRides);
        for (float w : wait_arr_) {
            if (w < 9000.0f) {
                valid.push_back(w);
            }
        }
        if (!valid.empty()) {
            const double mean = std::accumulate(valid.begin(), valid.end(), 0.0) / valid.size();
            double var = 0.0;
            for (double w : valid) {
                const double d = w - mean;
                var += d * d;
            }
            var /= valid.size();
            metrics_.mean_wait_samples.push_back(mean);
            metrics_.wait_variance_samples.push_back(var);
        }
        next_sample_sec_ = now_sec + kMetricsSampleIntervalSec;
    }

    std::optional<std::vector<int>> maybe_breakdown(int ride_id, int now_sec) {
        auto& ride = rides_[ride_id];
        if (ride.status == RideStatus::Broken) {
            return std::nullopt;
        }
        if (rng_.uniform01() >= gd::kRideBreakdownProbSec[ride_id]) {
            return std::nullopt;
        }
        return trigger_breakdown(ride_id, now_sec);
    }

    std::vector<int> trigger_breakdown(int ride_id, int now_sec) {
        auto& ride = rides_[ride_id];
        if (ride.status == RideStatus::Broken) {
            return {};
        }

        ride.status = RideStatus::Broken;
        const int repair = rng_.randint(kBreakdownRepairMinSec, kBreakdownRepairMaxSec + 1);
        ride.broken_until = now_sec + repair;
        ride.generation += 1;
        open_mask_[ride_id] = false;
        wait_arr_[ride_id] = 9999.0f;

        wheel_.schedule(ride.broken_until, Event{EventType::BreakdownEnd, -1, static_cast<int16_t>(ride_id), static_cast<int16_t>(ride.generation)});

        std::vector<int> route_now;
        for (const auto& [pid, _] : ride.pending_board) {
            (void)_;
            route_now.push_back(pid);
            ride.evacuation.push_back(pid);
        }
        ride.pending_board.clear();

        for (int pid : ride.on_ride) {
            if (std::find(ride.evacuating_on_ride.begin(), ride.evacuating_on_ride.end(), pid) == ride.evacuating_on_ride.end()) {
                ride.evacuating_on_ride.push_back(pid);
            }
        }

        if (!ride.evacuation_active && (!ride.evacuation.empty() || !ride.evacuating_on_ride.empty())) {
            ride.evacuation_active = true;
            wheel_.schedule(now_sec + kEvacIntervalSec, Event{EventType::EvacuateParty, -1, static_cast<int16_t>(ride_id), static_cast<int16_t>(ride.generation)});
        }

        return route_now;
    }

    void on_breakdown(int ride_id, const std::vector<int>& route_at_entrance, int now_sec) {
        const int ride_node_idx = gd::kRideNodeIdx[ride_id];
        std::vector<int32_t> walkers;
        for (int pid = 0; pid < parties_.count; ++pid) {
            if (parties_.walk_target_ride[pid] == ride_id) {
                cancel_walk(pid, now_sec, false);
                walkers.push_back(pid);
            }
        }
        if (!walkers.empty()) {
            route_parties(walkers, now_sec);
        }

        if (!route_at_entrance.empty()) {
            for (int pid : route_at_entrance) {
                parties_.location_node_idx[pid] = ride_node_idx;
                parties_.state[pid] = static_cast<int8_t>(PartyState::Evacuating);
            }
            route_parties(to_i32(route_at_entrance), now_sec);
        }
    }

    void cancel_walk(int party_id, int now_sec, bool completed) {
        const int target_ride = parties_.walk_target_ride[party_id];
        if (target_ride >= 0) {
            rides_[target_ride].incoming = std::max(0, rides_[target_ride].incoming - 1);
            parties_.walk_target_ride[party_id] = -1;
        }
        if (recording_ != nullptr) {
            const int walk_idx = active_walk_idx_[party_id];
            if (walk_idx >= 0) {
                auto& walk = recording_->walks[static_cast<size_t>(walk_idx)];
                walk.end_sec = now_sec;
                walk.cancelled = completed ? 0 : 1;
                active_walk_idx_[party_id] = -1;
            }
        }
    }

    std::vector<int32_t> handle_arrive(int party_id, int now_sec) {
        cancel_walk(party_id, now_sec, true);
        parties_.location_node_idx[party_id] = parties_.target_node_idx[party_id];

        const int target_ride = parties_.target_ride_id[party_id];
        if (target_ride == kExitRideId) {
            parties_.state[party_id] = static_cast<int8_t>(PartyState::Exited);
            metrics_.parties_exited += 1;
            return {};
        }
        if (target_ride == kRouteIdleCode) {
            return {party_id};
        }
        if (rides_[target_ride].status == RideStatus::Broken) {
            return {party_id};
        }

        parties_.state[party_id] = static_cast<int8_t>(PartyState::InQueue);
        schedule_boarding(target_ride, party_id, now_sec);
        return {};
    }

    void handle_ride_start(const Event& event, int now_sec) {
        auto& ride = rides_[event.ride_id];
        if (event.ride_generation != ride.generation) {
            return;
        }
        if (ride.pending_board.find(event.party_id) == ride.pending_board.end()) {
            return;
        }
        if (ride.status != RideStatus::Open) {
            return;
        }
        ride.pending_board.erase(event.party_id);
        ride.on_ride.push_back(event.party_id);
        parties_.state[event.party_id] = static_cast<int8_t>(PartyState::OnRide);
        wheel_.schedule(
            now_sec + gd::kRideDurationSec[event.ride_id],
            Event{EventType::RideComplete, event.party_id, event.ride_id, 0});
    }

    void handle_ride_complete(int party_id, int ride_id, int now_sec) {
        auto& ride = rides_[ride_id];
        ride.on_ride.erase(std::remove(ride.on_ride.begin(), ride.on_ride.end(), party_id), ride.on_ride.end());

        parties_.ride_history[party_id][ride_id] += 1;
        parties_.rides_completed[party_id] += 1;
        if (recording_ != nullptr) {
            recording_->ride_completions.push_back(
                PartyRideEvent{party_id, now_sec, static_cast<int16_t>(ride_id)});
        }
        if (parties_.must_do_remaining[party_id][ride_id]) {
            parties_.must_do_remaining[party_id][ride_id] = 0;
            compute_preference_order(
                parties_.preferences[party_id], parties_.must_do_remaining[party_id], parties_.preference_order[party_id]);
            compute_balk_sec(parties_.preferences[party_id], parties_.balk_sec[party_id]);
        }

        metrics_.rides_completed += 1;
        parties_.location_node_idx[party_id] = gd::kRideNodeIdx[ride_id];
        parties_.state[party_id] = static_cast<int8_t>(PartyState::Walking);
    }

    void on_breakdown_end(int ride_id, int generation, int now_sec) {
        auto& ride = rides_[ride_id];
        if (generation != ride.generation || ride.status != RideStatus::Broken) {
            return;
        }
        ride.status = RideStatus::Open;
        ride.next_board_sec = static_cast<double>(now_sec);
        open_mask_[ride_id] = true;
    }

    std::vector<int32_t> handle_evacuate(const Event& event, int now_sec) {
        auto& ride = rides_[event.ride_id];
        if (event.ride_generation != ride.generation) {
            return {};
        }

        int pid = -1;
        if (!ride.evacuation.empty()) {
            pid = ride.evacuation.front();
            ride.evacuation.erase(ride.evacuation.begin());
        } else if (!ride.evacuating_on_ride.empty()) {
            pid = ride.evacuating_on_ride.front();
            ride.evacuating_on_ride.erase(ride.evacuating_on_ride.begin());
            ride.on_ride.erase(std::remove(ride.on_ride.begin(), ride.on_ride.end(), pid), ride.on_ride.end());
        } else {
            ride.evacuation_active = false;
            return {};
        }

        parties_.location_node_idx[pid] = gd::kRideNodeIdx[event.ride_id];
        parties_.state[pid] = static_cast<int8_t>(PartyState::Walking);

        if (!ride.evacuation.empty() || !ride.evacuating_on_ride.empty()) {
            wheel_.schedule(
                now_sec + kEvacIntervalSec,
                Event{EventType::EvacuateParty, -1, event.ride_id, static_cast<int16_t>(ride.generation)});
        } else {
            ride.evacuation_active = false;
        }

        return {pid};
    }

    void schedule_boarding(int ride_id, int party_id, int now_sec) {
        auto& ride = rides_[ride_id];
        if (ride.status != RideStatus::Open) {
            return;
        }
        int start_sec = std::max(now_sec, static_cast<int>(std::ceil(ride.next_board_sec)));
        if (start_sec <= now_sec) {
            start_sec = now_sec + 1;
        }
        ride.next_board_sec = start_sec + 1.0 / gd::kRideCapacityPerSec[ride_id];
        ride.pending_board[party_id] = start_sec;
        wheel_.schedule(
            start_sec,
            Event{EventType::RideStart, party_id, static_cast<int16_t>(ride_id), static_cast<int16_t>(ride.generation)});
    }

    static std::vector<int32_t> to_i32(const std::vector<int>& src) {
        return std::vector<int32_t>(src.begin(), src.end());
    }

    void route_parties(const std::vector<int32_t>& party_ids, int now_sec) {
        if (party_ids.empty()) {
            return;
        }

        std::vector<int32_t> ids = party_ids;
        std::sort(ids.begin(), ids.end());
        ids.erase(
            std::remove_if(ids.begin(), ids.end(), [&](int pid) {
                return parties_.state[pid] == static_cast<int8_t>(PartyState::Exited);
            }),
            ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (ids.empty()) {
            return;
        }

        if (bc_out_ != nullptr) {
            for (int pid : ids) {
                const int target = route_one(
                    pid, now_sec, parties_, open_mask_, wait_arr_, duration_arr_, rng_.uniform01());
                bc_out_->push_back({build_observation(pid, now_sec), action_from_target(target)});
                assign_route(pid, target, now_sec);
            }
            return;
        }

        if (hold_routing_) {
            env_queue_ = std::move(ids);
            env_queue_pos_ = 0;
            env_now_sec_ = now_sec;
            return;
        }

        route_batch(
            ids,
            now_sec,
            parties_,
            open_mask_,
            wait_arr_,
            duration_arr_,
            rng_,
            [&](int party_id, int target) { assign_route(party_id, target, now_sec); });
    }

    void assign_route(int party_id, int target, int now_sec) {
        if (parties_.state[party_id] == static_cast<int8_t>(PartyState::Exited)) {
            return;
        }

        cancel_walk(party_id, now_sec, false);
        const int from_idx = parties_.location_node_idx[party_id];
        const float speed = parties_.effective_speed[party_id];

        if (target == kExitRideId) {
            const int dest_idx = gd::kEntranceNodeIdx;
            const int walk = party_walk_sec(from_idx, dest_idx, speed);
            parties_.target_ride_id[party_id] = kExitRideId;
            parties_.target_node_idx[party_id] = dest_idx;
            parties_.state[party_id] = static_cast<int8_t>(PartyState::Walking);
            record_walk(party_id, now_sec, walk, from_idx, dest_idx, kExitRideId);
            wheel_.schedule(now_sec + walk, Event{EventType::ArriveAtDestination, party_id, -1, 0});
            return;
        }

        if (target == kRouteIdleCode) {
            const int dest_idx = random_idle_node_idx(rng_, from_idx);
            const int walk = party_walk_sec(from_idx, dest_idx, speed);
            parties_.target_ride_id[party_id] = kRouteIdleCode;
            parties_.target_node_idx[party_id] = dest_idx;
            parties_.state[party_id] = static_cast<int8_t>(PartyState::Walking);
            record_walk(party_id, now_sec, walk, from_idx, dest_idx, kRouteIdleCode);
            wheel_.schedule(now_sec + walk, Event{EventType::ArriveAtDestination, party_id, -1, 0});
            return;
        }

        const int ride_id = target;
        const int dest_idx = gd::kRideNodeIdx[ride_id];
        const int walk = party_walk_to_ride_sec(from_idx, ride_id, speed);
        parties_.target_ride_id[party_id] = ride_id;
        parties_.target_node_idx[party_id] = dest_idx;
        parties_.state[party_id] = static_cast<int8_t>(PartyState::Walking);
        parties_.walk_target_ride[party_id] = ride_id;
        rides_[ride_id].incoming += 1;
        record_walk(party_id, now_sec, walk, from_idx, dest_idx, ride_id);
        wheel_.schedule(now_sec + walk, Event{EventType::ArriveAtDestination, party_id, -1, 0});
    }

    bool env_tick() {
        auto [now_sec, events] = wheel_.pop_next();
        if (now_sec > kDaySeconds) {
            return false;
        }
        env_now_sec_ = now_sec;

        update_wait_estimates(now_sec);
        maybe_sample(now_sec);

        for (int ride_id = 0; ride_id < kNumRides; ++ride_id) {
            auto route_now = maybe_breakdown(ride_id, now_sec);
            if (route_now.has_value()) {
                metrics_.breakdown_count += 1;
                on_breakdown(ride_id, *route_now, now_sec);
                if (!env_queue_.empty()) {
                    return true;
                }
            }
        }

        std::vector<int32_t> deciding;
        deciding.reserve(events.size());

        for (const auto& event : events) {
            switch (event.type) {
                case EventType::PartySpawn:
                    parties_.location_node_idx[event.party_id] = gd::kEntranceNodeIdx;
                    deciding.push_back(event.party_id);
                    break;
                case EventType::ArriveAtDestination:
                    append_deciding(handle_arrive(event.party_id, now_sec), deciding);
                    break;
                case EventType::RideStart:
                    handle_ride_start(event, now_sec);
                    break;
                case EventType::RideComplete:
                    deciding.push_back(event.party_id);
                    handle_ride_complete(event.party_id, event.ride_id, now_sec);
                    break;
                case EventType::BreakdownEnd:
                    on_breakdown_end(event.ride_id, event.ride_generation, now_sec);
                    break;
                case EventType::EvacuateParty:
                    append_deciding(handle_evacuate(event, now_sec), deciding);
                    break;
            }
        }

        if (!deciding.empty()) {
            route_parties(deciding, now_sec);
            if (!env_queue_.empty()) {
                return true;
            }
        }
        return !wheel_.empty();
    }

    Observation build_observation(int party_id, int now_sec) const {
        Observation obs{};
        for (int i = 0; i < kNumRides; ++i) {
            obs.guest[static_cast<size_t>(i)] = parties_.preferences[party_id][i];
        }
        obs.guest[35] = parties_.party_size[party_id] / 8.0f;
        obs.guest[36] = parties_.effective_speed[party_id] / 2.0f;
        obs.guest[37] = static_cast<float>(parties_.leave_sec[party_id] - now_sec) / static_cast<float>(kDaySeconds);
        obs.guest[38] = static_cast<float>(parties_.location_node_idx[party_id]) / static_cast<float>(gd::kNumNodes);
        obs.guest[39] = parties_.rides_completed[party_id] / 20.0f;
        int must_count = 0;
        float balk_sum = 0.0f;
        for (int i = 0; i < kNumRides; ++i) {
            must_count += parties_.must_do_remaining[party_id][i];
            balk_sum += parties_.balk_sec[party_id][i];
        }
        obs.guest[40] = static_cast<float>(must_count) / 5.0f;
        obs.guest[41] = gd::kNodeIdxToRide[parties_.location_node_idx[party_id]] >= 0 ? 1.0f : 0.0f;
        obs.guest[42] = parties_.state[party_id] / 16.0f;
        obs.guest[43] = balk_sum / static_cast<float>(kNumRides) / 3600.0f;
        obs.guest[44] = parties_.walk_target_ride[party_id] >= 0 ? 1.0f : 0.0f;

        for (int r = 0; r < kNumRides; ++r) {
            const size_t base = static_cast<size_t>(r * kRideDynamicFeatDim);
            obs.ride[base + 0] = std::min(wait_arr_[r], 3600.0f) / 3600.0f;
            obs.ride[base + 1] = static_cast<float>(rides_[r].incoming) / 100.0f;
            obs.ride[base + 2] = open_mask_[r] ? 1.0f : 0.0f;
            obs.ride[base + 3] = static_cast<float>(duration_arr_[r]) / 900.0f;
            obs.ride[base + 4] = static_cast<float>(gd::kRideCapacityPerSec[r]);
        }

        double mean = 0.0;
        double var = 0.0;
        int broken = 0;
        int valid = 0;
        for (int r = 0; r < kNumRides; ++r) {
            if (!open_mask_[r]) {
                ++broken;
            }
            if (wait_arr_[r] < 9000.0f) {
                mean += wait_arr_[r];
                ++valid;
            }
        }
        if (valid > 0) {
            mean /= valid;
            for (int r = 0; r < kNumRides; ++r) {
                if (wait_arr_[r] < 9000.0f) {
                    const double d = wait_arr_[r] - mean;
                    var += d * d;
                }
            }
            var /= valid;
        }
        obs.env[0] = static_cast<float>(now_sec) / static_cast<float>(kDaySeconds);
        obs.env[1] = static_cast<float>(mean / 3600.0);
        obs.env[2] = static_cast<float>(var / 1'000'000.0);
        obs.env[3] = static_cast<float>(broken) / static_cast<float>(kNumRides);
        return obs;
    }

    void record_walk(int party_id, int now_sec, int walk_sec, int from_idx, int to_idx, int target) {
        if (recording_ == nullptr) {
            return;
        }
        WalkRecord rec{};
        rec.party_id = party_id;
        rec.start_sec = now_sec;
        rec.end_sec = now_sec + walk_sec;
        rec.planned_end_sec = now_sec + walk_sec;
        rec.from_idx = static_cast<int16_t>(from_idx);
        rec.to_idx = static_cast<int16_t>(to_idx);
        rec.target_ride = static_cast<int16_t>(target);
        rec.cancelled = 0;
        active_walk_idx_[party_id] = static_cast<int>(recording_->walks.size());
        recording_->walks.push_back(rec);
    }

    void maybe_record_ride_sample(int now_sec) {
        if (recording_ == nullptr) {
            return;
        }
        if (now_sec < next_viz_sample_sec_) {
            return;
        }
        RideSample sample{};
        sample.sec = now_sec;
        for (int i = 0; i < kNumRides; ++i) {
            sample.wait[static_cast<size_t>(i)] = wait_arr_[i];
            sample.broken[static_cast<size_t>(i)] =
                rides_[i].status == RideStatus::Broken ? 1 : 0;
            sample.queue_len[static_cast<size_t>(i)] =
                static_cast<int32_t>(rides_[i].pending_board.size());
        }
        recording_->ride_samples.push_back(sample);
        next_viz_sample_sec_ = now_sec + viz_sample_interval_sec_;
    }

    void finalize_recording() {
        if (recording_ == nullptr) {
            return;
        }
        recording_->parties.clear();
        recording_->parties.reserve(static_cast<size_t>(parties_.count));
        for (int pid = 0; pid < parties_.count; ++pid) {
            PartyInfo info{};
            info.party_id = pid;
            info.size = parties_.party_size[pid];
            info.spawn_sec = parties_.spawn_sec[pid];
            info.leave_sec = parties_.leave_sec[pid];
            info.rides_completed = parties_.rides_completed[pid];
            recording_->parties.push_back(info);
        }
        // Close any still-active walks at day end.
        for (int pid = 0; pid < parties_.count; ++pid) {
            const int walk_idx = active_walk_idx_[pid];
            if (walk_idx >= 0) {
                auto& walk = recording_->walks[static_cast<size_t>(walk_idx)];
                walk.end_sec = kDaySeconds;
                walk.cancelled = 1;
                active_walk_idx_[pid] = -1;
            }
        }
    }

    Rng rng_;
    PartyArrays parties_;
    std::array<Ride, kNumRides> rides_{};
    TimingWheel wheel_;
    DayMetricsResult metrics_;
    std::vector<std::pair<int, int>> spawn_schedule_;
    int total_guests_ = 0;
    int next_sample_sec_ = 0;
    std::array<bool, kNumRides> open_mask_{};
    std::array<float, kNumRides> wait_arr_{};
    std::array<int, kNumRides> duration_arr_ = [] {
        std::array<int, kNumRides> arr{};
        for (int i = 0; i < kNumRides; ++i) {
            arr[i] = gd::kRideDurationSec[i];
        }
        return arr;
    }();

    bool hold_routing_ = false;
    bool env_done_ = false;
    int env_now_sec_ = 0;
    std::vector<int32_t> env_queue_;
    size_t env_queue_pos_ = 0;
    size_t last_var_sample_count_ = 0;
    std::vector<BCSample>* bc_out_ = nullptr;

    DayRecording* recording_ = nullptr;
    int viz_sample_interval_sec_ = 60;
    int next_viz_sample_sec_ = 0;
    std::vector<int> active_walk_idx_;
};

}  // namespace detail

std::array<float, kFlatObsDim> Observation::flat() const {
    std::array<float, kFlatObsDim> out{};
    size_t idx = 0;
    for (float v : guest) {
        out[idx++] = v;
    }
    for (float v : ride) {
        out[idx++] = v;
    }
    for (float v : env) {
        out[idx++] = v;
    }
    return out;
}

int action_from_target(int target_ride_id) {
    if (target_ride_id == kExitRideId) {
        return kNumRides;
    }
    if (target_ride_id == kRouteIdleCode) {
        return kNumRides + 1;
    }
    return target_ride_id;
}

int target_from_action(int action) {
    if (action == kNumRides) {
        return kExitRideId;
    }
    if (action == kNumRides + 1) {
        return kRouteIdleCode;
    }
    return action;
}

std::vector<BCSample> collect_bc_dataset(const int num_days, const uint64_t seed_start) {
    std::vector<BCSample> samples;
    samples.reserve(static_cast<size_t>(num_days) * 50000);
    for (int day = 0; day < num_days; ++day) {
        detail::Simulator sim(seed_start + static_cast<uint64_t>(day));
        sim.set_bc_recorder(&samples);
        sim.run();
    }
    return samples;
}

struct ParkEnv::Impl {
    std::unique_ptr<detail::Simulator> sim;
    uint64_t seed = 0;
    float episode_reward = 0.0f;
};

ParkEnv::ParkEnv(const uint64_t seed) : impl_(new Impl()) {
    impl_->seed = seed;
    impl_->sim = std::make_unique<detail::Simulator>(seed);
}

ParkEnv::~ParkEnv() {
    delete impl_;
}

ParkEnv::ParkEnv(ParkEnv&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

ParkEnv& ParkEnv::operator=(ParkEnv&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

Observation ParkEnv::reset(const uint64_t seed) {
    impl_->seed = seed;
    impl_->episode_reward = 0.0f;
    impl_->sim = std::make_unique<detail::Simulator>(seed);
    impl_->sim->env_begin(seed);
    impl_->sim->env_pump();
    return impl_->sim->env_build_obs(impl_->sim->env_current_party());
}

EnvStepResult ParkEnv::step(const int action) {
    EnvStepResult result{};
    result.reward = impl_->sim->env_reward_delta();
    impl_->episode_reward += result.reward;
    impl_->sim->env_apply_action(action);

    if (impl_->sim->env_pump()) {
        result.has_obs = true;
        result.obs = impl_->sim->env_build_obs(impl_->sim->env_current_party());
        return result;
    }

    result.done = true;
    result.has_obs = false;
    result.metrics = impl_->sim->env_finalize();
    result.reward += static_cast<float>(-result.metrics.avg_wait_variance() / 1000.0);
    impl_->episode_reward += static_cast<float>(-result.metrics.avg_wait_variance() / 1000.0);
    return result;
}

RolloutBatchResult ParkEnv::exchange_batch(const std::vector<int>& actions, const int max_obs) {
    RolloutBatchResult result{};
    if (!actions.empty()) {
        impl_->sim->env_apply_actions(actions, result.rewards, result.episode_done, result.metrics);
        result.n_rewards = static_cast<int>(result.rewards.size());
        for (float reward : result.rewards) {
            impl_->episode_reward += reward;
        }
        if (result.episode_done) {
            return result;
        }
    }

    if (max_obs <= 0) {
        return result;
    }

    if (!impl_->sim->env_ensure_routing_ready()) {
        result.episode_done = true;
        if (result.n_rewards == 0) {
            result.metrics = impl_->sim->env_finalize();
        }
        return result;
    }

    result.n_obs = impl_->sim->env_peek_obs_batch(max_obs, result.obs);
    return result;
}

DayMetricsResult run_day(const uint64_t seed) {
    detail::Simulator sim(seed);
    return sim.run();
}

DayRecording record_day(const uint64_t seed, const int sample_interval_sec) {
    DayRecording recording;
    detail::Simulator sim(seed);
    sim.set_recording(&recording, sample_interval_sec);
    sim.run();
    return recording;
}

}  // namespace park
