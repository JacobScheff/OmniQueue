#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace park {

constexpr int kNumRides = 34;
constexpr int kDaySeconds = 54000;
constexpr int kCloseDrainSec = 3 * 3600;  // post-close queue/ride drain window
constexpr int kSimHorizonSec = kDaySeconds + kCloseDrainSec;

constexpr int kExitRideId = -1;
constexpr int kRouteIdleCode = -2;

constexpr int kMaxRouteBatch = 256;
constexpr int kEvacIntervalSec = 4;
constexpr int kBreakdownRepairMinSec = 15 * 60;
constexpr int kBreakdownRepairMaxSec = 60 * 60;
constexpr int kMetricsSampleIntervalSec = 300;
constexpr int kMinDwellSec = 2 * 3600;

constexpr int kGuestFeatDim = 45;
constexpr int kRideDynamicFeatDim = 5;
constexpr int kEnvDynamicFeatDim = 4;
constexpr int kNumActions = 36;  // 34 rides + exit + idle
constexpr int kFlatObsDim = kGuestFeatDim + kNumRides * kRideDynamicFeatDim + kEnvDynamicFeatDim;

constexpr double kTotalGuestsMean = 50000.0;
constexpr double kTotalGuestsStd = 2500.0;
constexpr double kPartySizeMean = 3.2;
constexpr double kPartySizeStd = 1.0;
constexpr double kSpawnRushFraction = 0.50;
constexpr double kSpawnRushMeanSec = 10.0 * 60.0;
constexpr double kSpawnRushStdSec = 15.0 * 60.0;
constexpr int kSpawnRushClampSec = 2 * 3600;
constexpr double kSpawnDayMeanSec = 5.5 * 3600.0;
constexpr double kSpawnDayStdSec = 3.5 * 3600.0;
constexpr double kDwellMeanSec = 12.0 * 3600.0;
constexpr double kDwellStdSec = 2.5 * 3600.0;

constexpr double kBaseWalkingSpeed = 1.4;
constexpr double kMemberSpeedLogMu = 0.3364722366212129;  // log(1.4)
constexpr double kMemberSpeedLogSigma = 0.25;

// Near-shortest walk randomization (mirrored from config.py). Tables live in graph_data.hpp.
// Softmax P_i ∝ exp(-(sec_i - sec_min) / tau); routing feasibility still uses shortest.

constexpr double kBaseBalkSec = 40.0 * 60.0;  // 40 min floor (mirrored from config.py)
constexpr double kBalkScale = 5.0 * 60.0;     // +0–5 min by preference^exp (max ~45 min)
constexpr double kBalkPrefExp = 1.5;
constexpr double kMustDoPrefBoost = 10.0;
constexpr double kIdleWalkProb = 0.5;

// Heuristic ride-repeat dampening (mirrored from config.py)
constexpr int kRepeatTopK = 3;                    // Pass 2: top-K preference ranks
constexpr double kRepeatPrefThreshold = 0.04;     // Pass 2: high-pref mass floor
constexpr double kRepeatPrefScale = 2.0;          // max_repeats = 1 + floor(scale * pref * N)
constexpr int kRepeatMax = 3;                     // hard cap on Pass 2 completions
constexpr double kRepeatBalkFactor = 1.0;         // optional tighter balk on repeats
constexpr double kShortWaitSec = 12.0 * 60.0;     // Pass 3 absolute short-wait bar
constexpr double kShortWaitSlackSec = 2.0 * 60.0; // Pass 3 relative-to-best slack

// PPO reward shaping (mirrored from config.py PPO_* reward knobs)
constexpr float kWaitVarStepCoef = 0.002f;  // dense: -coef * var/1e6 every routing step
constexpr float kPrefRewardScale = 0.01f;
constexpr float kMustDoCompletionBonus = 0.005f;
constexpr float kUnfulfilledMustDoPenalty = 0.002f;
constexpr float kRoutingStepPenalty = 0.001f;  // fallback when no valid wait samples

enum class EventType : uint8_t {
    PartySpawn = 0,
    ArriveAtDestination = 1,
    RideStart = 2,
    RideComplete = 3,
    BreakdownEnd = 4,
    EvacuateParty = 5,
};

enum class PartyState : int8_t {
    Walking = 1,
    InQueue = 2,
    OnRide = 4,
    Evacuating = 8,
    Exited = 16,
};

enum class RideStatus : uint8_t { Open = 0, Broken = 1 };

struct Event {
    EventType type{};
    int32_t party_id = -1;
    int16_t ride_id = -1;
    int16_t ride_generation = 0;
};

struct DayMetricsResult {
    int total_parties = 0;
    int total_guests = 0;
    int rides_completed = 0;
    int parties_exited = 0;
    int breakdown_count = 0;
    std::vector<double> wait_variance_samples;
    std::vector<double> mean_wait_samples;
    double wall_time_sec = 0.0;

    double rides_per_party() const {
        return static_cast<double>(rides_completed) / std::max(1, total_parties);
    }

    double avg_wait_variance() const {
        if (wait_variance_samples.empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : wait_variance_samples) {
            sum += v;
        }
        return sum / static_cast<double>(wait_variance_samples.size());
    }
};

struct Observation {
    std::array<float, kGuestFeatDim> guest{};
    std::array<float, kNumRides * kRideDynamicFeatDim> ride{};
    std::array<float, kEnvDynamicFeatDim> env{};

    std::array<float, kFlatObsDim> flat() const;
};

struct BCSample {
    Observation obs;
    int action = 0;
};

struct EnvStepResult {
    Observation obs;
    float reward = 0.0f;
    bool done = false;
    bool has_obs = false;
    DayMetricsResult metrics;
};

struct RolloutBatchResult {
    std::vector<float> obs;
    std::vector<float> rewards;
    int n_obs = 0;
    int n_rewards = 0;
    bool episode_done = false;
    DayMetricsResult metrics;
};

/** One party walk segment for visualization replay. */
struct WalkRecord {
    int32_t party_id = -1;
    int32_t start_sec = 0;
    int32_t end_sec = 0;          // actual end (arrival or cancel time)
    int32_t planned_end_sec = 0;  // scheduled arrival (for position interpolation)
    int16_t from_idx = 0;
    int16_t to_idx = 0;
    int16_t target_ride = -1;  // ride id, kExitRideId, or kRouteIdleCode
    int16_t path_variant = 0;  // near-shortest OSM path index (0 = shortest)
    uint8_t cancelled = 0;
};

/** Periodic per-ride snapshot for wait / status display. */
struct RideSample {
    int32_t sec = 0;
    std::array<float, kNumRides> wait{};
    std::array<uint8_t, kNumRides> broken{};
    std::array<int32_t, kNumRides> queue_len{};
};

struct PartyInfo {
    int32_t party_id = 0;
    int32_t size = 0;
    int32_t spawn_sec = 0;
    int32_t leave_sec = 0;
    int32_t rides_completed = 0;
};

struct PartyRideEvent {
    int32_t party_id = -1;
    int32_t sec = 0;
    int16_t ride_id = -1;
};

/** Full-day recording consumed by the Pygame visualizer. */
struct DayRecording {
    DayMetricsResult metrics;
    std::vector<PartyInfo> parties;
    std::vector<WalkRecord> walks;
    std::vector<RideSample> ride_samples;
    std::vector<PartyRideEvent> ride_completions;
};

int action_from_target(int target_ride_id);
int target_from_action(int action);

DayMetricsResult run_day(uint64_t seed);
DayRecording record_day(uint64_t seed, int sample_interval_sec = 60);
std::vector<BCSample> collect_bc_dataset(int num_days, uint64_t seed_start);

/** Deterministic heuristic route helper for unit tests. */
struct RouteOneTestInput {
    int now_sec = 0;
    int leave_sec = kDaySeconds;
    int node_idx = 0;
    float speed = static_cast<float>(kBaseWalkingSpeed);
    std::array<int16_t, kNumRides> preference_order{};
    std::array<float, kNumRides> preferences{};
    std::array<float, kNumRides> balk_sec{};
    std::array<int16_t, kNumRides> ride_history{};
    std::array<bool, kNumRides> open_mask{};
    std::array<float, kNumRides> wait_times{};
    std::array<int, kNumRides> durations{};
    double rand_u01 = 1.0;  // >= idle prob → skip idle in Pass 4
};

int route_one_for_test(const RouteOneTestInput& input);

class ParkEnv {
public:
    explicit ParkEnv(uint64_t seed = 0);
    ~ParkEnv();
    ParkEnv(const ParkEnv&) = delete;
    ParkEnv& operator=(const ParkEnv&) = delete;
    ParkEnv(ParkEnv&& other) noexcept;
    ParkEnv& operator=(ParkEnv&& other) noexcept;

    Observation reset(uint64_t seed);
    EnvStepResult step(int action);
    RolloutBatchResult exchange_batch(const std::vector<int>& actions, int max_obs);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace park
