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

// Guest feats: 0..33 preferences, 34 remaining_pref_mass, 35..44 party state,
// 45 elapsed_since_spawn / DAY_SECONDS.
constexpr int kGuestFeatDim = 46;
// Per-ride dynamic feats (party-relative walk/history/must-do included):
// 0 wait, 1 incoming, 2 open, 3 duration, 4 capacity, 5 walk, 6 history, 7 must_do
constexpr int kRideDynamicFeatDim = 8;
constexpr int kEnvDynamicFeatDim = 4;
constexpr int kNumActions = 36;  // 34 rides + exit + idle
constexpr int kFlatObsDim = kGuestFeatDim + kNumRides * kRideDynamicFeatDim + kEnvDynamicFeatDim;

constexpr double kTotalGuestsMean = 50000.0;
constexpr double kTotalGuestsStd = 2500.0;
constexpr double kPartySizeMean = 3.2;
constexpr double kPartySizeStd = 1.0;
constexpr double kSpawnRushFraction = 0.65;
constexpr double kSpawnRushMeanSec = 8.0 * 60.0;
constexpr double kSpawnRushStdSec = 12.0 * 60.0;
constexpr int kSpawnRushClampSec = 2 * 3600;
constexpr double kSpawnDayMeanSec = 6.0 * 3600.0;
constexpr double kSpawnDayStdSec = 3.5 * 3600.0;
constexpr double kDwellMeanSec = 14.0 * 3600.0;
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
// Default (play/watch/visualize): popularity * U(1±noise), must-do boost, L1-normalize.
constexpr double kPrefPopularityNoise = 0.25;  // mirrored from config.PREF_POPULARITY_NOISE
// Training-only (BC / personal PPO): i.i.d. U(eps, 1), must-do boost, L1-normalize.
constexpr double kPrefRawEps = 1e-3;  // mirrored from config.PREF_RAW_EPS
constexpr double kIdleWalkProb = 0.5;

// Heuristic ride-repeat dampening (mirrored from config.py)
constexpr int kRepeatTopK = 3;                    // Pass 2: top-K preference ranks
constexpr double kRepeatPrefThreshold = 0.04;     // Pass 2: high-pref mass floor
constexpr double kRepeatPrefScale = 2.0;          // max_repeats = 1 + floor(scale * pref * N)
constexpr int kRepeatMax = 3;                     // hard cap on Pass 2 completions
constexpr double kRepeatBalkFactor = 1.0;         // optional tighter balk on repeats
constexpr double kShortWaitSec = 12.0 * 60.0;     // Pass 3 absolute short-wait bar
constexpr double kShortWaitSlackSec = 2.0 * 60.0; // Pass 3 relative-to-best slack

// PPO reward shaping (mirrored from config.py PPO_* reward knobs).
// Preference / must-do latency objective; wait variance is not rewarded.
constexpr float kPrefRewardScale = 0.05f;
constexpr float kMustDoCompletionBonus = 0.15f;
constexpr float kTimeDecay = 0.75f;
constexpr float kMustDoUrgencyCoef = 2e-5f;
constexpr float kPrefUrgencyCoef = 1e-5f;
constexpr bool kWeightByPartySize = true;
constexpr float kUnfulfilledMustDoPenalty = 2.0f;

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
    // Preference / must-do KPIs (diagnostic + PPO eval headlines).
    int64_t must_dos_assigned = 0;
    int64_t must_dos_completed = 0;
    double preference_score_sum = 0.0;       // Σ preference[ride] * party_size
    double must_do_latency_sum_sec = 0.0;    // Σ (complete_sec - spawn_sec) for must-dos
    int64_t must_do_latency_count = 0;

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

    double must_do_completion_rate() const {
        if (must_dos_assigned <= 0) {
            return 1.0;
        }
        return static_cast<double>(must_dos_completed) / static_cast<double>(must_dos_assigned);
    }

    double avg_preference_score_per_guest() const {
        if (total_guests <= 0) {
            return 0.0;
        }
        return preference_score_sum / static_cast<double>(total_guests);
    }

    double avg_must_do_latency_sec() const {
        if (must_do_latency_count <= 0) {
            return 0.0;
        }
        return must_do_latency_sum_sec / static_cast<double>(must_do_latency_count);
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
    int32_t wave_id = 0;  // co-timed routing cohort (same decide_routes call)
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
    std::vector<int32_t> party_ids;
    int n_obs = 0;
    int n_rewards = 0;
    bool episode_done = false;
    DayMetricsResult metrics;
};

/** Aggregate KPIs for personal-training focals (heuristic crowd excluded). */
struct PersonalDayStats {
    int n_focals = 0;
    int64_t must_dos_assigned = 0;
    int64_t must_dos_completed = 0;
    double preference_score_sum = 0.0;
    int rides_completed = 0;
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

/** Override for the interactive / shadow focal guest (always size 1). */
struct FocalPartyConfig {
    int spawn_sec = 0;
    int leave_sec = kDaySeconds;
    std::array<float, kNumRides> preference_weights{};
    std::array<uint8_t, kNumRides> must_dos{};
};

/** Per-focal-guest preference / itinerary KPIs for interactive play. */
struct FocalPartyStats {
    int32_t party_id = -1;
    int32_t spawn_sec = 0;
    int32_t leave_sec = 0;
    int32_t exit_sec = -1;
    int32_t rides_completed = 0;
    int32_t must_dos_assigned = 0;
    int32_t must_dos_completed = 0;
    int32_t top3_hits = 0;
    float preference_score = 0.0f;
    uint8_t exited = 0;
    std::array<float, kNumRides> preferences{};
    std::array<uint8_t, kNumRides> must_dos_initial{};
    std::vector<PartyRideEvent> completions;
};

/** Result of advancing a hybrid play session until the next decision point. */
struct PlayStepResult {
    bool done = false;
    bool needs_human = false;
    bool needs_ppo_batch = false;
    int now_sec = 0;
    int focal_party_id = -1;
    Observation human_obs{};
    std::vector<float> ppo_obs;
    std::vector<int32_t> ppo_party_ids;
    int n_ppo = 0;
    DayMetricsResult metrics;
    FocalPartyStats focal;
};

int action_from_target(int target_ride_id);
int target_from_action(int action);

DayMetricsResult run_day(uint64_t seed);
DayRecording record_day(uint64_t seed, int sample_interval_sec = 60);
std::vector<BCSample> collect_bc_dataset(int num_days, uint64_t seed_start);

/** Heuristic crowd + heuristic focal day with a custom focal guest profile. */
struct PlayDayResult {
    DayMetricsResult metrics;
    DayRecording recording;
    FocalPartyStats focal;
};

PlayDayResult run_play_day(
    uint64_t seed,
    const FocalPartyConfig& focal,
    int sample_interval_sec = 60,
    bool record = true);

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
    /** Personal planner training: N focals + heuristic crowd. Returns first obs batch size via exchange_batch. */
    void reset_personal(uint64_t seed, int n_focals);
    EnvStepResult step(int action);
    RolloutBatchResult exchange_batch(const std::vector<int>& actions, int max_obs);
    PersonalDayStats personal_stats() const;

    /** Interactive / shadow hybrid session (focal guest + crowd policy).
     *  focal_policy: 0=human, 1=heuristic, 2=ppo
     */
    void reset_play(
        uint64_t seed,
        const FocalPartyConfig& focal,
        bool crowd_auto_heuristic,
        int focal_policy,
        bool soft_human_leave,
        bool enable_recording,
        int sample_interval_sec = 60);
    PlayStepResult play_advance();
    void play_apply_human_action(int action);
    void play_apply_ppo_actions(const std::vector<int>& actions);
    /** Mid-day preference / must-do update for the focal guest (keeps location & history). */
    void play_update_focal_preferences(const FocalPartyConfig& focal);
    /** Focal PartyState as int (Walking=1, InQueue=2, OnRide=4, Evacuating=8, Exited=16). */
    int play_focal_state() const;
    /** Per-ride completion counts for the focal guest (length NUM_RIDES). */
    std::array<int16_t, kNumRides> play_focal_ride_history() const;
    const DayRecording& play_recording() const;
    FocalPartyStats play_focal_stats() const;
    int play_now_sec() const;
    int play_focal_party_id() const;
    bool play_done() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace park
