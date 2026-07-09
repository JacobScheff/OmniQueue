#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace park {

constexpr int kNumRides = 35;
constexpr int kDaySeconds = 54000;

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
constexpr int kNumActions = 37;  // 35 rides + exit + idle
constexpr int kFlatObsDim = kGuestFeatDim + kNumRides * kRideDynamicFeatDim + kEnvDynamicFeatDim;

constexpr double kTotalGuestsMean = 50000.0;
constexpr double kTotalGuestsStd = 2500.0;
constexpr double kPartySizeMean = 3.2;
constexpr double kPartySizeStd = 1.0;
constexpr double kSpawnMeanSec = 3.0 * 3600.0;
constexpr double kSpawnStdSec = 2.0 * 3600.0;
constexpr double kDwellMeanSec = 10.0 * 3600.0;
constexpr double kDwellStdSec = 2.0 * 3600.0;

constexpr double kBaseWalkingSpeed = 1.4;
constexpr double kMemberSpeedLogMu = 0.3364722366212129;  // log(1.4)
constexpr double kMemberSpeedLogSigma = 0.25;

constexpr double kBaseBalkSec = 40.0 * 60.0;  // 40 min floor (mirrored from config.py)
constexpr double kBalkScale = 5.0 * 60.0;     // +0–5 min by preference^exp (max ~45 min)
constexpr double kBalkPrefExp = 1.5;
constexpr double kMustDoPrefBoost = 10.0;
constexpr double kIdleWalkProb = 0.5;

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

int action_from_target(int target_ride_id);
int target_from_action(int action);

DayMetricsResult run_day(uint64_t seed);
std::vector<BCSample> collect_bc_dataset(int num_days, uint64_t seed_start);

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
