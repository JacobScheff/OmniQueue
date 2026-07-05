#pragma once

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

constexpr double kBaseBalkSec = 600.0;
constexpr double kBalkScale = 2400.0;
constexpr double kBalkPrefExp = 1.5;
constexpr double kMustDoPrefBoost = 10.0;
constexpr double kIdleWalkProb = 0.5;

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
};

DayMetricsResult run_day(uint64_t seed);

}  // namespace park
