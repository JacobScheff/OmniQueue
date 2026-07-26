#ifndef RECORDING_HPP
#define RECORDING_HPP

// Include after SimMetrics is defined (see Simulation.hpp).

#include <cstdint>
#include <utility>
#include <vector>

enum class TripKind : uint8_t {
    Pickup = 0,
    Dropoff = 1,
    Idle = 2,
};

struct CityGeometry {
    int width = 0;
    int height = 0;
    std::vector<std::pair<int, int>> nodes;   // (x, y) per intersection index
    std::vector<std::pair<int, int>> edges;   // (u, v) undirected street endpoints
};

struct TripRecord {
    int32_t vehicleId = -1;
    int32_t fromNode = 0;
    int32_t toNode = 0;
    int32_t startSec = 0;
    int32_t endSec = 0;
    int32_t requestId = -1;
    TripKind kind = TripKind::Pickup;
};

struct RequestRecord {
    int32_t id = -1;
    int32_t origin = 0;
    int32_t dest = 0;
    int32_t spawnSec = 0;
    int32_t assignSec = -1;
    int32_t pickupSec = -1;
    int32_t dropoffSec = -1;
    uint8_t status = 0;  // RequestStatus as uint8_t
};

struct MetricSample {
    int32_t sec = 0;
    int32_t pending = 0;
    int32_t freeVehicles = 0;
    int32_t busyVehicles = 0;
    int32_t completed = 0;
    int32_t spawned = 0;
    float meanWait = 0.0f;
};

struct RecordConfig {
    int cityWidth = 1200;
    int cityHeight = 1200;
    int numIntersections = 80;
    int numVehicles = 25;
    int numRequests = 3840;
    int horizonSec = 86400;
    double vehicleSpeed = 2.0;
    int vehicleCapacity = 1;
};

/** Full-day (or shift) recording consumed by the Pygame visualizer. */
struct DayRecording {
    CityGeometry city;
    std::vector<TripRecord> trips;
    std::vector<RequestRecord> requests;
    std::vector<MetricSample> samples;
    std::vector<int32_t> vehicleStartNodes;  // index = vehicle id
    SimMetrics metrics;

    int numVehicles = 0;
    int numRequests = 0;
    int horizonSec = 0;
    int numIntersections = 0;
    double vehicleSpeed = 0.0;
};

#endif
