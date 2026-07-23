#ifndef REQUEST_HPP
#define REQUEST_HPP

#include <cstdint>

enum class RequestStatus : uint8_t {
    Scheduled = 0,  // seeded, not yet visible to dispatch
    Pending = 1,
    Assigned = 2,
    PickedUp = 3,
    Completed = 4,
    Cancelled = 5,
};

struct Request {
    int id = -1;
    int origin = 0;
    int destination = 0;
    int size = 1;

    int spawnTime = 0;
    int assignTime = -1;
    int pickupTime = -1;
    int dropoffTime = -1;

    int assignedVehicle = -1;
    RequestStatus status = RequestStatus::Scheduled;

    bool isOpen() const {
        return status == RequestStatus::Pending;
    }

    int waitTime() const {
        if (pickupTime < 0) return -1;
        return pickupTime - spawnTime;
    }

    int tripTime() const {
        if (dropoffTime < 0 || pickupTime < 0) return -1;
        return dropoffTime - pickupTime;
    }
};

#endif
