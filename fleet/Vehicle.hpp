#ifndef VEHICLE_HPP
#define VEHICLE_HPP

#include <cstdint>

#include "City.hpp"
#include "Position.hpp"

enum class VehicleStatus : uint8_t {
    Idle = 0,
    EnRoutePickup = 1,
    EnRouteDropoff = 2,
    EnRouteIdle = 3,
};

struct Vehicle {
    int id = -1;
    VehicleStatus status = VehicleStatus::Idle;

    // Current / trip endpoints as intersection indices into City::intersections.
    int node = 0;
    int fromNode = 0;
    int toNode = 0;

    // Trip timing for position lerp (seconds).
    int departTime = 0;
    int arriveTime = 0;

    int assignedRequest = -1;
    int capacity = 1;
    float soc = 1.0f;  // reserved for later charger phase

    bool isFree() const {
        return status == VehicleStatus::Idle;
    }

    Position getPosition(const City& city, int now) const {
        if (status == VehicleStatus::Idle || arriveTime <= departTime || now <= departTime) {
            return city.intersections[static_cast<size_t>(node)]->pos;
        }
        if (now >= arriveTime) {
            return city.intersections[static_cast<size_t>(toNode)]->pos;
        }
        const float t = static_cast<float>(now - departTime) /
                        static_cast<float>(arriveTime - departTime);
        const Position& a = city.intersections[static_cast<size_t>(fromNode)]->pos;
        const Position& b = city.intersections[static_cast<size_t>(toNode)]->pos;
        return Position{
            a.x + static_cast<int>(t * (b.x - a.x)),
            a.y + static_cast<int>(t * (b.y - a.y)),
        };
    }
};

#endif
