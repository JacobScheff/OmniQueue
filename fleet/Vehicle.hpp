#include "City.hpp"
#include "Position.hpp"

class Vehicle {
    public:
        Vehicle();

        Position getPosition() const {
            int x = intersectionOne->pos.x + t * (intersectionTwo->pos.x - intersectionOne->pos.x);
            int y = intersectionOne->pos.y + t * (intersectionTwo->pos.y - intersectionOne->pos.y);
            return Position{x, y};
        }

    private:
        Intersection* intersectionOne;
        Intersection* intersectionTwo;
        int t; // Lerp between both intersections
};