#include "City.hpp"

class Request {
    public:
        Request(Intersection* start, Intersection* destination, int time) : start(start), destination(destination), requestedAtTime(time) {}

    private:
        Intersection* start;
        Intersection* destination;

        int requestedAtTime;
};