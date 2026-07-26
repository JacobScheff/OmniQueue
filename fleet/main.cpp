#include <iostream>

#include "City.hpp"
#include "Simulation.hpp"

int main() {
    City city(/*width=*/1000, /*height=*/1000, /*numIntersections=*/80, /*seed=*/42);

    SimConfig cfg;
    cfg.numVehicles = 30;
    cfg.numRequests = 300;
    cfg.horizonSec = 3600;
    cfg.vehicleSpeed = 2.0;
    cfg.seed = 42;
    cfg.verbose = false;

    Simulation sim(city, cfg);
    const SimMetrics metrics = sim.run();
    printMetrics(metrics, cfg);
    return 0;
}
