#include "Visualization.hpp"
#include "City.hpp"

int main() {
    City city(100, 100, 50, 42);
    Visualization viz(city);
    viz.show();
    return 0;
}
