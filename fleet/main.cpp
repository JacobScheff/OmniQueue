#include "Visualization.hpp"
#include "City.hpp"

int main() {
    City city(100, 100, 50);
    Visualization viz(city);
    viz.show();
    return 0;
}
