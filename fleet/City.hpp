#include "iostream"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>

using namespace std;

struct Street;

struct Intersection {
    public:
        Intersection(int x, int y) : x(x), y(y) {}

        int x;
        int y;

        vector<Street*> streets;
};

struct Street {
    public:
        Street(Intersection* I1, Intersection* I2) : I1(I1), I2(I2) {}

        Intersection* I1;
        Intersection* I2;
};

struct City {
    public:
        City(int width, int height, int numIntersections, int seed = 42)
            : width(width), height(height), numIntersections(numIntersections) {
                int avgStreetsPerIntersection = 4;

                std::default_random_engine generator(seed);
                std::normal_distribution<double> distribution{avgStreetsPerIntersection, 1.0};

                for (int i = 0; i < numIntersections; ++i) {
                    int x = rand() % width;
                    int y = rand() % height;
                    intersections.push_back(new Intersection(x, y));
                }

                for (int i = 0; i < numIntersections; ++i) {
                    int numStreets = int(distribution(generator));
                    numStreets = max(1, numStreets);

                    int currStreets = 0;
                    if (intersectionToStreets.find(intersections[i]) == intersectionToStreets.end()) {
                        intersectionToStreets[intersections[i]] = vector<Street*>();
                    } else {
                        currStreets = intersectionToStreets[intersections[i]].size();
                    }

                    int streetsToCreate = numStreets - currStreets;
                    if (streetsToCreate <= 0) continue;

                    // Sort intersections by distance to the current intersection
                    vector<Intersection*> sortedIntersections = intersections;
                    sort(sortedIntersections.begin(), sortedIntersections.end(), [i](Intersection* a, Intersection* b) {
                        return sqrt(pow(a->x - intersections[i]->x, 2) + pow(a->y - intersections[i]->y, 2)) < sqrt(pow(b->x - intersections[i]->x, 2) + pow(b->y - intersections[i]->y, 2));
                    });

                    int k = 1; // Skip the current intersection
                    for (int j = 0; j < numStreets - currStreets; ++j) {
                        // Create new street between intersections[i] and the closest intersection that is not connected to it
                        Intersection* targetIntersection = sortedIntersections[k];

                        // Check if street from intersections[i] to targetIntersection or targetIntersection to intersections[i] already exists
                        if (std::find(intersectionToStreets[intersections[i]].begin(), intersectionToStreets[intersections[i]].end(), Street(intersections[i], targetIntersection)) != intersectionToStreets[intersections[i]].end()) {
                            continue;
                        }

                        if (std::find(intersectionToStreets[targetIntersection].begin(), intersectionToStreets[targetIntersection].end(), Street(targetIntersection, intersections[i])) != intersectionToStreets[targetIntersection].end()) {
                            continue;
                        }

                        Street newStreet(intersections[i], targetIntersection);

                        streets.insert(newStreet);
                        intersectionToStreets[intersections[i]].push_back(newStreet);
                        intersectionToStreets[targetIntersection].push_back(newStreet);
                    }
                }
                
            }

        int width;
        int height;
        int numIntersections;

        vector<Intersection*> intersections;
        unordered_map<Intersection*, vector<Street>> intersectionToStreets;
        unordered_set<Street> streets;
};