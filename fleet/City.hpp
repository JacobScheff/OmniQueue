#ifndef CITY_HPP
#define CITY_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Position.hpp"

struct Street;

struct Intersection {
    Intersection(Position pos) : pos(pos) {}

    Position pos;
};

struct Street {
    Street(Intersection* I1, Intersection* I2) : I1(I1), I2(I2) {}

    Intersection* I1;
    Intersection* I2;

    bool operator==(const Street& other) const {
        return (I1 == other.I1 && I2 == other.I2) || (I1 == other.I2 && I2 == other.I1);
    }
};

namespace std {
template <>
struct hash<Street> {
    size_t operator()(const Street& s) const {
        auto a = reinterpret_cast<uintptr_t>(s.I1);
        auto b = reinterpret_cast<uintptr_t>(s.I2);
        if (a > b) std::swap(a, b);
        return std::hash<uintptr_t>()(a) ^ (std::hash<uintptr_t>()(b) << 1);
    }
};
}

struct City {
    City(int width, int height, int numIntersections, int seed = 42)
        : width(width), height(height), numIntersections(numIntersections) {
        int avgStreetsPerIntersection = 4;

        std::default_random_engine generator(seed);
        std::normal_distribution<double> streetDist{
            static_cast<double>(avgStreetsPerIntersection), 1.0};
        std::uniform_int_distribution<int> xDist(0, width - 1);
        std::uniform_int_distribution<int> yDist(0, height - 1);

        for (int i = 0; i < numIntersections; ++i) {
            intersections.push_back(new Intersection(Position{xDist(generator), yDist(generator)}));
        }

        for (int i = 0; i < numIntersections; ++i) {
            int numStreets = std::max(1, static_cast<int>(streetDist(generator)));
            int currStreets = static_cast<int>(intersectionToStreets[intersections[i]].size());
            int streetsToCreate = numStreets - currStreets;
            if (streetsToCreate <= 0) continue;

            std::vector<Intersection*> sortedIntersections = intersections;
            std::sort(sortedIntersections.begin(), sortedIntersections.end(),
                      [this, i](Intersection* a, Intersection* b) {
                          auto dist2 = [this, i](Intersection* p) {
                              double dx = p->pos.x - intersections[i]->pos.x;
                              double dy = p->pos.y - intersections[i]->pos.y;
                              return dx * dx + dy * dy;
                          };
                          return dist2(a) < dist2(b);
                      });

            int created = 0;
            for (size_t k = 1; k < sortedIntersections.size() && created < streetsToCreate; ++k) {
                Intersection* target = sortedIntersections[k];
                Street candidate(intersections[i], target);

                auto& fromStreets = intersectionToStreets[intersections[i]];
                if (std::find(fromStreets.begin(), fromStreets.end(), candidate) != fromStreets.end()) {
                    continue;
                }

                streets.insert(candidate);
                intersectionToStreets[intersections[i]].push_back(candidate);
                intersectionToStreets[target].push_back(candidate);
                ++created;
            }
        }
    }

    ~City() {
        for (Intersection* intersection : intersections) {
            delete intersection;
        }
    }

    City(const City&) = delete;
    City& operator=(const City&) = delete;

    int width;
    int height;
    int numIntersections;

    std::vector<Intersection*> intersections;
    std::unordered_map<Intersection*, std::vector<Street>> intersectionToStreets;
    std::unordered_set<Street> streets;
};

#endif
