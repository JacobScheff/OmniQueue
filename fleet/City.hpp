#ifndef CITY_HPP
#define CITY_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
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

inline int orientation(Position a, Position b, Position c) {
    const long long v =
        1LL * (b.y - a.y) * (c.x - b.x) - 1LL * (b.x - a.x) * (c.y - b.y);
    if (v > 0) return 1;
    if (v < 0) return -1;
    return 0;
}

inline bool onSegment(Position a, Position b, Position p) {
    return p.x >= std::min(a.x, b.x) && p.x <= std::max(a.x, b.x) &&
           p.y >= std::min(a.y, b.y) && p.y <= std::max(a.y, b.y);
}

// True if AB and CD cross or overlap in their interiors.
// Meeting only at a shared endpoint is allowed.
inline bool segmentsOverlap(Position a, Position b, Position c, Position d) {
    const bool shareEndpoint = a == c || a == d || b == c || b == d;

    const int o1 = orientation(a, b, c);
    const int o2 = orientation(a, b, d);
    const int o3 = orientation(c, d, a);
    const int o4 = orientation(c, d, b);

    if (o1 != o2 && o3 != o4) {
        return !shareEndpoint;
    }

    auto interiorOn = [](Position p, Position s, Position t) {
        if (p == s || p == t) return false;
        return orientation(s, t, p) == 0 && onSegment(s, t, p);
    };

    return interiorOn(c, a, b) || interiorOn(d, a, b) ||
           interiorOn(a, c, d) || interiorOn(b, c, d);
}

inline bool streetsOverlap(const Street& a, const Street& b) {
    if (!a.I1 || !a.I2 || !b.I1 || !b.I2) return false;
    if (a == b) return true;
    return segmentsOverlap(a.I1->pos, a.I2->pos, b.I1->pos, b.I2->pos);
}

inline long long dist2(Position a, Position b) {
    const long long dx = static_cast<long long>(a.x) - b.x;
    const long long dy = static_cast<long long>(a.y) - b.y;
    return dx * dx + dy * dy;
}

// Evenly spaced-ish points via Poisson-disk rejection sampling.
inline std::vector<Position> samplePoissonDisk(
    int width, int height, int count, std::default_random_engine& rng) {
    std::uniform_int_distribution<int> xDist(0, std::max(0, width - 1));
    std::uniform_int_distribution<int> yDist(0, std::max(0, height - 1));

    const double area = static_cast<double>(std::max(1, width)) * std::max(1, height);
    const double minDist =
        0.75 * std::sqrt(area / static_cast<double>(std::max(1, count)));
    const long long minDist2 = static_cast<long long>(minDist * minDist);

    std::vector<Position> points;
    points.reserve(static_cast<size_t>(count));

    const int maxAttempts = std::max(count * 60, 100);
    for (int attempt = 0; attempt < maxAttempts && static_cast<int>(points.size()) < count;
         ++attempt) {
        Position candidate{xDist(rng), yDist(rng)};
        bool ok = true;
        for (const Position& existing : points) {
            if (candidate == existing || dist2(candidate, existing) < minDist2) {
                ok = false;
                break;
            }
        }
        if (ok) {
            points.push_back(candidate);
        }
    }

    // If the disk radius was too strict, fill remaining with unique random points.
    int guard = 0;
    while (static_cast<int>(points.size()) < count && guard++ < count * 200) {
        Position candidate{xDist(rng), yDist(rng)};
        bool unique = true;
        for (const Position& existing : points) {
            if (candidate == existing) {
                unique = false;
                break;
            }
        }
        if (unique) {
            points.push_back(candidate);
        }
    }

    return points;
}

struct City {
    City(int width, int height, int numIntersections, int seed = 42,
         int avgStreetsPerIntersection = 5)
        : width(width), height(height), numIntersections(numIntersections) {
        avgStreetsPerIntersection = std::max(2, std::min(12, avgStreetsPerIntersection));

        std::default_random_engine generator(seed);

        // 1) Place intersections with Poisson-disk sampling.
        const std::vector<Position> positions =
            samplePoissonDisk(width, height, numIntersections, generator);
        this->numIntersections = static_cast<int>(positions.size());
        for (const Position& pos : positions) {
            intersections.push_back(new Intersection(pos));
        }

        const int n = static_cast<int>(intersections.size());
        if (n <= 1) {
            return;
        }

        // 2) Euclidean MST — connected and non-crossing.
        struct EdgeCand {
            int u;
            int v;
            long long d2;
            bool operator<(const EdgeCand& other) const { return d2 < other.d2; }
        };

        std::vector<EdgeCand> allPairs;
        allPairs.reserve(static_cast<size_t>(n) * (n - 1) / 2);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                allPairs.push_back(
                    EdgeCand{i, j, dist2(intersections[i]->pos, intersections[j]->pos)});
            }
        }
        std::sort(allPairs.begin(), allPairs.end());

        std::vector<int> parent(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) parent[static_cast<size_t>(i)] = i;
        auto findRoot = [&](int x) {
            while (parent[static_cast<size_t>(x)] != x) {
                parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
                x = parent[static_cast<size_t>(x)];
            }
            return x;
        };
        auto unite = [&](int a, int b) {
            a = findRoot(a);
            b = findRoot(b);
            if (a == b) return false;
            parent[static_cast<size_t>(a)] = b;
            return true;
        };

        auto tryAddStreet = [&](int u, int v) -> bool {
            Street candidate(intersections[static_cast<size_t>(u)],
                             intersections[static_cast<size_t>(v)]);
            if (streets.find(candidate) != streets.end()) {
                return false;
            }
            for (const Street& existing : streets) {
                if (streetsOverlap(candidate, existing)) {
                    return false;
                }
            }
            streets.insert(candidate);
            intersectionToStreets[candidate.I1].push_back(candidate);
            intersectionToStreets[candidate.I2].push_back(candidate);
            undirectedEdges.emplace_back(std::min(u, v), std::max(u, v));
            return true;
        };

        int mstEdges = 0;
        for (const EdgeCand& edge : allPairs) {
            if (!unite(edge.u, edge.v)) {
                continue;
            }
            // Euclidean MST edges never cross; add unconditionally for connectivity.
            Street street(intersections[static_cast<size_t>(edge.u)],
                          intersections[static_cast<size_t>(edge.v)]);
            streets.insert(street);
            intersectionToStreets[street.I1].push_back(street);
            intersectionToStreets[street.I2].push_back(street);
            undirectedEdges.emplace_back(std::min(edge.u, edge.v), std::max(edge.u, edge.v));
            ++mstEdges;
            if (mstEdges == n - 1) {
                break;
            }
        }

        // 3) Add short non-crossing chords until average degree ~ target.
        const int targetStreetCount =
            std::max(n - 1, (n * avgStreetsPerIntersection) / 2);
        for (const EdgeCand& edge : allPairs) {
            if (static_cast<int>(streets.size()) >= targetStreetCount) {
                break;
            }
            tryAddStreet(edge.u, edge.v);
        }

        // Construct the distance matrix
        constructDistanceMatrix();
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
    std::vector<std::pair<int, int>> undirectedEdges;

    std::vector<std::vector<int>> distanceMatrix;
    std::unordered_map<Intersection*, int> indexOf;

    static constexpr int kUnreachable = std::numeric_limits<int>::max();

    int distance(int u, int v) const {
        return distanceMatrix[static_cast<size_t>(u)][static_cast<size_t>(v)];
    }

    bool reachable(int u, int v) const {
        return distance(u, v) < kUnreachable;
    }

    // Travel time in seconds at constant speed (distance units per second).
    int travelTime(int u, int v, double speed = 1.0) const {
        const int d = distance(u, v);
        if (d >= kUnreachable) return kUnreachable;
        if (speed <= 0.0) return kUnreachable;
        return std::max(1, static_cast<int>(std::lround(static_cast<double>(d) / speed)));
    }

    private:
        void constructDistanceMatrix() {
            const int n = static_cast<int>(intersections.size());
            distanceMatrix.assign(static_cast<size_t>(n),
                                  std::vector<int>(static_cast<size_t>(n), kUnreachable));

            indexOf.clear();
            indexOf.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                indexOf[intersections[static_cast<size_t>(i)]] = i;
            }

            // Adjacency list; street weight = rounded Euclidean length.
            std::vector<std::vector<std::pair<int, int>>> adj(static_cast<size_t>(n));
            for (const Street& street : streets) {
                const int u = indexOf[street.I1];
                const int v = indexOf[street.I2];
                const int w = std::max(
                    1, static_cast<int>(std::lround(
                           std::sqrt(static_cast<double>(dist2(street.I1->pos, street.I2->pos))))));
                adj[static_cast<size_t>(u)].push_back({v, w});
                adj[static_cast<size_t>(v)].push_back({u, w});
            }

            // Multi-source Dijkstra: every intersection is seeded as a source in
            // one shared heap; entries carry (dist, source, node) so each source's
            // search settles independently.
            struct Entry {
                int dist;
                int source;
                int node;
                bool operator>(const Entry& other) const { return dist > other.dist; }
            };
            std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;

            for (int s = 0; s < n; ++s) {
                distanceMatrix[static_cast<size_t>(s)][static_cast<size_t>(s)] = 0;
                heap.push(Entry{0, s, s});
            }

            while (!heap.empty()) {
                const Entry top = heap.top();
                heap.pop();
                auto& distFromSource = distanceMatrix[static_cast<size_t>(top.source)];
                if (top.dist > distFromSource[static_cast<size_t>(top.node)]) {
                    continue;  // Stale entry.
                }
                for (const auto& [next, weight] : adj[static_cast<size_t>(top.node)]) {
                    const int candidate = top.dist + weight;
                    if (candidate < distFromSource[static_cast<size_t>(next)]) {
                        distFromSource[static_cast<size_t>(next)] = candidate;
                        heap.push(Entry{candidate, top.source, next});
                    }
                }
            }
        }
};

#endif
