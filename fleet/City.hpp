#include <vector>

using namespace std;

struct Intersection {
    public:
        Intersection(int id) : id(id) {}

        int id;

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
        City(int m, int n) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    intersections.push_back(new Intersection(i * n + j));
                }
            }

            // Add streets between intersections (horizontal and vertical)
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n - 1; j++) {
                    streets.push_back(new Street(intersections[i * n + j], intersections[i * n + j + 1]));
                }
            }
            for (int i = 0; i < m - 1; i++) {
                for (int j = 0; j < n; j++) {
                    streets.push_back(new Street(intersections[i * n + j], intersections[(i + 1) * n + j]));
                }
            }
        };

        void addStreet(Street* street);

    private:
        vector<Intersection*> intersections;
        vector<Street*> streets;
};