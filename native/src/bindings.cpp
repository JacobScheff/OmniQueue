#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "park_sim.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_park_sim, m) {
    m.doc() = "C++ discrete event simulator for OmniQueue";

    py::class_<park::DayMetricsResult>(m, "DayMetrics")
        .def_readonly("total_parties", &park::DayMetricsResult::total_parties)
        .def_readonly("total_guests", &park::DayMetricsResult::total_guests)
        .def_readonly("rides_completed", &park::DayMetricsResult::rides_completed)
        .def_readonly("parties_exited", &park::DayMetricsResult::parties_exited)
        .def_readonly("breakdown_count", &park::DayMetricsResult::breakdown_count)
        .def_readonly("wait_variance_samples", &park::DayMetricsResult::wait_variance_samples)
        .def_readonly("mean_wait_samples", &park::DayMetricsResult::mean_wait_samples)
        .def_readonly("wall_time_sec", &park::DayMetricsResult::wall_time_sec)
        .def("rides_per_party", &park::DayMetricsResult::rides_per_party)
        .def("rides_per_guest", [](const park::DayMetricsResult& self) {
            if (self.total_guests == 0) {
                return 0.0;
            }
            return static_cast<double>(self.rides_completed) / self.total_guests;
        })
        .def("avg_wait_variance", [](const park::DayMetricsResult& self) {
            if (self.wait_variance_samples.empty()) {
                return 0.0;
            }
            double sum = 0.0;
            for (double v : self.wait_variance_samples) {
                sum += v;
            }
            return sum / self.wait_variance_samples.size();
        });

    m.def("run_day", &park::run_day, py::arg("seed") = 0, "Simulate one park day and return metrics.");
    m.def("is_available", []() { return true; });
}
