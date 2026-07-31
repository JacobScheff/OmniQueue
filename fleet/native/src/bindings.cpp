#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "FleetEnv.hpp"

namespace py = pybind11;

namespace {

py::array_t<float> observation_flat(const fleet_env::Observation& obs) {
    py::array_t<float> arr(fleet_env::kFlatObsDim);
    std::memcpy(arr.mutable_data(), obs.flat.data(), fleet_env::kFlatObsDim * sizeof(float));
    return arr;
}

}  // namespace

PYBIND11_MODULE(_fleet_sim, m) {
    m.doc() = "C++ discrete-event fleet dispatch simulator";

    m.attr("NUM_ACTIONS") = fleet_env::kNumActions;
    m.attr("FLAT_OBS_DIM") = fleet_env::kFlatObsDim;

    py::class_<SimMetrics>(m, "SimMetrics")
        .def_readonly("requests_spawned", &SimMetrics::requestsSpawned)
        .def_readonly("requests_completed", &SimMetrics::requestsCompleted)
        .def_readonly("requests_cancelled", &SimMetrics::requestsCancelled)
        .def_readonly("assignments", &SimMetrics::assignments)
        .def_readonly("wait_sum", &SimMetrics::waitSum)
        .def_readonly("trip_sum", &SimMetrics::tripSum)
        .def("mean_wait", &SimMetrics::meanWait)
        .def("mean_trip", &SimMetrics::meanTrip)
        .def("completion_rate", &SimMetrics::completionRate);

    py::class_<fleet_env::EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("city_width", &fleet_env::EnvConfig::cityWidth)
        .def_readwrite("city_height", &fleet_env::EnvConfig::cityHeight)
        .def_readwrite("num_intersections", &fleet_env::EnvConfig::numIntersections)
        .def_readwrite("num_vehicles", &fleet_env::EnvConfig::numVehicles)
        .def_readwrite("num_requests", &fleet_env::EnvConfig::numRequests)
        .def_readwrite("horizon_sec", &fleet_env::EnvConfig::horizonSec)
        .def_readwrite("vehicle_speed", &fleet_env::EnvConfig::vehicleSpeed)
        .def_readwrite("vehicle_capacity", &fleet_env::EnvConfig::vehicleCapacity)
        .def_readwrite("avg_streets_per_intersection",
                       &fleet_env::EnvConfig::avgStreetsPerIntersection);

    py::class_<fleet_env::Observation>(m, "Observation")
        .def("flat", &observation_flat)
        .def_readonly("vehicle_id", &fleet_env::Observation::vehicleId)
        .def_readonly("n_requests", &fleet_env::Observation::nRequests);

    py::class_<fleet_env::EnvStepResult>(m, "EnvStepResult")
        .def_readonly("obs", &fleet_env::EnvStepResult::obs)
        .def_readonly("reward", &fleet_env::EnvStepResult::reward)
        .def_readonly("done", &fleet_env::EnvStepResult::done)
        .def_readonly("has_obs", &fleet_env::EnvStepResult::has_obs)
        .def_readonly("metrics", &fleet_env::EnvStepResult::metrics);

    py::class_<fleet_env::FleetEnv>(m, "FleetEnv")
        .def(py::init<uint64_t, fleet_env::EnvConfig>(), py::arg("seed") = 0,
             py::arg("config") = fleet_env::EnvConfig{})
        .def("reset", &fleet_env::FleetEnv::reset, py::arg("seed"))
        .def("step", &fleet_env::FleetEnv::step, py::arg("action"))
        .def("enable_recording", &fleet_env::FleetEnv::enable_recording,
             py::arg("sample_interval_sec") = 60)
        .def_property_readonly("recording", &fleet_env::FleetEnv::recording)
        .def_property_readonly("metrics", &fleet_env::FleetEnv::metrics);

    py::enum_<TripKind>(m, "TripKind")
        .value("Pickup", TripKind::Pickup)
        .value("Dropoff", TripKind::Dropoff)
        .value("Idle", TripKind::Idle);

    py::class_<CityGeometry>(m, "CityGeometry")
        .def_readonly("width", &CityGeometry::width)
        .def_readonly("height", &CityGeometry::height)
        .def_readonly("nodes", &CityGeometry::nodes)
        .def_readonly("edges", &CityGeometry::edges);

    py::class_<TripRecord>(m, "TripRecord")
        .def_readonly("vehicle_id", &TripRecord::vehicleId)
        .def_readonly("from_node", &TripRecord::fromNode)
        .def_readonly("to_node", &TripRecord::toNode)
        .def_readonly("start_sec", &TripRecord::startSec)
        .def_readonly("end_sec", &TripRecord::endSec)
        .def_readonly("request_id", &TripRecord::requestId)
        .def_readonly("kind", &TripRecord::kind);

    py::class_<RequestRecord>(m, "RequestRecord")
        .def_readonly("id", &RequestRecord::id)
        .def_readonly("origin", &RequestRecord::origin)
        .def_readonly("dest", &RequestRecord::dest)
        .def_readonly("spawn_sec", &RequestRecord::spawnSec)
        .def_readonly("assign_sec", &RequestRecord::assignSec)
        .def_readonly("pickup_sec", &RequestRecord::pickupSec)
        .def_readonly("dropoff_sec", &RequestRecord::dropoffSec)
        .def_readonly("status", &RequestRecord::status);

    py::class_<MetricSample>(m, "MetricSample")
        .def_readonly("sec", &MetricSample::sec)
        .def_readonly("pending", &MetricSample::pending)
        .def_readonly("free_vehicles", &MetricSample::freeVehicles)
        .def_readonly("busy_vehicles", &MetricSample::busyVehicles)
        .def_readonly("completed", &MetricSample::completed)
        .def_readonly("spawned", &MetricSample::spawned)
        .def_readonly("mean_wait", &MetricSample::meanWait);

    py::class_<RecordConfig>(m, "RecordConfig")
        .def(py::init<>())
        .def_readwrite("city_width", &RecordConfig::cityWidth)
        .def_readwrite("city_height", &RecordConfig::cityHeight)
        .def_readwrite("num_intersections", &RecordConfig::numIntersections)
        .def_readwrite("num_vehicles", &RecordConfig::numVehicles)
        .def_readwrite("num_requests", &RecordConfig::numRequests)
        .def_readwrite("horizon_sec", &RecordConfig::horizonSec)
        .def_readwrite("vehicle_speed", &RecordConfig::vehicleSpeed)
        .def_readwrite("vehicle_capacity", &RecordConfig::vehicleCapacity)
        .def_readwrite("avg_streets_per_intersection",
                       &RecordConfig::avgStreetsPerIntersection);

    py::class_<DayRecording>(m, "DayRecording")
        .def_readonly("city", &DayRecording::city)
        .def_readonly("trips", &DayRecording::trips)
        .def_readonly("requests", &DayRecording::requests)
        .def_readonly("samples", &DayRecording::samples)
        .def_readonly("vehicle_start_nodes", &DayRecording::vehicleStartNodes)
        .def_readonly("metrics", &DayRecording::metrics)
        .def_readonly("num_vehicles", &DayRecording::numVehicles)
        .def_readonly("num_requests", &DayRecording::numRequests)
        .def_readonly("horizon_sec", &DayRecording::horizonSec)
        .def_readonly("num_intersections", &DayRecording::numIntersections)
        .def_readonly("vehicle_speed", &DayRecording::vehicleSpeed);

    m.def(
        "record_day",
        [](uint64_t seed, const RecordConfig& cfg, int sample_interval_sec) {
            return record_day(seed, cfg, sample_interval_sec);
        },
        py::arg("seed") = 0, py::arg("config") = RecordConfig{},
        py::arg("sample_interval_sec") = 60,
        "Simulate one seeded heuristic day/shift and return a DayRecording.");
}
