#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "park_sim.hpp"

namespace py = pybind11;

namespace {

py::array_t<float> observation_flat(const park::Observation& obs) {
    auto flat = obs.flat();
    py::array_t<float> arr(park::kFlatObsDim);
    std::memcpy(arr.mutable_data(), flat.data(), park::kFlatObsDim * sizeof(float));
    return arr;
}

py::array_t<float> vector_to_obs_batch(const std::vector<float>& data, int n) {
    if (n <= 0) {
        return py::array_t<float>(std::vector<py::ssize_t>{0, park::kFlatObsDim});
    }
    py::array_t<float> arr({n, park::kFlatObsDim});
    std::memcpy(
        arr.mutable_data(),
        data.data(),
        static_cast<size_t>(n) * park::kFlatObsDim * sizeof(float));
    return arr;
}

py::array_t<float> vector_to_rewards(const std::vector<float>& data) {
    const py::ssize_t n = static_cast<py::ssize_t>(data.size());
    py::array_t<float> arr(n);
    if (n > 0) {
        std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(float));
    }
    return arr;
}

}  // namespace

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
        .def("avg_wait_variance", &park::DayMetricsResult::avg_wait_variance)
        .def("rides_per_guest", [](const park::DayMetricsResult& self) {
            if (self.total_guests == 0) {
                return 0.0;
            }
            return static_cast<double>(self.rides_completed) / self.total_guests;
        });

    py::class_<park::Observation>(m, "Observation")
        .def("flat", &observation_flat)
        .def_property_readonly(
            "guest",
            [](const park::Observation& o) {
                py::array_t<float> arr(park::kGuestFeatDim);
                std::memcpy(arr.mutable_data(), o.guest.data(), park::kGuestFeatDim * sizeof(float));
                return arr;
            })
        .def_property_readonly(
            "ride",
            [](const park::Observation& o) {
                py::array_t<float> arr({park::kNumRides, park::kRideDynamicFeatDim});
                std::memcpy(arr.mutable_data(), o.ride.data(), o.ride.size() * sizeof(float));
                return arr;
            })
        .def_property_readonly(
            "env",
            [](const park::Observation& o) {
                py::array_t<float> arr(park::kEnvDynamicFeatDim);
                std::memcpy(arr.mutable_data(), o.env.data(), park::kEnvDynamicFeatDim * sizeof(float));
                return arr;
            });

    py::class_<park::BCSample>(m, "BCSample")
        .def_readonly("obs", &park::BCSample::obs)
        .def_readonly("action", &park::BCSample::action)
        .def_readonly("wave_id", &park::BCSample::wave_id);

    py::class_<park::EnvStepResult>(m, "EnvStepResult")
        .def_readonly("obs", &park::EnvStepResult::obs)
        .def_readonly("reward", &park::EnvStepResult::reward)
        .def_readonly("done", &park::EnvStepResult::done)
        .def_readonly("has_obs", &park::EnvStepResult::has_obs)
        .def_readonly("metrics", &park::EnvStepResult::metrics);

    py::class_<park::RolloutBatchResult>(m, "RolloutBatchResult")
        .def_readonly("n_obs", &park::RolloutBatchResult::n_obs)
        .def_readonly("n_rewards", &park::RolloutBatchResult::n_rewards)
        .def_readonly("episode_done", &park::RolloutBatchResult::episode_done)
        .def_readonly("metrics", &park::RolloutBatchResult::metrics)
        .def_property_readonly("obs", [](const park::RolloutBatchResult& self) {
            return vector_to_obs_batch(self.obs, self.n_obs);
        })
        .def_property_readonly("rewards", [](const park::RolloutBatchResult& self) {
            return vector_to_rewards(self.rewards);
        });

    py::class_<park::WalkRecord>(m, "WalkRecord")
        .def_readonly("party_id", &park::WalkRecord::party_id)
        .def_readonly("start_sec", &park::WalkRecord::start_sec)
        .def_readonly("end_sec", &park::WalkRecord::end_sec)
        .def_readonly("planned_end_sec", &park::WalkRecord::planned_end_sec)
        .def_readonly("from_idx", &park::WalkRecord::from_idx)
        .def_readonly("to_idx", &park::WalkRecord::to_idx)
        .def_readonly("target_ride", &park::WalkRecord::target_ride)
        .def_readonly("path_variant", &park::WalkRecord::path_variant)
        .def_readonly("cancelled", &park::WalkRecord::cancelled);

    py::class_<park::RideSample>(m, "RideSample")
        .def_readonly("sec", &park::RideSample::sec)
        .def_property_readonly(
            "wait",
            [](const park::RideSample& s) {
                py::array_t<float> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), s.wait.data(), park::kNumRides * sizeof(float));
                return arr;
            })
        .def_property_readonly(
            "broken",
            [](const park::RideSample& s) {
                py::array_t<uint8_t> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), s.broken.data(), park::kNumRides * sizeof(uint8_t));
                return arr;
            })
        .def_property_readonly(
            "queue_len",
            [](const park::RideSample& s) {
                py::array_t<int32_t> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), s.queue_len.data(), park::kNumRides * sizeof(int32_t));
                return arr;
            });

    py::class_<park::PartyInfo>(m, "PartyInfo")
        .def_readonly("party_id", &park::PartyInfo::party_id)
        .def_readonly("size", &park::PartyInfo::size)
        .def_readonly("spawn_sec", &park::PartyInfo::spawn_sec)
        .def_readonly("leave_sec", &park::PartyInfo::leave_sec)
        .def_readonly("rides_completed", &park::PartyInfo::rides_completed);

    py::class_<park::PartyRideEvent>(m, "PartyRideEvent")
        .def_readonly("party_id", &park::PartyRideEvent::party_id)
        .def_readonly("sec", &park::PartyRideEvent::sec)
        .def_readonly("ride_id", &park::PartyRideEvent::ride_id);

    py::class_<park::DayRecording>(m, "DayRecording")
        .def_readonly("metrics", &park::DayRecording::metrics)
        .def_readonly("parties", &park::DayRecording::parties)
        .def_readonly("walks", &park::DayRecording::walks)
        .def_readonly("ride_samples", &park::DayRecording::ride_samples)
        .def_readonly("ride_completions", &park::DayRecording::ride_completions);

    py::class_<park::ParkEnv>(m, "ParkEnv")
        .def(py::init<uint64_t>(), py::arg("seed") = 0)
        .def("reset", &park::ParkEnv::reset, py::arg("seed"))
        .def("step", &park::ParkEnv::step, py::arg("action"))
        .def(
            "exchange_batch",
            &park::ParkEnv::exchange_batch,
            py::arg("actions"),
            py::arg("max_obs"),
            "Apply a batch of actions, then collect up to max_obs pending observations.");

    m.attr("NUM_RIDES") = park::kNumRides;
    m.attr("NUM_ACTIONS") = park::kNumActions;
    m.attr("GUEST_FEAT_DIM") = park::kGuestFeatDim;
    m.attr("RIDE_DYNAMIC_FEAT_DIM") = park::kRideDynamicFeatDim;
    m.attr("ENV_DYNAMIC_FEAT_DIM") = park::kEnvDynamicFeatDim;
    m.attr("FLAT_OBS_DIM") = park::kFlatObsDim;
    m.attr("DAY_SECONDS") = park::kDaySeconds;
    m.attr("EXIT_RIDE_ID") = park::kExitRideId;
    m.attr("ROUTE_IDLE_CODE") = park::kRouteIdleCode;

    m.def("run_day", &park::run_day, py::arg("seed") = 0, "Simulate one park day and return metrics.");
    m.def(
        "record_day",
        &park::record_day,
        py::arg("seed") = 0,
        py::arg("sample_interval_sec") = 60,
        "Simulate one park day and return a visualization recording.");
    m.def(
        "collect_bc_dataset",
        &park::collect_bc_dataset,
        py::arg("num_days") = 1,
        py::arg("seed_start") = 0,
        "Collect heuristic routing samples for behavioral cloning.");
    m.def(
        "route_one_for_test",
        [](int now_sec,
           int leave_sec,
           int node_idx,
           float speed,
           const std::vector<int16_t>& preference_order,
           const std::vector<float>& preferences,
           const std::vector<float>& balk_sec,
           const std::vector<int16_t>& ride_history,
           const std::vector<uint8_t>& open_mask,
           const std::vector<float>& wait_times,
           const std::vector<int>& durations,
           double rand_u01) {
            if (preference_order.size() != static_cast<size_t>(park::kNumRides) ||
                preferences.size() != static_cast<size_t>(park::kNumRides) ||
                balk_sec.size() != static_cast<size_t>(park::kNumRides) ||
                ride_history.size() != static_cast<size_t>(park::kNumRides) ||
                open_mask.size() != static_cast<size_t>(park::kNumRides) ||
                wait_times.size() != static_cast<size_t>(park::kNumRides) ||
                durations.size() != static_cast<size_t>(park::kNumRides)) {
                throw std::invalid_argument("route_one_for_test: all ride arrays must have length NUM_RIDES");
            }
            park::RouteOneTestInput in;
            in.now_sec = now_sec;
            in.leave_sec = leave_sec;
            in.node_idx = node_idx;
            in.speed = speed;
            in.rand_u01 = rand_u01;
            for (int i = 0; i < park::kNumRides; ++i) {
                in.preference_order[i] = preference_order[i];
                in.preferences[i] = preferences[i];
                in.balk_sec[i] = balk_sec[i];
                in.ride_history[i] = ride_history[i];
                in.open_mask[i] = open_mask[i] != 0;
                in.wait_times[i] = wait_times[i];
                in.durations[i] = durations[i];
            }
            return park::route_one_for_test(in);
        },
        py::arg("now_sec"),
        py::arg("leave_sec"),
        py::arg("node_idx"),
        py::arg("speed"),
        py::arg("preference_order"),
        py::arg("preferences"),
        py::arg("balk_sec"),
        py::arg("ride_history"),
        py::arg("open_mask"),
        py::arg("wait_times"),
        py::arg("durations"),
        py::arg("rand_u01") = 1.0,
        "Deterministic heuristic routing helper for unit tests.");
    m.def("is_available", []() { return true; });
    m.attr("HAS_EXCHANGE_BATCH") = true;
}
