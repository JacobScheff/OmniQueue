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
        .def_readonly("must_dos_assigned", &park::DayMetricsResult::must_dos_assigned)
        .def_readonly("must_dos_completed", &park::DayMetricsResult::must_dos_completed)
        .def_readonly("preference_score_sum", &park::DayMetricsResult::preference_score_sum)
        .def_readonly("must_do_latency_sum_sec", &park::DayMetricsResult::must_do_latency_sum_sec)
        .def_readonly("must_do_latency_count", &park::DayMetricsResult::must_do_latency_count)
        .def("rides_per_party", &park::DayMetricsResult::rides_per_party)
        .def("avg_wait_variance", &park::DayMetricsResult::avg_wait_variance)
        .def("must_do_completion_rate", &park::DayMetricsResult::must_do_completion_rate)
        .def("avg_preference_score_per_guest", &park::DayMetricsResult::avg_preference_score_per_guest)
        .def("avg_must_do_latency_sec", &park::DayMetricsResult::avg_must_do_latency_sec)
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

    py::class_<park::PersonalDayStats>(m, "PersonalDayStats")
        .def_readonly("n_focals", &park::PersonalDayStats::n_focals)
        .def_readonly("must_dos_assigned", &park::PersonalDayStats::must_dos_assigned)
        .def_readonly("must_dos_completed", &park::PersonalDayStats::must_dos_completed)
        .def_readonly("preference_score_sum", &park::PersonalDayStats::preference_score_sum)
        .def_readonly("rides_completed", &park::PersonalDayStats::rides_completed)
        .def_property_readonly(
            "must_do_completion_rate",
            [](const park::PersonalDayStats& self) {
                if (self.must_dos_assigned <= 0) {
                    return 0.0;
                }
                return static_cast<double>(self.must_dos_completed) /
                       static_cast<double>(self.must_dos_assigned);
            })
        .def_property_readonly(
            "avg_preference_score_per_guest",
            [](const park::PersonalDayStats& self) {
                if (self.n_focals <= 0) {
                    return 0.0;
                }
                return self.preference_score_sum / static_cast<double>(self.n_focals);
            });

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
        })
        .def_property_readonly("party_ids", [](const park::RolloutBatchResult& self) {
            const py::ssize_t n = static_cast<py::ssize_t>(self.party_ids.size());
            py::array_t<int32_t> arr(n);
            if (n > 0) {
                std::memcpy(
                    arr.mutable_data(),
                    self.party_ids.data(),
                    self.party_ids.size() * sizeof(int32_t));
            }
            return arr;
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

    py::class_<park::FocalPartyConfig>(m, "FocalPartyConfig")
        .def(py::init<>())
        .def_readwrite("spawn_sec", &park::FocalPartyConfig::spawn_sec)
        .def_readwrite("leave_sec", &park::FocalPartyConfig::leave_sec)
        .def_readwrite("distance_preference", &park::FocalPartyConfig::distance_preference)
        .def_property(
            "preference_weights",
            [](const park::FocalPartyConfig& c) {
                py::array_t<float> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), c.preference_weights.data(), park::kNumRides * sizeof(float));
                return arr;
            },
            [](park::FocalPartyConfig& c, const py::array_t<float>& arr) {
                if (arr.size() != park::kNumRides) {
                    throw std::invalid_argument("preference_weights must have length NUM_RIDES");
                }
                std::memcpy(c.preference_weights.data(), arr.data(), park::kNumRides * sizeof(float));
            })
        .def_property(
            "must_dos",
            [](const park::FocalPartyConfig& c) {
                py::array_t<uint8_t> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), c.must_dos.data(), park::kNumRides * sizeof(uint8_t));
                return arr;
            },
            [](park::FocalPartyConfig& c, const py::array_t<uint8_t>& arr) {
                if (arr.size() != park::kNumRides) {
                    throw std::invalid_argument("must_dos must have length NUM_RIDES");
                }
                std::memcpy(c.must_dos.data(), arr.data(), park::kNumRides * sizeof(uint8_t));
            });

    py::class_<park::FocalPartyStats>(m, "FocalPartyStats")
        .def_readonly("party_id", &park::FocalPartyStats::party_id)
        .def_readonly("spawn_sec", &park::FocalPartyStats::spawn_sec)
        .def_readonly("leave_sec", &park::FocalPartyStats::leave_sec)
        .def_readonly("exit_sec", &park::FocalPartyStats::exit_sec)
        .def_readonly("rides_completed", &park::FocalPartyStats::rides_completed)
        .def_readonly("must_dos_assigned", &park::FocalPartyStats::must_dos_assigned)
        .def_readonly("must_dos_completed", &park::FocalPartyStats::must_dos_completed)
        .def_readonly("top3_hits", &park::FocalPartyStats::top3_hits)
        .def_readonly("preference_score", &park::FocalPartyStats::preference_score)
        .def_readonly("exited", &park::FocalPartyStats::exited)
        .def_property_readonly(
            "preferences",
            [](const park::FocalPartyStats& s) {
                py::array_t<float> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), s.preferences.data(), park::kNumRides * sizeof(float));
                return arr;
            })
        .def_property_readonly(
            "must_dos_initial",
            [](const park::FocalPartyStats& s) {
                py::array_t<uint8_t> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), s.must_dos_initial.data(), park::kNumRides * sizeof(uint8_t));
                return arr;
            })
        .def_readonly("completions", &park::FocalPartyStats::completions);

    py::class_<park::PlayStepResult>(m, "PlayStepResult")
        .def_readonly("done", &park::PlayStepResult::done)
        .def_readonly("needs_human", &park::PlayStepResult::needs_human)
        .def_readonly("needs_ppo_batch", &park::PlayStepResult::needs_ppo_batch)
        .def_readonly("now_sec", &park::PlayStepResult::now_sec)
        .def_readonly("focal_party_id", &park::PlayStepResult::focal_party_id)
        .def_readonly("human_obs", &park::PlayStepResult::human_obs)
        .def_readonly("n_ppo", &park::PlayStepResult::n_ppo)
        .def_readonly("metrics", &park::PlayStepResult::metrics)
        .def_readonly("focal", &park::PlayStepResult::focal)
        .def_property_readonly("ppo_obs", [](const park::PlayStepResult& self) {
            return vector_to_obs_batch(self.ppo_obs, self.n_ppo);
        })
        .def_property_readonly("ppo_party_ids", [](const park::PlayStepResult& self) {
            py::array_t<int32_t> arr(static_cast<py::ssize_t>(self.ppo_party_ids.size()));
            if (!self.ppo_party_ids.empty()) {
                std::memcpy(
                    arr.mutable_data(),
                    self.ppo_party_ids.data(),
                    self.ppo_party_ids.size() * sizeof(int32_t));
            }
            return arr;
        });

    py::class_<park::PlayDayResult>(m, "PlayDayResult")
        .def_readonly("metrics", &park::PlayDayResult::metrics)
        .def_readonly("recording", &park::PlayDayResult::recording)
        .def_readonly("focal", &park::PlayDayResult::focal);

    py::class_<park::ParkEnv>(m, "ParkEnv")
        .def(py::init<uint64_t>(), py::arg("seed") = 0)
        .def("reset", &park::ParkEnv::reset, py::arg("seed"))
        .def(
            "reset_personal",
            &park::ParkEnv::reset_personal,
            py::arg("seed"),
            py::arg("n_focals"),
            "Start a personal-planner day: N focal parties + heuristic crowd.")
        .def("personal_stats", &park::ParkEnv::personal_stats)
        .def("step", &park::ParkEnv::step, py::arg("action"))
        .def(
            "exchange_batch",
            &park::ParkEnv::exchange_batch,
            py::arg("actions"),
            py::arg("max_obs"),
            "Apply a batch of actions, then collect up to max_obs pending observations.")
        .def(
            "reset_play",
            [](park::ParkEnv& env,
               uint64_t seed,
               const park::FocalPartyConfig& focal,
               bool crowd_auto_heuristic,
               int focal_policy,
               bool soft_human_leave,
               bool enable_recording,
               int sample_interval_sec) {
                env.reset_play(
                    seed,
                    focal,
                    crowd_auto_heuristic,
                    focal_policy,
                    soft_human_leave,
                    enable_recording,
                    sample_interval_sec);
            },
            py::arg("seed"),
            py::arg("focal"),
            py::arg("crowd_auto_heuristic") = true,
            py::arg("focal_policy") = 0,
            py::arg("soft_human_leave") = true,
            py::arg("enable_recording") = true,
            py::arg("sample_interval_sec") = 60,
            "Start a hybrid play/shadow session. focal_policy: 0=human, 1=heuristic, 2=ppo.")
        .def("play_advance", &park::ParkEnv::play_advance)
        .def("play_apply_human_action", &park::ParkEnv::play_apply_human_action, py::arg("action"))
        .def("play_apply_ppo_actions", &park::ParkEnv::play_apply_ppo_actions, py::arg("actions"))
        .def(
            "play_update_focal_preferences",
            &park::ParkEnv::play_update_focal_preferences,
            py::arg("focal"),
            "Update focal preference weights / must-dos mid-day without resetting location or history.")
        .def(
            "play_focal_state",
            &park::ParkEnv::play_focal_state,
            "Focal PartyState as int: Walking=1, InQueue=2, OnRide=4, Evacuating=8, Exited=16.")
        .def(
            "play_focal_ride_history",
            [](const park::ParkEnv& env) {
                const auto hist = env.play_focal_ride_history();
                py::array_t<int16_t> arr(park::kNumRides);
                std::memcpy(arr.mutable_data(), hist.data(), park::kNumRides * sizeof(int16_t));
                return arr;
            },
            "Per-ride completion counts for the focal guest.")
        .def(
            "play_recording",
            &park::ParkEnv::play_recording,
            py::return_value_policy::reference_internal)
        .def("play_focal_stats", &park::ParkEnv::play_focal_stats)
        .def("play_now_sec", &park::ParkEnv::play_now_sec)
        .def("play_focal_party_id", &park::ParkEnv::play_focal_party_id)
        .def("play_done", &park::ParkEnv::play_done);

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
        "run_play_day",
        &park::run_play_day,
        py::arg("seed"),
        py::arg("focal"),
        py::arg("sample_interval_sec") = 60,
        py::arg("record") = true,
        "Heuristic crowd + heuristic focal day with a custom focal guest.");
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
           double rand_u01,
           float distance_preference) {
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
            in.distance_preference = distance_preference;
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
        py::arg("distance_preference") = 1.0f,
        "Deterministic heuristic routing helper for unit tests.");
    m.def("is_available", []() { return true; });
    m.attr("HAS_EXCHANGE_BATCH") = true;
}
