"""Historical comparison regressions without loading ML models or the DES."""
from datetime import date
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Park import config
from Park.comparison.forecast_worker import visible_count
from Park.comparison.history import HistoricalDay, load_day
from Park.comparison.runner import Choice, Party, run_policy

N = config.NUM_RIDES


class Walks:
    def __init__(self):
        self.distances = [[0.0]*(N+1) for _ in range(N+1)]

    def matrix(self, location):
        indices = list(range(N))+[location, N]
        return [[self.distances[i][j] for j in indices] for i in indices]


def profile():
    return dict(preferences=[1/N]*N, must_dos=[0], speed=1.0)


class PickNext:
    name = "test"

    def __init__(self, order):
        self.order = order
        self.seen = []

    def choose(self, state, board):
        self.seen.append((state, board))
        return Choice(next((j for j in self.order if j not in state.completed and board[j] is not None), None))


class CompareTests(unittest.TestCase):
    def day(self, rows, closing=None):
        times = [100000+i*300 for i in range(len(rows))]
        return HistoricalDay("2026-11-18", times, [r+[None]*(N-len(r)) for r in rows],
                             closing or times[-1]+300, {})

    def test_yesterday_and_complete_day_fallback(self):
        from Park.companion.server.ride_map import KNOWN_ENTITY_IDS
        day = dict(timestamps=[100000], expected_timestamps=[100000],
                   rides=[dict(id=KNOWN_ENTITY_IDS[0], name="Ride")], tokens=[[1]], closing_at=100300)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"archive.gz"
            with gzip.open(path, "wt") as target:
                json.dump(dict(version=1, days={"2026-11-17": day, "2026-11-18": day},
                               incomplete_days={"2026-11-19": day}), target)
            self.assertEqual(load_day(path, today=date(2026, 11, 19)).key, "2026-11-18")
            self.assertEqual(load_day(path, today=date(2026, 11, 21)).key, "2026-11-18")
            with self.assertRaisesRegex(ValueError, "No complete cached day"):
                load_day(path, "2026-11-19")

    def test_interpolated_history_is_hidden_until_right_endpoint(self):
        day = self.day([[0], [600], [0]])
        tick = day.timestamps[1]
        day.interpolated[tick] = day.timestamps[2]
        self.assertIsNone(day.board_at(tick, observed=True)[0])
        self.assertEqual(day.board_at(tick)[0], 600)
        raw = dict(timestamps=day.timestamps, interpolated_ticks={str(tick): day.timestamps[::2]})
        self.assertEqual(visible_count(raw, tick+299), 1)
        self.assertEqual(visible_count(raw, day.timestamps[2]), 3)

    def test_completed_rides_position_and_clock_reach_next_decision(self):
        day = self.day([[0, 0]]*8)
        walks = Walks()
        walks.distances[N][0] = 10
        walks.distances[0][1] = 25
        policy = PickNext([0, 1])
        result = run_policy(day, walks, policy, profile())
        second = policy.seen[1][0]
        self.assertEqual(second.completed, {0})
        self.assertEqual(second.location, 0)
        self.assertEqual(second.now, day.opening+10+config.RIDES[0]["duration_sec"])
        self.assertEqual(result["metrics"]["rides_completed"], 2)
        self.assertEqual(sum(e["end"]-e["start"] for e in result["events"]), day.closing-day.opening)

    def test_wait_is_sampled_at_arrival_not_decision(self):
        day = self.day([[0], [600], [600], [600], [600], [600], [600], [600]])
        walks = Walks()
        walks.distances[N][0] = 301
        result = run_policy(day, walks, PickNext([0]), profile())
        queued = next(e for e in result["events"] if e["kind"] == "queue")
        self.assertEqual(queued["posted_wait_sec"], 600)
        self.assertEqual(queued["start"], day.opening+301)

    def test_queue_and_on_ride_interruptions_do_not_complete(self):
        for wait in (0, 600):
            with self.subTest(wait=wait):
                day = self.day([[wait], [None], [None], [None]])
                rides = [dict(r) for r in config.RIDES]
                rides[0]["duration_sec"] = 700
                with patch.object(config, "RIDES", rides):
                    result = run_policy(day, Walks(), PickNext([0]), profile())
                self.assertEqual(result["metrics"]["rides_completed"], 0)
                aborted = next(e for e in result["events"] if e["kind"] == "interrupted")
                self.assertEqual(aborted["start"], day.timestamps[1])

    def test_unexpected_long_wait_reserves_walk_out_and_no_completion(self):
        day = self.day([[3000]]*5)
        walks = Walks()
        walks.distances[0][N] = 123
        result = run_policy(day, walks, PickNext([0]), profile())
        self.assertEqual(result["metrics"]["exit_time"], day.closing)
        self.assertEqual(result["metrics"]["rides_completed"], 0)
        self.assertEqual(result["events"][-1]["kind"], "walk")
        self.assertEqual(result["events"][-1]["end"]-result["events"][-1]["start"], 123)

    def test_idle_policy_keeps_observing_the_full_day(self):
        day = self.day([[0]]*4)
        policy = PickNext([])
        result = run_policy(day, Walks(), policy, profile())
        self.assertGreater(len(policy.seen), 3)
        self.assertEqual(result["events"][-1]["end"], day.closing)
        self.assertEqual(result["metrics"]["idle_sec"], day.closing-day.opening)

    def test_repeat_selection_rejected(self):
        day = self.day([[0]]*8)
        class Repeat:
            name = "repeat"
            def choose(self, state, board):
                return Choice(0)
        with self.assertRaisesRegex(ValueError, "completed ride"):
            run_policy(day, Walks(), Repeat(), profile())

    def test_dp_rebuilds_context_from_latest_history_position_and_completions(self):
        from Park.comparison.policies import DPPolicy
        from Park.companion.server.ride_map import KNOWN_ENTITY_IDS
        class Forecasts:
            def __init__(self):
                self.origins = []
            def forecast(self, now):
                self.origins.append(now)
                return dict(as_of="2026-11-18T16:00:00+00:00", history_ticks=len(self.origins),
                            closing_timestamp=1795021200,
                            snapshots=[dict(time="2026-11-18T16:00:00+00:00", rides=[
                                dict(id=KNOWN_ENTITY_IDS[0], wait_minutes={"p90": 5})])])
        forecasts = Forecasts()
        walks = Walks()
        walks.distances[0][1] = 456
        policy = DPPolicy(walks, profile(), 1795021200, forecasts)
        contexts = []
        def planner(ctx):
            contexts.append(ctx)
            return dict(routes=[], status="uncertified")
        with patch("Park.comparison.policies.plan", side_effect=planner):
            policy.choose(Party(1795017600), [0]*N)
            choice = policy.choose(Party(1795018200, 0, {0}), [0]*N)
        self.assertEqual(forecasts.origins, [1795017600, 1795018200])
        self.assertTrue(contexts[1].rides[0].completed)
        self.assertEqual(contexts[1].distances[N][1], 456)
        self.assertEqual(choice.detail["forecast_history_ticks"], 2)
        self.assertIsNone(choice.ride_id)

    def test_rl_observation_uses_shared_walks_and_recorded_hours(self):
        import numpy as np
        from Park.comparison.policies import RLPolicy
        from Park.training.features import GUEST_FEAT_DIM, RIDE_DYNAMIC_FEAT_DIM, FLAT_OBS_DIM
        class Graph:
            num_nodes = 100
            def node_to_idx(self, node):
                return 10
        # Exercise the real observation adapter without loading a checkpoint.
        policy = object.__new__(RLPolicy)
        policy.np, policy.park = np, Graph()
        policy.walks, policy.profile = Walks(), profile()
        policy.opening, policy.closing = 100000, 100000+16*3600
        policy.walks.distances[0][1] = 123
        state = Party(policy.opening+15*3600, 0, {0})
        obs = policy.observation(state, [0]*N)
        self.assertEqual(obs.shape, (FLAT_OBS_DIM,))
        self.assertAlmostEqual(obs[-3], 15/16)
        self.assertAlmostEqual(obs[36], 3600/config.DAY_SECONDS)
        rides = obs[GUEST_FEAT_DIM:-3].reshape(N, RIDE_DYNAMIC_FEAT_DIM)
        self.assertEqual(rides[0, 2], 0)  # completed ride is unavailable to RL too
        self.assertAlmostEqual(rides[1, 5], 123/3600)
        self.assertAlmostEqual(rides[0, 6], 1/10)


if __name__ == "__main__":
    unittest.main()
