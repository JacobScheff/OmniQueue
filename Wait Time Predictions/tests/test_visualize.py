"""Checks for archive rendering, location caching, and visualizer inference."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from ml.data import CLOSED, DOWN, MISSING, OVERFLOW, TOKEN_LABELS, UNKNOWN, save_dataset
from ml.model import WaitPredictor
from ml.visualize import PARK_ID, Visualizer, load_locations, observed_ride


class VisualizerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.ticks = [1793548800 + index * 300 for index in range(6)]
        self.rides = [{"id": "ride-a", "name": "Ride A"}]
        self.day = {"timestamps": self.ticks, "expected_timestamps": self.ticks,
                    "rides": self.rides, "tokens": [[2]] * 6}
        self.settings = SimpleNamespace(DATASET=self.root / "data.json.gz", CHECKPOINT=self.root / "waits.pt",
                                        DEVICE="cpu", SAMPLES=4, SEED=42, TEMPERATURE=1.0)
        self.save()
        model = WaitPredictor(1, embedding_size=4, hidden_size=8)
        torch.save({"version": 1, "park_id": PARK_ID, "rides": self.rides, "token_labels": TOKEN_LABELS,
                    "model_config": {"num_rides": 1, "embedding_size": 4, "hidden_size": 8},
                    "model_state": model.state_dict()}, self.settings.CHECKPOINT)
        self.app = Visualizer(self.settings)

    def tearDown(self):
        self.app.close()
        self.directory.cleanup()

    def save(self):
        save_dataset(self.settings.DATASET, {"version": 1, "park_id": PARK_ID,
                                            "days": {"2026-11-01": self.day}, "incomplete_days": {}})

    def result(self, job):
        self.app.jobs[job["job"]][1].result(timeout=10)
        return self.app.job_payload(job["job"])

    def test_observed_waits_preserve_zero_overflow_and_statuses(self):
        for token, status, wait in [(0, "OPERATING", 0), (2, "OPERATING", 10),
                                    (OVERFLOW, "OPERATING", "300+"), (DOWN, "DOWN", None),
                                    (CLOSED, "CLOSED", None), (UNKNOWN, "UNKNOWN", None),
                                    (MISSING, "MISSING", None)]:
            with self.subTest(token=token):
                result = observed_ride(self.rides[0], token)
                self.assertEqual((result["status"], result["wait"]), (status, wait))

    def test_day_payload_distinguishes_interpolation_from_recorded_data(self):
        self.day["timestamps"] = self.ticks[:2] + self.ticks[3:]
        self.day["tokens"] = [[0], [0], [3], [4], [5]]
        self.save()
        result = self.app.day_payload("2026-11-01")
        self.assertEqual(result["timestamps"], self.ticks)
        self.assertEqual(result["snapshots"][1]["source"], "observed")
        self.assertEqual(result["snapshots"][2]["source"], "interpolated")
        self.assertEqual(result["snapshots"][2]["endpoints"], [self.ticks[1], self.ticks[3]])
        self.assertEqual(result["snapshots"][2]["rides"][0]["wait"], 10)

    def test_async_forecast_reuses_jobs_and_does_not_consume_future_actuals(self):
        with redirect_stdout(io.StringIO()) as output:
            first_job = self.app.start_forecast("2026-11-01", "08:15")
            first = self.result(first_job)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(first["state"], "complete")
        self.assertEqual([row["timestamp"] for row in first["snapshots"]], self.ticks[3:])
        self.assertTrue(all(row["source"] == "predicted" for row in first["snapshots"]))
        self.assertEqual(first_job, self.app.start_forecast("2026-11-01", "08:15"))
        self.day["tokens"][3:] = [[DOWN], [DOWN], [DOWN]]
        self.save()
        second_job = self.app.start_forecast("2026-11-01", "08:15")
        self.assertNotEqual(second_job, first_job)
        second = self.result(second_job)
        self.assertEqual(first, second)

    def test_in_progress_day_can_forecast_after_a_repaired_gap(self):
        expected = self.ticks + [self.ticks[-1] + 300 * i for i in range(1, 8)]
        self.day["expected_timestamps"] = expected
        self.day["timestamps"] = self.ticks[:2] + self.ticks[3:]
        self.day["tokens"] = [[2]] * 5
        save_dataset(self.settings.DATASET, {"version": 1, "park_id": PARK_ID,
                                            "days": {}, "incomplete_days": {"2026-11-01": self.day}})
        payload = self.app.day_payload("2026-11-01")
        self.assertEqual(len(payload["snapshots"]), 6)
        self.assertEqual(payload["snapshots"][2]["source"], "interpolated")
        result = self.result(self.app.start_forecast("2026-11-01", "08:20"))
        self.assertEqual(result["state"], "complete")
        self.assertEqual([row["timestamp"] for row in result["snapshots"]], expected[4:])

    def test_forecast_fails_if_interpolation_requires_a_future_endpoint(self):
        self.day["timestamps"] = [self.ticks[0], self.ticks[3], self.ticks[4], self.ticks[5]]
        self.day["tokens"] = [[2]] * 4
        self.save()
        job = self.app.start_forecast("2026-11-01", "08:15")
        with self.assertRaisesRegex(ValueError, "Missing 2"):
            self.result(job)

    def test_invalid_days_times_and_missing_checkpoints_have_actionable_errors(self):
        with self.assertRaisesRegex(ValueError, "Choose a day"):
            self.app.day_payload("2026-12-01")
        for time in ("08:00", "10:00"):
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, "after opening"):
                self.app.start_forecast("2026-11-01", time)
        self.settings.CHECKPOINT = self.root / "missing.pt"
        with self.assertRaisesRegex(ValueError, "ml/train.py"):
            self.app.start_forecast("2026-11-01", "08:15")

    def test_coordinates_are_cached_and_survive_a_refresh_failure(self):
        import httpx

        path = self.root / "locations.json"
        response = Mock()
        response.json.return_value = {"id": PARK_ID, "children": [
            {"id": "ride-a", "entityType": "ATTRACTION", "location": {"latitude": 33.81, "longitude": -117.92}},
            {"id": "bad", "entityType": "ATTRACTION", "location": {"latitude": 0, "longitude": 0}},
        ]}
        with patch("httpx.get", return_value=response) as get:
            locations, warning = load_locations(path)
            self.assertEqual(warning, "")
            self.assertEqual(list(locations), ["ride-a"])
            self.assertEqual(json.loads(path.read_text()), locations)
            self.assertEqual(load_locations(path), (locations, ""))
            get.assert_called_once()
        with patch("httpx.get", side_effect=httpx.ConnectError("Offline")):
            saved, warning = load_locations(path, refresh=True)
            self.assertEqual(saved, locations)
            self.assertIn("saved locations", warning)


if __name__ == "__main__":
    unittest.main()
