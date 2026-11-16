"""Check numeric-token filtering, error statistics, and held-out reports."""

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from ml.data import CLOSED, DOWN, MISSING, OVERFLOW, TOKEN_LABELS, UNKNOWN, save_dataset
from ml.model import WaitPredictor
from ml.test import evaluate, large_wait_errors, numeric_wait_errors, print_wait_errors, summarize_wait_errors


class WaitErrorTests(unittest.TestCase):
    def test_only_numeric_pairs_include_zero_and_300_but_exclude_overflow(self):
        predictions = torch.tensor([0, 60, 0, OVERFLOW, 1, DOWN, 1, 1, MISSING, 3, 1])
        targets = torch.tensor([0, 0, 60, 1, OVERFLOW, 1, CLOSED, UNKNOWN, 1, MISSING, DOWN])
        errors = numeric_wait_errors(predictions, targets)
        self.assertEqual(errors.tolist(), [0, 300, -300])
        rides = [{"id": "ride", "name": "Ride"}]
        records = large_wait_errors(predictions.reshape(1, -1, 1), targets.reshape(1, -1, 1),
                                    [{"rides": rides, "timestamps": [1793548800 + 300 * i for i in range(12)]}], rides)
        self.assertEqual([row["error_minutes"] for row in records], [300, -300])

    def test_pooled_absolute_and_signed_statistics_and_directions(self):
        stats = summarize_wait_errors([-10, -5, 0, 5, 20])
        self.assertEqual(stats["count"], 5)
        absolute = stats["absolute_minutes"]
        self.assertEqual([absolute[key] for key in ("min", "q1", "median", "q3", "max")],
                         [0, 5, 5, 10, 20])
        self.assertEqual(absolute["mean"], 8)
        self.assertAlmostEqual(absolute["std"], math.sqrt(46))
        self.assertAlmostEqual(stats["rmse_minutes"], math.sqrt(110))
        self.assertEqual(stats["signed_minutes"]["mean"], 2)
        self.assertAlmostEqual(stats["signed_minutes"]["std"], math.sqrt(106))
        self.assertEqual(stats["overpredictions"],
                         {"count": 2, "fraction": .4, "mean_magnitude_minutes": 12.5})
        self.assertEqual(stats["underpredictions"],
                         {"count": 2, "fraction": .4, "mean_magnitude_minutes": 7.5})
        self.assertEqual(stats["exact_matches"], {"count": 1, "fraction": .2})

    def test_single_pair_and_empty_pairs_do_not_produce_nan(self):
        stats = summarize_wait_errors([5])
        self.assertEqual(stats["absolute_minutes"]["std"], 0)
        self.assertEqual(stats["signed_minutes"]["std"], 0)
        self.assertEqual(stats["underpredictions"]["count"], 0)
        self.assertIsNone(stats["underpredictions"]["mean_magnitude_minutes"])
        empty = summarize_wait_errors([])
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["absolute_minutes"])
        output = io.StringIO()
        with redirect_stdout(output):
            print_wait_errors({**empty, "actual_numeric_wait_count": 0})
        self.assertIn("N/A (no comparable numeric wait pairs)", output.getvalue())
        self.assertNotIn("nan", output.getvalue().lower())

    def test_evaluator_pools_pairs_across_days_and_excludes_padding_and_breaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = SimpleNamespace(DEVICE="cpu", BATCH_SIZE=2,
                                       DATASET=root / "data.json.gz", CHECKPOINT=root / "model.pt")
            rides = [{"id": "ride", "name": "Ride"}]

            def day(tokens, step=300):
                return {"rides": rides, "tokens": [[token] for token in tokens],
                        "timestamps": [1793548800 + step * index for index in range(len(tokens))]}

            save_dataset(settings.DATASET, {"version": 1, "park_id": "park", "days": {
                "2026-11-01": day([0, 1, 2, 3, DOWN, OVERFLOW, CLOSED]),
                "2026-11-02": day([0, 0]),
                "2026-11-03": day([0, 30], step=600),
            }})
            model_config = {"num_rides": 1, "embedding_size": 2, "hidden_size": 4}
            model = WaitPredictor(**model_config)
            with torch.no_grad():
                model.output.weight.zero_()
                model.output.bias.zero_()
                model.output.bias[2] = 10  # Always predict ten minutes.
            checkpoint = {"version": 1, "token_labels": TOKEN_LABELS, "park_id": "park",
                          "model_config": model_config, "model_state": model.state_dict(), "rides": rides,
                          "train_dates": ["2026-10-30"], "validation_dates": ["2026-10-31"],
                          "test_dates": ["2026-11-01", "2026-11-02", "2026-11-03"]}
            torch.save(checkpoint, settings.CHECKPOINT)
            output = io.StringIO()
            with redirect_stdout(output):
                result = evaluate(settings)
            stats = result["wait_errors"]
            self.assertEqual(stats["count"], 4)  # +5, 0, -5, +10 minutes.
            self.assertEqual(stats["actual_numeric_wait_count"], 4)
            self.assertEqual(stats["absolute_minutes"]["mean"], 5)
            self.assertEqual(stats["signed_minutes"]["mean"], 2.5)
            self.assertEqual(stats["overpredictions"]["count"], 2)
            self.assertEqual(stats["underpredictions"]["count"], 1)
            self.assertAlmostEqual(result["accuracy"], 1 / 6)
            for expected in ("Wait-time MAE 5.00 min", "Q1", "population std dev",
                             "Overpredictions: 2 (50.0%)", "Underpredictions: 1 (25.0%)",
                             "Exact wait matches: 1 (25.0%)"):
                self.assertIn(expected, output.getvalue())
            # Non-wait predictions count against accuracy but not numeric errors.
            with torch.no_grad():
                model.output.bias.zero_()
                model.output.bias[DOWN] = 10
            checkpoint["model_state"] = model.state_dict()
            torch.save(checkpoint, settings.CHECKPOINT)
            output = io.StringIO()
            with redirect_stdout(output):
                result = evaluate(settings)
            self.assertEqual(result["wait_errors"]["count"], 0)
            self.assertEqual(result["wait_errors"]["actual_numeric_wait_count"], 4)
            self.assertIn("0/4 actual numeric waits (0.0%)", output.getvalue())
            self.assertIn("Wait-time errors: N/A", output.getvalue())

    def test_large_errors_use_target_time_and_correct_ride_across_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = SimpleNamespace(DEVICE="cpu", BATCH_SIZE=2,
                                       DATASET=root / "data.json.gz", CHECKPOINT=root / "model.pt")
            rides = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]
            days = {}
            for index, (actual_a, actual_b) in enumerate(((12, 2), (11, 11), (20, 20))):
                key = f"2026-11-{index + 1:02}"
                start = int(datetime.fromisoformat(key + "T23:55:00-08:00").timestamp())
                days[key] = {
                    "rides": [{"id": "b", "name": f"Historical B {index}"},
                              {"id": "a", "name": f"Historical A {index}"}],
                    "timestamps": [start, start + 300], "tokens": [[0, 0], [actual_b, actual_a]],
                }
            save_dataset(settings.DATASET, {"version": 1, "park_id": "park", "days": days})
            model_config = {"num_rides": 2, "embedding_size": 2, "hidden_size": 4}
            model = WaitPredictor(**model_config)
            with torch.no_grad():
                model.output.weight.zero_()
                model.output.bias.zero_()
                model.output.bias.reshape(2, -1)[0, 20] = 10  # A: 100 min.
                model.output.bias.reshape(2, -1)[1, 2] = 10   # B: 10 min.
            torch.save({"version": 1, "token_labels": TOKEN_LABELS, "park_id": "park",
                        "model_config": model_config, "model_state": model.state_dict(), "rides": rides,
                        "train_dates": ["2026-10-30"], "validation_dates": ["2026-10-31"],
                        "test_dates": sorted(days)}, settings.CHECKPOINT)
            output = io.StringIO()
            with redirect_stdout(output):
                result = evaluate(settings)
            records = result["large_wait_errors"]
            self.assertEqual([row["error_minutes"] for row in records], [45, -45, -90])
            self.assertEqual([row["ride_name"] for row in records],
                             ["Historical A 1", "Historical B 1", "Historical B 2"])
            self.assertEqual([row["time"] for row in records],
                             ["2026-11-03T00:00:00-08:00"] * 2 + ["2026-11-04T00:00:00-08:00"])
            self.assertIn("Wait-time errors > 40 min: 3", output.getvalue())
            self.assertIn("2026-11-03 00:00 PST | Historical A 1 | predicted 100 min | actual 55 min", output.getvalue())
            self.assertIn("2026-11-04 00:00 PST | Historical B 2 | predicted 10 min | actual 100 min", output.getvalue())


if __name__ == "__main__":
    unittest.main()
