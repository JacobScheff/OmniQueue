"""Check numeric-token filtering, error statistics, and held-out reports."""

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from ml.data import CLOSED, DOWN, MISSING, OVERFLOW, TOKEN_LABELS, UNKNOWN, save_dataset
from ml.model import WaitPredictor
from ml.test import HORIZON_MINUTES, evaluate, numeric_wait_errors, print_horizon, summarize_wait_errors


class WaitErrorTests(unittest.TestCase):
    def test_only_numeric_pairs_include_zero_and_300_but_exclude_overflow(self):
        predictions = torch.tensor([0, 60, 0, OVERFLOW, 1, DOWN, 1, 1, MISSING, 3, 1])
        targets = torch.tensor([0, 0, 60, 1, OVERFLOW, 1, CLOSED, UNKNOWN, 1, MISSING, DOWN])
        errors = numeric_wait_errors(predictions, targets)
        self.assertEqual(errors.tolist(), [0, 300, -300])

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
            print_horizon(240, {"accuracy": None, "forecast_count": 0,
                                "wait_errors": {**empty, "actual_numeric_wait_count": 0}})
        self.assertIn("accuracy N/A | MAE N/A", output.getvalue())
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
            self.assertEqual(tuple(result["horizons"]), HORIZON_MINUTES)
            stats = result["horizons"][5]["wait_errors"]
            self.assertEqual(stats["count"], 4)  # +5, 0, -5, +10 minutes.
            self.assertEqual(stats["actual_numeric_wait_count"], 4)
            self.assertEqual(stats["absolute_minutes"]["mean"], 5)
            self.assertEqual(stats["signed_minutes"]["mean"], 2.5)
            self.assertEqual(stats["overpredictions"]["count"], 2)
            self.assertEqual(stats["underpredictions"]["count"], 1)
            self.assertAlmostEqual(result["horizons"][5]["accuracy"], 1 / 6)
            self.assertEqual(result["horizons"][5]["scored_targets"], 6)
            # Closure scoring is appended after the original report, never within horizon rows.
            self.assertNotIn("detection:", output.getvalue())
            self.assertNotIn("statuses", result["horizons"][5])
            self.assertLess(output.getvalue().index("10 hours:"),
                            output.getvalue().index("Closure/reopening timing"))
            for label in ("5 minutes:", "15 minutes:", "30 minutes:", "1 hour:",
                          "2 hours:", "4 hours:", "6 hours:", "8 hours:", "10 hours:"):
                self.assertIn(label, output.getvalue())
            self.assertEqual(result["transitions"]["closing"]["actual_events"], 5)
            self.assertEqual(result["transitions"]["closing"]["missed_events"], 5)
            self.assertEqual(result["transitions"]["closing"]["accuracy"], 0.)
            self.assertEqual(result["transitions"]["reopening"]["actual_events"], 1)
            self.assertEqual(result["transitions"]["reopening"]["accuracy"], 1.)
            self.assertEqual(result["horizons"][5]["forecast_count"], 7)
            self.assertIsNone(result["horizons"][240]["accuracy"])
            for expected in ("MAE 5.00 min", "RMSE", "bias +2.50 min", "numeric pairs 4/4"):
                self.assertIn(expected, output.getvalue())
            self.assertNotIn("predicted ", output.getvalue())
            self.assertNotIn("errors > 40", output.getvalue())
            # Non-wait predictions count against accuracy but not numeric errors.
            with torch.no_grad():
                model.output.bias.zero_()
                model.output.bias[DOWN] = 10
            checkpoint["model_state"] = model.state_dict()
            torch.save(checkpoint, settings.CHECKPOINT)
            output = io.StringIO()
            with redirect_stdout(output):
                result = evaluate(settings)
            self.assertEqual(result["horizons"][5]["wait_errors"]["count"], 0)
            self.assertEqual(result["horizons"][5]["wait_errors"]["actual_numeric_wait_count"], 4)
            self.assertIn("numeric pairs 0/4", output.getvalue())
            self.assertIn("MAE N/A", output.getvalue())


if __name__ == "__main__":
    unittest.main()
