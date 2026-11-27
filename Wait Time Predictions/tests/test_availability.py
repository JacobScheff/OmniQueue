"""Outage duration, scheduled closure outputs, and exact raw wait semantics."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ml.availability import fit_breakdown_hazards, history_ages
from ml.backfill import PARK_ID, reencode_cached_history
from ml.data import CLOSED, DOWN, REFURBISHMENT, UNKNOWN, VOCAB_SIZE, load_dataset, ride_token, save_dataset
from ml.model import WaitPredictor, token_loss
from ml.run import summarize
from ml.test import forecast_day


RIDES = [{"id": "ride", "name": "Ride"}]


def day(values):
    return {"rides": RIDES, "timestamps": [1793548800 + 300 * i for i in range(len(values))],
            "tokens": [[value] for value in values]}


def constant_model(**kwargs):
    model = WaitPredictor(1, embedding_size=2, hidden_size=4, **kwargs)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(-30)
        model.output.bias[2] = 30
    return model


class AvailabilityTests(unittest.TestCase):
    def test_only_exact_thirteen_maps_to_five(self):
        for minutes, token in ((5, 1), (12, 2), (13, 1), (14, 3), (15, 3), (130, 26)):
            self.assertEqual(ride_token({"status": "OPERATING", "wait": minutes}), token)
        self.assertEqual(ride_token({"status": "DOWN", "wait": 13}), DOWN)

    def test_closed_is_a_trainable_target_for_new_models(self):
        logits = torch.zeros(1, 1, 4, VOCAB_SIZE, requires_grad=True)
        targets = torch.tensor([[[DOWN, CLOSED, UNKNOWN, REFURBISHMENT]]])
        loss, count = token_loss(logits, targets, predict_closed=True)
        loss.backward()
        self.assertEqual(count.item(), 2)
        self.assertGreater(logits.grad[0, 0, 1].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 0, 2:].abs().sum().item(), 0)

    def test_duration_learning_and_censoring(self):
        training = [day([2] + [DOWN] * 6 + [2])] * 20
        hazards = fit_breakdown_hazards(training, RIDES)
        self.assertLess(hazards[0][0], .05)
        self.assertGreater(hazards[0][5], .8)
        self.assertIsNone(fit_breakdown_hazards([day([DOWN, DOWN, 2])], RIDES))
        self.assertIsNone(fit_breakdown_hazards([day([2, DOWN, CLOSED])], RIDES))
        broken = day([2, DOWN, 2])
        broken["timestamps"][-1] += 300
        self.assertIsNone(fit_breakdown_hazards([broken], RIDES))
        interpolated = day([2, DOWN, 2])
        interpolated["interpolated_ticks"] = {str(interpolated["timestamps"][1]): [0, 1000]}
        self.assertIsNone(fit_breakdown_hazards([interpolated], RIDES))

    def test_elapsed_duration_delays_reopening_even_with_neural_wait_bias(self):
        model = constant_model(predict_closed=True, breakdown_hazards=[[0., 0., 0., 1.]])
        short = torch.tensor([[2], [DOWN]])
        paths = model.forecast(short, torch.zeros(6, 6), steps=5, samples=4)
        self.assertEqual(paths[0, :, 0].tolist(), [DOWN, DOWN, DOWN, 2, 2])
        older = torch.tensor([[2], [DOWN], [DOWN], [DOWN], [DOWN]])
        paths = model.forecast(older, torch.zeros(5, 6), steps=1, samples=4)
        self.assertTrue((paths == 2).all())
        self.assertEqual(history_ages(older, [0, 300, 600, 900, 1500]).tolist(), [1])

    def test_evaluator_uses_median_remaining_duration_and_matches_batches(self):
        model = constant_model(predict_closed=True, breakdown_hazards=[[.2] * 48])
        source = day([DOWN] * 8 + [2] * 3)
        def indexed(batch):
            return {(minutes, int(origin)): prediction.tolist()
                    for minutes, origins, predictions, _ in forecast_day(
                        model, source, RIDES, torch.device("cpu"), batch)
                    for origin, prediction in zip(origins, predictions)}
        small = indexed(2)
        self.assertEqual(small, indexed(20))
        self.assertEqual(small[5, 0], [DOWN])
        self.assertEqual(small[15, 0], [DOWN])
        self.assertEqual(small[30, 0], [2])

    def test_closed_outputs_and_legacy_compatibility(self):
        for enabled, expected in ((True, CLOSED), (False, 2)):
            model = constant_model(predict_closed=enabled)
            with torch.no_grad():
                model.output.bias[CLOSED] = 100
            paths = model.forecast(torch.tensor([[2]]), torch.zeros(2, 6), steps=2, samples=4)
            self.assertTrue((paths == expected).all())

    def test_calibrated_evaluator_does_not_read_future_statuses(self):
        model = constant_model(predict_closed=True, breakdown_hazards=[[.2] * 48])
        original = day([2, DOWN] + [DOWN] * 7)
        changed = day([2, DOWN] + [CLOSED] * 7)
        def origin_forecasts(source):
            return {minutes: prediction[origins == 1].tolist()
                    for minutes, origins, prediction, _ in forecast_day(
                        model, source, RIDES, torch.device("cpu"), 20) if (origins == 1).any()}
        self.assertEqual(origin_forecasts(original), origin_forecasts(changed))

    def test_majority_unavailable_hides_conditional_wait(self):
        for values, status in (([DOWN, DOWN, 2], "DOWN"), ([CLOSED, CLOSED, 2], "CLOSED"),
                               ([CLOSED, DOWN, 2], "CLOSED")):
            row = summarize(torch.tensor(values).view(-1, 1, 1), RIDES, [0])[0]["rides"][0]
            self.assertEqual(row["status"], status)
            self.assertIsNone(row["wait_minutes"])

    def test_offline_reencoding_uses_raw_minutes_and_preserves_fifteen(self):
        source = day([3, 3])
        source.update(source="themeparks.wiki/history")
        times = ["2026-11-01T08:00:00-08:00", "2026-11-01T08:05:00-08:00"]
        states = [{"time": time, "status": "OPERATING", "queue": {"STANDBY": {"waitTime": wait}}}
                  for time, wait in zip(times, [13, 15])]
        document = {"id": PARK_ID, "entityType": "PARK", "range": {"from": "2026-11-01"},
                    "entities": [{**RIDES[0], "entityType": "ATTRACTION",
                                  "opening": {}, "history": states}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "waits.json.gz"
            save_dataset(path, {"version": 1, "days": {"2026-11-01": source}})
            with gzip.open(root / "2026-11-01.json.gz", "wt", encoding="utf-8") as target:
                json.dump(document, target)
            reencode_cached_history(path, root)
            self.assertEqual(load_dataset(path)["days"]["2026-11-01"]["tokens"], [[1], [3]])
            self.assertEqual(len(list(root.glob("*.bak"))), 1)


if __name__ == "__main__":
    unittest.main()
