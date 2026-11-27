"""Availability marginals, observed outages, and causal forecast decoding."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ml.availability import decode_availability, wait_quantiles
from ml.data import CLOSED, DOWN, OVERFLOW, REFURBISHMENT, UNKNOWN, VOCAB_SIZE, TOKEN_LABELS, save_dataset
from ml.model import build_model
from ml.run import predict, summarize_probabilities


RIDES = [{"id": "ride", "name": "Ride"}]


def distribution(down=0., closed=0.):
    values = torch.zeros(1, VOCAB_SIZE)
    values[:, 1] = 1 - down - closed
    values[:, DOWN], values[:, CLOSED] = down, closed
    return values


def constant_model(rides=1):
    model = build_model(num_rides=rides, embedding_size=4, hidden_size=8,
                        architecture="ride_horizon_v3", attention_heads=2, dropout=0,
                        predict_closed=True, breakdown_hazards=[[.1] * 48] * rides).eval()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(-40)
        model.output.bias[1] = 40
        model.local_gate.weight.zero_()
        model.local_gate.bias.fill_(-40)
        model.availability_output.weight.zero_()
        model.availability_output.bias.copy_(torch.tensor([.85, .15, 1e-12]).log())
    return model


class TemporalForecastTests(unittest.TestCase):
    def test_predicted_breakdowns_cannot_accumulate_into_majority_downtime(self):
        model = constant_model()
        history = torch.ones(2, 1, dtype=torch.long)
        calendar = torch.zeros(37, 6)
        probabilities = model.forecast_probabilities(history, calendar, steps=36)
        torch.testing.assert_close(probabilities[:, 0, DOWN], torch.full((36,), .15))
        torch.manual_seed(42)
        paths = model.forecast(history, calendar, steps=36, samples=10000)
        mass = (paths == DOWN).float().mean(0)[:, 0]
        self.assertTrue(((mass - .15).abs() < .015).all())
        # Flat marginals produce stable rank paths, not independent tick noise.
        self.assertTrue((paths[:, 1:] == paths[:, :-1]).all())
        rows = summarize_probabilities(probabilities, RIDES, list(range(36)), history[-1])
        self.assertTrue(all(s["rides"][0]["wait_minutes"]["p50"] == 5 for s in rows))

    def test_only_an_observed_outage_gets_a_residual_survival_floor(self):
        model = constant_model()
        logits = distribution(.15).log().view(1, 1, 1, VOCAB_SIZE).expand(2, 3, 1, -1)
        ages = torch.tensor([[0], [1]])
        probabilities = model.direct_probabilities(logits, ages, [[1, 2, 12]])
        torch.testing.assert_close(probabilities[0, :, 0, DOWN], torch.full((3,), .15))
        torch.testing.assert_close(probabilities[1, :, 0, DOWN], torch.tensor([.9, .81, .9 ** 12]))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 3, 1))
        # Batched one-lead queries (evaluation) agree with all-lead production.
        for index, lead in enumerate((1, 2, 12)):
            separate = model.direct_probabilities(logits[:, :1], ages, [[lead]])
            torch.testing.assert_close(separate[:, 0], probabilities[:, index])

    def test_evaluator_and_map_agree_with_observed_outages_and_saved_hazards(self):
        from ml.data import time_features
        from ml.test import forecast_day
        model = constant_model()
        timestamps = [1793548800 + 300 * i for i in range(10)]
        day = {"rides": RIDES, "timestamps": timestamps,
               "tokens": [[1], [DOWN], [DOWN], [1], [1], [1], [1], [1], [1], [1]]}
        def indexed(batch):
            return {(minutes, int(origin)): prediction.tolist()
                    for minutes, origins, predictions, _ in forecast_day(model, day, RIDES, torch.device("cpu"), batch)
                    for origin, prediction in zip(origins, predictions)}
        expected = indexed(2)
        self.assertEqual(expected, indexed(20))
        for (minutes, origin), prediction in expected.items():
            history = torch.tensor(day["tokens"][:origin + 1])
            future = timestamps[origin + 1:origin + 1 + minutes // 5]
            calendar = torch.tensor(time_features(timestamps[1:origin + 1] + future))
            probabilities = model.forecast_probabilities(history, calendar, steps=len(future))
            previous = history[-1]
            for values in probabilities:
                previous = decode_availability(values, previous)
            self.assertEqual(prediction, previous.tolist())

    def test_observed_survival_uses_the_correct_age_and_keeps_closed_mass_separate(self):
        model = constant_model()
        model.breakdown_hazards = [[0., .25, .5, 1.]]
        logits = distribution(0., .2).log().view(1, 1, 1, VOCAB_SIZE).expand(4, 2, 1, -1)
        ages = torch.tensor([[1], [2], [3], [4]])
        probabilities = model.direct_probabilities(logits, ages, [1, 2])
        expected = torch.tensor([[1., .75], [.75, .375], [.5, 0.], [0., 0.]]) * .8
        torch.testing.assert_close(probabilities[:, :, 0, DOWN], expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(probabilities[..., CLOSED], torch.full((4, 2, 1), .2))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(4, 2, 1))

    def test_hysteresis_prevents_ambiguous_five_minute_flips_but_allows_short_outages(self):
        previous = torch.tensor([1])
        predictions = []
        for risk in (.49, .51, .49, .9, .51, .49, .1, .9, .1):
            previous = decode_availability(distribution(risk), previous)
            predictions.append(previous.item())
        self.assertEqual(predictions, [1, 1, 1, DOWN, DOWN, DOWN, 1, DOWN, 1])

    def test_operating_mass_competes_as_one_class_and_waits_use_conditional_median(self):
        probabilities = distribution(.2)
        probabilities[:, 1] = 0
        probabilities[:, :10] = .08  # Every individual wait bucket loses to DOWN.
        token = decode_availability(probabilities, torch.tensor([1]))
        self.assertEqual(token.item(), 4)
        probabilities = distribution(.5, .25)
        self.assertEqual(decode_availability(probabilities, torch.tensor([1])).item(), DOWN)

    def test_unavailable_subtypes_need_evidence_to_switch_and_do_not_show_waits(self):
        values = torch.stack([distribution(.7, .1), distribution(.39, .41), distribution(.1, .8)])
        snapshots = summarize_probabilities(values, RIDES, [0, 300, 600], torch.tensor([DOWN]))
        self.assertEqual([s["rides"][0]["status"] for s in snapshots], ["DOWN", "DOWN", "CLOSED"])
        self.assertTrue(all(s["rides"][0]["wait_minutes"] is None for s in snapshots))
        self.assertAlmostEqual(snapshots[1]["rides"][0]["closed_probability"], .41, places=6)

    def test_zero_overflow_and_locked_rides_keep_their_meaning(self):
        probabilities = torch.zeros(1, 4, VOCAB_SIZE)
        for index, token in enumerate((0, OVERFLOW, REFURBISHMENT, UNKNOWN)):
            probabilities[0, index, token] = 1
        quantiles = wait_quantiles(probabilities)
        self.assertEqual(quantiles[0, :2].tolist(), [[0] * 3, [OVERFLOW] * 3])
        rows = summarize_probabilities(probabilities, RIDES * 4, [0], torch.tensor([0, OVERFLOW, 1, 1]))[0]["rides"]
        self.assertEqual(rows[0]["wait_minutes"]["p50"], 0)
        self.assertEqual(rows[1]["wait_minutes"]["p50"], "300+")
        self.assertEqual([r["status"] for r in rows[2:]], ["REFURBISHMENT", "UNKNOWN"])
        model = constant_model(3)
        history = torch.tensor([[1, REFURBISHMENT, UNKNOWN]] * 2)
        paths = model.forecast(history, torch.zeros(4, 6), steps=3, samples=20)
        self.assertTrue((paths[..., 1] == REFURBISHMENT).all())
        self.assertTrue((paths[..., 2] == UNKNOWN).all())
        # A uniform draw can be exactly zero; zero-mass operating buckets must
        # never unlock refurbishment or an entirely unobserved attraction.
        with patch("torch.rand", return_value=torch.zeros(3, 20)), \
                patch("torch.rand_like", return_value=torch.zeros(3, 20)):
            paths = model.forecast(history, torch.zeros(4, 6), steps=3, samples=20)
        self.assertTrue((paths[..., 1] == REFURBISHMENT).all())
        self.assertTrue((paths[..., 2] == UNKNOWN).all())

    def test_map_prediction_is_sample_independent_and_target_time_matches_full_timeline(self):
        from ml.parks import get_park
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = constant_model()
            settings = SimpleNamespace(DATASET=root / "data.gz", CHECKPOINT=root / "model.pt", DEVICE="cpu",
                                       SEED=42, SAMPLES=1, TEMPERATURE=1., HISTORY=None,
                                       PREDICT_DAY="2026-11-01", PREDICT_START_TIME="08:10", STEPS=6)
            timestamps = [1793548800 + i * 300 for i in range(8)]  # 08:00 PST after DST ends.
            save_dataset(settings.DATASET, {"version": 1, "park_id": get_park("disneyland").id,
                         "days": {settings.PREDICT_DAY: {"rides": RIDES, "tokens": [[1]] * 8,
                                  "timestamps": timestamps, "expected_timestamps": timestamps}}})
            # Reconstruct through a real checkpoint with the existing v3 layout.
            model_config = dict(num_rides=1, embedding_size=4, hidden_size=8, architecture="ride_horizon_v3",
                                attention_heads=2, dropout=0, predict_closed=True, breakdown_hazards=[[.1] * 48])
            torch.save({"version": 1, "park_id": get_park("disneyland").id, "rides": RIDES,
                        "token_labels": TOKEN_LABELS, "model_config": model_config,
                        "model_state": model.state_dict()}, settings.CHECKPOINT)
            def forecast_probabilities(*args, steps, **kwargs):
                # The requested 08:25 tick is ambiguous. Only decoding the
                # earlier strong DOWN forecast establishes its correct label.
                risks = (.9, .51, .49, .51, .1, .49)
                return torch.stack([distribution(risk) for risk in risks[:steps]])
            with patch.object(type(model), "forecast", side_effect=AssertionError("Display must not sample")), \
                    patch.object(type(model), "forecast_probabilities", side_effect=forecast_probabilities):
                first = predict(settings, print_output=False)
                settings.SAMPLES, settings.SEED = 1000, 901
                second = predict(settings, print_output=False)
                selected = predict(settings, target_time="08:25", print_output=False)
            self.assertEqual(first["forecast_method"], "direct_probabilities")
            self.assertEqual(first["snapshots"], second["snapshots"])
            self.assertEqual(selected["snapshots"], first["snapshots"][3:4])
            self.assertEqual(selected["snapshots"][0]["rides"][0]["status"], "DOWN")


if __name__ == "__main__":
    unittest.main()
