"""Verify held-out horizon rollouts never consume future observations."""

import copy
import unittest

import torch

from ml.data import CLOSED, REFURBISHMENT, UNKNOWN, VOCAB_SIZE, time_features
from ml.model import WaitPredictor
from ml.test import HORIZON_MINUTES, forecast_day


class PrefixSumPredictor(WaitPredictor):
    """A deterministic recurrent predictor with an exactly known rollout."""

    def __init__(self):
        super().__init__(1, embedding_size=2, hidden_size=4)

    def forward(self, tokens, next_time_features, hidden=None):
        assert not torch.is_grad_enabled()
        totals = tokens.cumsum(dim=1)
        if hidden is not None:
            totals = totals + hidden[0].unsqueeze(1)
        logits = torch.full((*tokens.shape, VOCAB_SIZE), -torch.inf, device=tokens.device)
        logits.scatter_(-1, (totals % 61).unsqueeze(-1), 0.)
        return logits, totals[:, -1].unsqueeze(0)


def day(values, timestamps=None):
    rides = [{"id": "ride", "name": "Ride"}]
    return {"rides": rides, "tokens": [[value] for value in values],
            "timestamps": timestamps or [1793548800 + 300 * i for i in range(len(values))]}


def indexed(results):
    return {(minutes, origin): (prediction.tolist(), target.tolist())
            for minutes, origins, predictions, targets in results
            for origin, prediction, target in zip(origins.tolist(), predictions, targets)}


class AutoregressiveTests(unittest.TestCase):
    def test_horizons_and_batched_prefix_states_match_independent_forecasts(self):
        model = PrefixSumPredictor()
        source = day([1 + i % 5 for i in range(126)])
        results = indexed(forecast_day(model, source, source["rides"], torch.device("cpu"), 7))
        timestamps = source["timestamps"]
        for minutes in HORIZON_MINUTES:
            steps = minutes // 5
            self.assertEqual(sum(horizon == minutes for horizon, _ in results), len(timestamps) - steps)
        for origin in (0, 2, 5):
            steps = min(max(HORIZON_MINUTES) // 5, len(timestamps) - origin - 1)
            features = torch.tensor(time_features(timestamps[1:origin + 1] +
                                                   timestamps[origin + 1:origin + steps + 1]), dtype=torch.float32)
            # Delta distributions make the production sampler deterministic.
            paths = model.forecast(torch.tensor(source["tokens"][:origin + 1]), features,
                                   steps=steps, samples=1)
            for minutes in HORIZON_MINUTES:
                if minutes // 5 <= steps:
                    self.assertEqual(results[minutes, origin][0], paths[0, minutes // 5 - 1].tolist())
        # Changing batching cannot change a deterministic forecast or its target.
        other = indexed(forecast_day(model, source, source["rides"], torch.device("cpu"), 64))
        self.assertEqual(results, other)

    def test_future_actuals_change_targets_but_not_an_earlier_forecast(self):
        model = PrefixSumPredictor()
        source = day([1] * 124)
        changed = copy.deepcopy(source)
        changed["tokens"][3:] = [[20]] * (len(source["tokens"]) - 3)
        before = indexed(forecast_day(model, source, source["rides"], torch.device("cpu"), 8))
        after = indexed(forecast_day(model, changed, changed["rides"], torch.device("cpu"), 8))
        for minutes in HORIZON_MINUTES:
            self.assertEqual(before[minutes, 2][0], after[minutes, 2][0])
            self.assertNotEqual(before[minutes, 2][1], after[minutes, 2][1])
        # At origin 0 the recurrent sum feeds itself: 1 -> 2 -> 4.
        self.assertEqual(before[15, 0][0], [4])

    def test_session_breaks_and_end_of_day_limit_real_elapsed_horizons(self):
        model = PrefixSumPredictor()
        start = 1793548800
        source = day([1] * 7, [start + 300 * i for i in (0, 1, 2, 8, 9, 10, 11)])
        results = indexed(forecast_day(model, source, source["rides"], torch.device("cpu"), 8))
        self.assertEqual({origin for minutes, origin in results if minutes == 5}, {0, 1, 3, 4, 5})
        self.assertEqual({origin for minutes, origin in results if minutes == 15}, {3})
        self.assertFalse(any(minutes >= 30 for minutes, _ in results))

    def test_origins_cannot_use_an_interpolations_future_endpoint(self):
        source = day([1] * 7)
        source["interpolated_ticks"] = {str(source["timestamps"][1]):
                                         [source["timestamps"][0], source["timestamps"][3]]}
        results = indexed(forecast_day(PrefixSumPredictor(), source, source["rides"], torch.device("cpu"), 8))
        self.assertEqual({origin for minutes, origin in results if minutes == 5}, {0, 3, 4, 5})

    def test_forecast_output_masks_and_refurbishment_absence_locks(self):
        source = day([REFURBISHMENT, UNKNOWN, 1, 1, 1, 1])
        model = WaitPredictor(1, embedding_size=2, hidden_size=4)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[CLOSED] = 100
            model.output.bias[2] = 10
        before = {key: value.clone() for key, value in model.state_dict().items()}
        results = indexed(forecast_day(model, source, source["rides"], torch.device("cpu"), 2))
        self.assertEqual(results[15, 0][0], [REFURBISHMENT])
        self.assertEqual(results[15, 2][0], [2])
        absent = day([UNKNOWN, UNKNOWN, 1, 1, 1])
        results = indexed(forecast_day(model, absent, absent["rides"], torch.device("cpu"), 2))
        self.assertEqual(results[15, 0][0], [UNKNOWN])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(before[key], value)
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
