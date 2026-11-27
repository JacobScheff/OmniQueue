"""Batched days/leads must reproduce independent, causal held-out forecasts."""

import copy
import unittest
from unittest.mock import patch

import torch

from ml.availability import decode_availability, decode_availability_sequence
from ml.data import CLOSED, DOWN, REFURBISHMENT, UNKNOWN, VOCAB_SIZE
from ml.model import build_model
from ml.test import empty_transitions, forecast_day, forecast_days


RIDES = [{"id": str(index), "name": str(index)} for index in range(3)]


def sources():
    first = 1793548800
    def day(values, offsets, timezone):
        return {"rides": RIDES, "tokens": values, "timestamps": [first + 300 * i for i in offsets],
                "timezone": timezone}
    long = day([[1 + i % 3, DOWN if i < 4 else 3, UNKNOWN if i < 5 else 4]
                for i in range(126)], range(126), "America/Los_Angeles")
    short = day([[REFURBISHMENT if i < 3 else 4, CLOSED if i < 4 else 2, 0]
                 for i in range(10)], (0, 1, 2, 3, 10, 11, 12, 13, 14, 15), "America/New_York")
    repaired = day([[DOWN if i < 2 else 5, 2, 0] for i in range(12)], range(12), "America/Los_Angeles")
    repaired["interpolated_ticks"] = {str(first + 300): [first, first + 900]}
    return [long, short, repaired]


def indexed(results):
    return {(minutes, int(origin)): (prediction.tolist(), target.tolist())
            for minutes, origins, predictions, targets in results
            for origin, prediction, target in zip(origins, predictions, targets)}


class ParallelEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.model = build_model(num_rides=3, embedding_size=4, hidden_size=8,
                                 architecture="ride_horizon_v3", attention_heads=2, dropout=.1,
                                 predict_closed=True, breakdown_hazards=[[.1] * 48, [.2] * 48, [.3] * 48]).eval()
        self.days = sources()

    def test_parallel_days_and_leads_match_serial_horizons_targets_and_transition_metrics(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        parallel = forecast_days(self.model, self.days, RIDES, torch.device("cpu"), 32, horizon_batch_size=8)
        for day, (forecasts, transitions) in zip(self.days, parallel):
            expected_transitions = empty_transitions()
            expected = forecast_day(self.model, day, RIDES, torch.device("cpu"), 32,
                                    transition_totals=expected_transitions)
            self.assertEqual(indexed(forecasts), indexed(expected))
            for name in transitions:
                transitions[name]["errors"].sort()
                expected_transitions[name]["errors"].sort()
            self.assertEqual(transitions, expected_transitions)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_chunk_sizes_padding_and_other_days_cannot_change_forecasts(self):
        expected = forecast_days(self.model, self.days, RIDES, torch.device("cpu"), 64, horizon_batch_size=16)
        actual = forecast_days(self.model, self.days, RIDES, torch.device("cpu"), 11, horizon_batch_size=3)
        self.assertEqual([indexed(item[0]) for item in expected], [indexed(item[0]) for item in actual])
        # A changed suffix can change later encodings/targets, but cannot enter
        # an earlier prefix, nor affect another day's model state.
        changed = copy.deepcopy(self.days)
        changed[0]["tokens"][3:] = [[40, CLOSED, DOWN]] * (len(changed[0]["tokens"]) - 3)
        modified = forecast_days(self.model, changed, RIDES, torch.device("cpu"), 64, horizon_batch_size=8)
        for day_index, ((forecasts, _), (other, _)) in enumerate(zip(expected, modified)):
            original, new = indexed(forecasts), indexed(other)
            for key in original:
                if day_index or key[1] < 3:
                    self.assertEqual(original[key][0], new[key][0])

    def test_history_is_encoded_once_and_independent_leads_share_queries(self):
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward, \
                patch.object(self.model, "horizon_logits", wraps=self.model.horizon_logits) as query:
            forecast_days(self.model, self.days, RIDES, torch.device("cpu"), 64, horizon_batch_size=8)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(forward.call_args.args[0].shape[0], 3)
        self.assertTrue(any(call.args[2].shape[1] == 8 for call in query.call_args_list))

    def test_sequence_decoder_keeps_hysteresis_across_lead_chunk_boundaries(self):
        probabilities = torch.zeros(2, 9, 3, VOCAB_SIZE)
        for step, risk in enumerate((.49, .9, .51, .49, .1, .9, .8, .51, .1)):
            probabilities[:, step, :, 1] = 1 - risk
            probabilities[:, step, :, DOWN] = risk
        initial = torch.tensor([[1, DOWN, CLOSED], [DOWN, 1, 1]])
        previous = initial
        expected = []
        for values in probabilities.unbind(1):
            previous = decode_availability(values, previous)
            expected.append(previous)
        expected = torch.stack(expected, dim=1)
        torch.testing.assert_close(decode_availability_sequence(probabilities, initial), expected)
        first = decode_availability_sequence(probabilities[:, :3], initial)
        last = decode_availability_sequence(probabilities[:, 3:], first[:, -1])
        torch.testing.assert_close(torch.cat((first, last), dim=1), expected)

    def test_legacy_models_keep_their_autoregressive_policy(self):
        model = build_model(num_rides=3, embedding_size=4, hidden_size=8).eval()
        days = self.days[1:]
        parallel = forecast_days(model, days, RIDES, torch.device("cpu"), 16)
        for day, (forecasts, transitions) in zip(days, parallel):
            expected_transitions = empty_transitions()
            expected = forecast_day(model, day, RIDES, torch.device("cpu"), 16,
                                    transition_totals=expected_transitions)
            self.assertEqual(indexed(forecasts), indexed(expected))
            self.assertEqual(transitions, expected_transitions)

    def test_invalid_batch_sizes_fail_before_evaluation(self):
        for batch, leads in ((0, 8), (1, 0)):
            with self.assertRaisesRegex(ValueError, "positive"):
                forecast_days(self.model, self.days, RIDES, torch.device("cpu"), batch, horizon_batch_size=leads)


if __name__ == "__main__":
    unittest.main()
