"""Next closure/reopening timing without changing the existing horizon report."""

import copy
import io
import unittest
from contextlib import redirect_stdout

import torch

from ml.data import CLOSED, DOWN, MISSING, OVERFLOW, REFURBISHMENT, UNKNOWN, VOCAB_SIZE
from ml.model import WaitPredictor
from ml.test import empty_transitions, forecast_day, print_transitions, score_transitions


class TransitionPredictor(WaitPredictor):
    """Predict a one-tick breakdown, then reopen and remain operating."""

    def __init__(self):
        super().__init__(1, embedding_size=2, hidden_size=4)

    def forward(self, tokens, next_time_features, hidden=None):
        next_token = torch.where(tokens == 1, DOWN, 2)
        logits = torch.full((*tokens.shape, VOCAB_SIZE), -torch.inf, device=tokens.device)
        logits.scatter_(-1, next_token.unsqueeze(-1), 0.)
        return logits, torch.zeros(1, len(tokens), 1, device=tokens.device)


class ClosureTimingTests(unittest.TestCase):
    def score(self, origin, predicted, actual, observed=None):
        return score_transitions(torch.tensor([[origin]]), torch.tensor(predicted).view(1, -1, 1),
                                 torch.tensor(actual).view(1, -1, 1),
                                 torch.tensor(observed).view(1, -1, 1) if observed is not None else None)

    def test_early_late_and_exact_next_transitions(self):
        closing = self.score(OVERFLOW, [DOWN, 2, DOWN], [2, CLOSED, DOWN])["closing"]
        self.assertEqual(closing["errors"], [-5.])
        self.assertEqual(closing["matched_events"], 1)
        self.assertEqual(closing["actual_events"], 1)
        reopening = self.score(CLOSED, [DOWN, DOWN, 0], [DOWN, 2, 3])["reopening"]
        self.assertEqual(reopening["errors"], [5.])
        self.assertEqual(self.score(DOWN, [DOWN, 0], [CLOSED, OVERFLOW])["reopening"]["errors"], [0.])

    def test_misses_false_alarms_and_no_events_have_no_invented_timing_errors(self):
        missed = self.score(1, [1, 1, 1], [1, CLOSED, CLOSED])["closing"]
        self.assertEqual(missed["missed_events"], 1)
        self.assertEqual(missed["errors"], [])
        false = self.score(DOWN, [0, 0, 0], [DOWN, CLOSED, CLOSED])["reopening"]
        self.assertEqual(false["false_alarms"], 1)
        self.assertEqual(false["actual_events"], 0)
        self.assertEqual(false["errors"], [])
        self.assertEqual(self.score(1, [2, 3], [0, OVERFLOW]), empty_transitions())

    def test_invalid_or_interpolated_actuals_end_timing_coverage(self):
        for invalid in (UNKNOWN, REFURBISHMENT, MISSING):
            self.assertEqual(self.score(1, [1, 1, DOWN], [1, invalid, DOWN]), empty_transitions())
            self.assertEqual(self.score(invalid, [DOWN, 1], [DOWN, 1]), empty_transitions())
        self.assertEqual(self.score(1, [1, DOWN, DOWN], [1, DOWN, DOWN], [True, False, True]),
                         empty_transitions())
        # A known event before the coverage boundary remains eligible.
        self.assertEqual(self.score(1, [DOWN, 1], [DOWN, UNKNOWN])["closing"]["errors"], [0.])

    def test_rollout_accumulates_all_ticks_without_changing_horizon_predictions(self):
        ticks = [1793548800 + 300 * i for i in range(6)]
        source = {"rides": [{"id": "ride", "name": "Ride"}], "timestamps": ticks,
                  "tokens": [[value] for value in (1, 1, DOWN, DOWN, 2, 2)]}
        model = TransitionPredictor()
        expected = forecast_day(model, source, source["rides"], torch.device("cpu"), 1)
        for batch_size in (1, 3, 20):
            totals = empty_transitions()
            actual = forecast_day(model, source, source["rides"], torch.device("cpu"), batch_size,
                                  transition_totals=totals)
            for name in ("closing", "reopening"):
                self.assertEqual(totals[name]["errors"], [-5., 0.])
                self.assertEqual(totals[name]["actual_events"], 2)
                self.assertEqual(totals[name]["matched_events"], 2)
            # Compare by horizon/origin; batching only changes row grouping.
            def rows(results):
                return {(m, o): (p.tolist(), t.tolist()) for m, origins, ps, ts in results
                        for o, p, t in zip(origins.tolist(), ps, ts)}
            self.assertEqual(rows(actual), rows(expected))
        # Later actuals affect scoring, but never the earlier generated trajectory.
        changed = copy.deepcopy(source)
        changed["tokens"][1:] = [[1]] * 5
        totals = empty_transitions()
        forecast_day(model, changed, source["rides"], torch.device("cpu"), 2, transition_totals=totals)
        self.assertEqual(totals["closing"]["actual_events"], 0)
        self.assertEqual(totals["closing"]["false_alarms"], 5)

    def test_session_breaks_and_interpolation_cannot_create_transitions(self):
        for indices, interpolated in (((0, 2), {}), ((0, 1), {"1793549100": [1793548800, 1793549400]})):
            source = {"rides": [{"id": "ride", "name": "Ride"}],
                      "timestamps": [1793548800 + index * 300 for index in indices],
                      "tokens": [[1], [DOWN]], "interpolated_ticks": interpolated}
            totals = empty_transitions()
            forecast_day(TransitionPredictor(), source, source["rides"], torch.device("cpu"), 2,
                         transition_totals=totals)
            self.assertEqual(totals, empty_transitions())

    def test_footer_reports_coverage_and_timing_with_explicit_empty_results(self):
        totals = self.score(1, [1, DOWN, 2], [DOWN, DOWN, 2])
        output = io.StringIO()
        with redirect_stdout(output):
            result = print_transitions(totals)
        self.assertIn("Closing: exact 0.0% (0/1) | timing MAE 5.00 min | bias +5.00 min", output.getvalue())
        self.assertIn("Reopening: exact N/A (0/0) | timing MAE N/A | bias N/A", output.getvalue())
        self.assertEqual(result["closing"]["accuracy"], 0.)
        self.assertEqual(result["closing"]["timing_errors"]["count"], 1)
        self.assertNotIn("nan", output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
