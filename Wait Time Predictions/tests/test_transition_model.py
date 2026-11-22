"""Causality, ordered errors, and generated-input training for v2."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ml.data import CLOSED, DOWN, OVERFLOW, REFURBISHMENT, UNKNOWN, VOCAB_SIZE, save_dataset, time_features
from ml.model import build_model, token_loss
from ml.run import summarize_probabilities
from ml.test import forecast_day
from ml.train import Days, horizon_loss, train, rollout_loss


def predictor():
    return build_model(num_rides=3, embedding_size=4, hidden_size=8,
                       architecture="ride_transition_v2", attention_blocks=2, attention_heads=2)


class TransitionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_full_streaming_and_branched_states_agree_without_future_leakage(self):
        model = predictor().eval()
        tokens = torch.tensor([[[2, DOWN, UNKNOWN], [2, DOWN, 4], [3, DOWN, 4],
                                [3, 1, 4], [3, 1, CLOSED]]] * 2)
        calendar = torch.randn(2, 5, 6)
        with torch.no_grad():
            full, states = model(tokens, calendar, return_state_sequence=True)
            hidden, streamed = None, []
            for tick in range(5):
                logits, hidden = model(tokens[:, tick:tick + 1], calendar[:, tick:tick + 1], hidden)
                streamed.append(logits)
            torch.testing.assert_close(full, torch.cat(streamed, dim=1))
            torch.testing.assert_close(states[:, :, -1], hidden)
            # A saved origin must reproduce the rest of an observed sequence.
            suffix, suffix_state = model(tokens[:, 3:], calendar[:, 3:], states[:, :, 2])
            torch.testing.assert_close(full[:, 3:], suffix)
            torch.testing.assert_close(hidden, suffix_state)
            changed = tokens.clone()
            changed[:, 3:] = 50
            modified, _ = model(changed, calendar + torch.tensor([0., 0., 0., 10., 10.])[None, :, None])
            torch.testing.assert_close(full[:, :3], modified[:, :3])
        self.assertEqual(states.shape, (1, 2, 5, 3, 10))
        self.assertEqual(hidden[0, 0, :, -1].tolist(), [3., 2., 1.])
        self.assertTrue(torch.isfinite(full[..., :CLOSED + 1]).all())
        torch.testing.assert_close(full.exp().sum(-1), torch.ones_like(tokens, dtype=torch.float32))

    def test_park_conditions_enter_other_rides_recurrent_memory_without_mixing_days(self):
        model = predictor().eval()
        tokens = torch.ones(2, 3, 3, dtype=torch.long)
        calendar = torch.zeros(2, 3, 6)
        with torch.no_grad():
            _, before = model(tokens, calendar)
            tokens[0, :2, 1] = DOWN
            _, after = model(tokens, calendar)
        self.assertGreater((before[0, 0, 0, :8] - after[0, 0, 0, :8]).abs().max().item(), 1e-5)
        torch.testing.assert_close(before[:, 1], after[:, 1])

    def test_ordered_loss_distinguishes_near_and_far_errors_without_ordering_statuses(self):
        near = torch.full((1, 1, 1, VOCAB_SIZE), -20.)
        near[..., 5], near[..., 6] = 0., 1.
        far = torch.full_like(near, -20.)
        far[..., 5], far[..., 35] = 0., 1.
        target = torch.tensor([[[5]]])
        self.assertAlmostEqual(token_loss(near, target)[0].item(), token_loss(far, target)[0].item(), places=6)
        self.assertLess(token_loss(near, target, ordinal_weight=2.)[0].item(),
                        token_loss(far, target, ordinal_weight=2.)[0].item())
        for label in (DOWN, CLOSED, OVERFLOW):
            target.fill_(label)
            torch.testing.assert_close(token_loss(near, target, predict_closed=True)[0],
                                       token_loss(near, target, predict_closed=True, ordinal_weight=2.)[0])
        loss, count = token_loss(near, torch.tensor([[[-1]]]), ordinal_weight=2.)
        self.assertEqual(count.item(), 0)
        self.assertEqual(loss.item(), 0.)

    def test_rollout_uses_generated_inputs_and_backpropagates_through_recurrence(self):
        model = predictor().eval()
        tokens = torch.ones(2, 6, 3, dtype=torch.long)
        features = torch.randn(2, 6, 6)
        targets = torch.full_like(tokens, 40)
        logits, states = model(tokens, features, return_state_sequence=True)
        consumed = []
        handle = model.register_forward_pre_hook(lambda module, args: consumed.append(args[0].clone()))
        try:
            loss = rollout_loss(model, tokens, features, targets, logits, states,
                                steps=4, origins=2, ordinal_weight=2.)
        finally:
            handle.remove()
        loss.backward()
        self.assertEqual(len(consumed), 3)
        self.assertTrue(all((inputs != 40).all() for inputs in consumed))
        for parameter in (model.gru.weight_hh_l0, model.park_context[0].weight,
                          model.delta_output.weight, model.local_gate.weight,
                          model.availability_output.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_rollouts_cannot_cross_session_breaks_or_padding(self):
        model = predictor().eval()
        tokens = torch.ones(2, 5, 3, dtype=torch.long)
        features = torch.zeros(2, 5, 6)
        logits, states = model(tokens, features, return_state_sequence=True)
        targets = tokens.clone()
        targets[0, 2] = UNKNOWN  # scheduled break
        targets[1, 2:] = UNKNOWN  # shorter padded day
        with patch.object(model, "forward", wraps=model.forward) as forward:
            loss = rollout_loss(model, tokens, features, targets, logits, states, steps=3, origins=2)
        self.assertEqual(loss.item(), 0.)
        forward.assert_not_called()

    def test_interpolation_cannot_leak_a_future_endpoint_into_training_prefixes(self):
        rides = [{"id": "ride", "name": "Ride"}]
        source = {"rides": rides, "timestamps": [0, 300, 600, 900],
                  "tokens": [[1], [20], [40], [2]], "interpolated_ticks": {"300": [0, 600]}}
        inputs, _, targets = Days([source], rides)[0]
        self.assertEqual(inputs[:, 0].tolist(), [1, UNKNOWN, 40])
        self.assertEqual(targets[:, 0].tolist(), [UNKNOWN, 40, 2])
        self.assertEqual(source["tokens"][1], [20])

    def test_v2_checkpoint_round_trip_sampling_locks_and_evaluation_batches(self):
        settings = dict(num_rides=3, embedding_size=4, hidden_size=8,
                        architecture="ride_transition_v2", attention_heads=2)
        model = build_model(**settings).eval()
        checkpoint = io.BytesIO()
        torch.save({"model_config": settings, "model_state": model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored = build_model(**saved["model_config"])
        restored.load_state_dict(saved["model_state"])
        history = torch.tensor([[1, REFURBISHMENT, UNKNOWN]] * 4)
        paths = restored.forecast(history, torch.zeros(7, 6), steps=4, samples=4)
        self.assertTrue((paths[..., 1] == REFURBISHMENT).all())
        self.assertTrue((paths[..., 2] == UNKNOWN).all())
        self.assertTrue((paths[..., 0] <= DOWN).all())
        rides = [{"id": str(i), "name": str(i)} for i in range(3)]
        source = {"rides": rides, "timestamps": [1793548800 + 300 * i for i in range(9)],
                  "tokens": [[1, 2, 3]] * 9}
        def indexed(batch):
            return {(minutes, int(origin)): prediction.tolist()
                    for minutes, origins, predictions, _ in forecast_day(restored, source, rides, torch.device("cpu"), batch)
                    for origin, prediction in zip(origins, predictions)}
        self.assertEqual(indexed(2), indexed(20))

    def test_validation_selects_best_checkpoint_and_keeps_latest_for_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = SimpleNamespace(DATASET=root / "days.gz", CHECKPOINT=root / "model.pt",
                                       DEVICE="cpu", SEED=42, EPOCHS=5, BATCH_SIZE=2,
                                       EMBEDDING_SIZE=4, HIDDEN_SIZE=8, LEARNING_RATE=.001,
                                       TRAIN_SPLIT=.7, VAL_SPLIT=.15, TEST_SPLIT=.15,
                                       EARLY_STOPPING_PATIENCE=2)
            source = {"rides": [{"id": "ride", "name": "Ride"}],
                      "timestamps": [0, 300, 600], "tokens": [[1], [1], [1]]}
            save_dataset(settings.DATASET, {"version": 1, "park_id": "7340550b-c14d-4def-80bb-acdb51d49a66",
                                           "days": {str(i): source for i in range(5)}})
            validations = iter([1., .8, 1.2, 1.3])
            def fake_epoch(model, loader, device, optimizer=None, *, metrics, **kwargs):
                metrics.update(nll=.5, rollout_loss=.5)
                return (.5 if optimizer else next(validations)), .9
            with patch("ml.train.epoch", side_effect=fake_epoch), redirect_stdout(io.StringIO()):
                train(settings)
            selected = torch.load(settings.CHECKPOINT, weights_only=True)
            latest = torch.load(root / "model.latest.pt", weights_only=True)
            self.assertEqual(selected["epoch"], 2)
            self.assertEqual(latest["epoch"], 4)
            self.assertEqual(latest["best_epoch"], 2)
            self.assertEqual(selected["selection"], "validation_forecast_objective")

    def test_horizon_training_cannot_read_future_tokens_and_trains_the_query(self):
        model = build_model(num_rides=3, embedding_size=4, hidden_size=8,
                            architecture="ride_horizon_v3", attention_heads=2).eval()
        tokens = torch.ones(2, 6, 3, dtype=torch.long)
        features = torch.randn(2, 6, 6)
        targets = torch.full_like(tokens, 2)
        _, states = model(tokens, features, return_state_sequence=True)
        loss = horizon_loss(model, tokens, features, targets, states, horizons=(6,), origins=2)
        changed = tokens.clone()
        changed[:, 1:] = DOWN
        _, changed_states = model(changed, features, return_state_sequence=True)
        other = horizon_loss(model, changed, features, targets, changed_states, horizons=(6,), origins=2)
        torch.testing.assert_close(loss, other)
        loss.backward()
        for parameter in (model.horizon_query[0].weight, model.gru.weight_ih_l0):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_direct_production_and_evaluator_match_and_do_not_read_actual_future(self):
        model = build_model(num_rides=3, embedding_size=4, hidden_size=8,
                            architecture="ride_horizon_v3", attention_heads=2).eval()
        rides = [{"id": str(i), "name": str(i)} for i in range(3)]
        timestamps = [1793548800 + 300 * i for i in range(10)]
        source = {"rides": rides, "timestamps": timestamps, "tokens": [[1, 2, 3]] * 10}
        def indexed(batch, day=source):
            return {(minutes, int(origin)): prediction.tolist()
                    for minutes, origins, predictions, _ in forecast_day(model, day, rides, torch.device("cpu"), batch)
                    for origin, prediction in zip(origins, predictions)}
        expected = indexed(2)
        self.assertEqual(expected, indexed(20))
        changed = {**source, "tokens": source["tokens"][:3] + [[40, DOWN, CLOSED]] * 7}
        actual = indexed(20, changed)
        for minutes in (5, 15, 30):
            self.assertEqual(expected[minutes, 2], actual[minutes, 2])
            history = torch.tensor(source["tokens"][:3])
            calendar = torch.tensor(time_features(timestamps[1:3 + minutes // 5]))
            probabilities = model.forecast_probabilities(history, calendar, steps=minutes // 5)
            from ml.availability import decode_availability
            previous = history[-1]
            for values in probabilities:
                previous = decode_availability(values, previous)
            self.assertEqual(expected[minutes, 2], previous.tolist())
            summary = summarize_probabilities(probabilities, rides, timestamps[3:3 + minutes // 5], history[-1])
            for index, row in enumerate(summary[-1]["rides"]):
                self.assertEqual(row.get("status", "OPERATING"),
                                 {DOWN: "DOWN", CLOSED: "CLOSED"}.get(previous[index].item(), "OPERATING"))

    def test_direct_forecasts_preserve_the_remaining_duration_of_an_observed_outage(self):
        model = build_model(num_rides=1, embedding_size=4, hidden_size=8,
                            architecture="ride_horizon_v3", attention_heads=2,
                            predict_closed=True, breakdown_hazards=[[0., 0., 0., 1.]])
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-40)
            model.output.bias[2] = 40
            model.local_gate.weight.zero_()
            model.local_gate.bias.fill_(-40)
            model.availability_output.weight.zero_()
            model.availability_output.bias.copy_(torch.tensor([40., -40., -40.]))
        paths = model.forecast(torch.tensor([[2], [DOWN]]), torch.zeros(6, 6), steps=5, samples=4)
        self.assertEqual(paths[:, :, 0].tolist(), [[DOWN, DOWN, DOWN, 2, 2]] * 4)

    def test_direct_supervision_cannot_cross_session_breaks_or_padding(self):
        model = build_model(num_rides=3, embedding_size=4, hidden_size=8,
                            architecture="ride_horizon_v3", attention_heads=2).eval()
        tokens = torch.ones(2, 5, 3, dtype=torch.long)
        features = torch.zeros(2, 5, 6)
        _, states = model(tokens, features, return_state_sequence=True)
        targets = tokens.clone()
        targets[0, 2] = UNKNOWN
        targets[1, 2:] = UNKNOWN
        with patch.object(model, "horizon_logits", wraps=model.horizon_logits) as query:
            loss = horizon_loss(model, tokens, features, targets, states, horizons=(3, 4, 6), origins=2)
        self.assertEqual(loss.item(), 0.)
        query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
