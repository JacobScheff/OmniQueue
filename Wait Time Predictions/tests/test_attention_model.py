"""Causality, recurrent branching, and checkpoints for the ride-attention model."""

import io
import unittest

import torch

from ml.data import DOWN, REFURBISHMENT, UNKNOWN, VOCAB_SIZE
from ml.model import WaitPredictor, token_loss
from ml.test import forecast_day


def predictor(**overrides):
    settings = dict(num_rides=3, embedding_size=4, hidden_size=8,
                    architecture="ride_gru_attention_v1", attention_blocks=2,
                    attention_heads=2, dropout=0.1)
    return WaitPredictor(**(settings | overrides))


class AttentionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_full_day_matches_streaming_and_does_not_read_future_ticks(self):
        model = predictor().eval()
        tokens = torch.randint(0, 61, (2, 5, 3))
        calendar = torch.randn(2, 5, 6)
        with torch.no_grad():
            full, final_state = model(tokens, calendar)
            hidden, streamed = None, []
            for tick in range(tokens.size(1)):
                logits, hidden = model(tokens[:, tick:tick + 1], calendar[:, tick:tick + 1], hidden)
                streamed.append(logits)
            torch.testing.assert_close(full, torch.cat(streamed, dim=1))
            torch.testing.assert_close(final_state, hidden)
            changed_tokens, changed_calendar = tokens.clone(), calendar.clone()
            changed_tokens[:, 3:] = DOWN
            changed_calendar[:, 3:] += 10
            changed, _ = model(changed_tokens, changed_calendar)
        torch.testing.assert_close(full[:, :3], changed[:, :3])
        self.assertEqual(full.shape, (2, 5, 3, VOCAB_SIZE))
        self.assertEqual(hidden.shape, (1, 2, 3, 8))

    def test_attention_shares_information_between_rides_without_mixing_batches(self):
        model = predictor().eval()
        tokens = torch.zeros(2, 2, 3, dtype=torch.long)
        calendar = torch.zeros(2, 2, 6)
        with torch.no_grad():
            before, _ = model(tokens, calendar)
            tokens[0, :, 1] = DOWN
            after, _ = model(tokens, calendar)
        self.assertGreater((before[0, :, 0] - after[0, :, 0]).abs().max().item(), 1e-5)
        torch.testing.assert_close(before[1], after[1])

    def test_branching_recurrent_states_preserves_batch_and_ride_identity(self):
        model = predictor().eval()
        tokens = torch.randint(0, 61, (2, 4, 3))
        calendar = torch.randn(2, 4, 6)
        with torch.no_grad():
            _, hidden = model(tokens, calendar)
            batched, _ = model(tokens[:, :1], calendar[:, :1], hidden)
            separate = [model(tokens[i:i + 1, :1], calendar[i:i + 1, :1], hidden[:, i:i + 1])[0]
                        for i in range(2)]
        torch.testing.assert_close(batched, torch.cat(separate, dim=0))

    def test_gradients_reach_shared_gru_ride_embeddings_and_both_attention_blocks(self):
        model = predictor()
        logits, _ = model(torch.randint(0, 61, (2, 4, 3)), torch.randn(2, 4, 6))
        loss, _ = token_loss(logits, torch.randint(0, 61, (2, 4, 3)))
        loss.backward()
        parameters = [model.gru.weight_hh_l0, model.ride_embedding.weight]
        parameters.extend(block.self_attn.in_proj_weight for block in model.attention)
        for parameter in parameters:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_checkpoint_round_trip_and_forecast_status_locks(self):
        settings = dict(num_rides=3, embedding_size=4, hidden_size=8,
                        architecture="ride_gru_attention_v1", attention_blocks=2,
                        attention_heads=2, dropout=0.1)
        model = WaitPredictor(**settings).eval()
        history = torch.tensor([[1, REFURBISHMENT, UNKNOWN], [2, REFURBISHMENT, UNKNOWN]])
        calendar = torch.randn(1, 2, 6)
        checkpoint = io.BytesIO()
        torch.save({"model_config": settings, "model_state": model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored = WaitPredictor(**saved["model_config"]).eval()
        restored.load_state_dict(saved["model_state"])
        with torch.no_grad():
            torch.testing.assert_close(model(history.unsqueeze(0), calendar)[0],
                                       restored(history.unsqueeze(0), calendar)[0])
        paths = restored.forecast(history, torch.randn(4, 6), steps=3, samples=4)
        self.assertEqual(paths.shape, (4, 3, 3))
        self.assertTrue((paths[:, :, 0] <= DOWN).all())
        self.assertTrue((paths[:, :, 1] == REFURBISHMENT).all())
        self.assertTrue((paths[:, :, 2] == UNKNOWN).all())

    def test_held_out_evaluator_accepts_per_ride_states(self):
        model = predictor()
        rides = [{"id": str(i), "name": str(i)} for i in range(3)]
        day = {"rides": rides, "timestamps": [1793548800 + 300 * i for i in range(9)],
               "tokens": [[1, 2, 3]] * 9}
        small = forecast_day(model, day, rides, torch.device("cpu"), 2)
        large = forecast_day(model, day, rides, torch.device("cpu"), 8)
        for horizon in (5, 15, 30):
            rows = []
            for results in (small, large):
                rows.append({origin: (prediction.tolist(), target.tolist())
                             for minutes, origins, predictions, targets in results if minutes == horizon
                             for origin, prediction, target in zip(origins.tolist(), predictions, targets)})
            self.assertTrue(rows[0])
            self.assertEqual(rows[0], rows[1])

    def test_rejects_invalid_attention_settings(self):
        for settings in ({"hidden_size": 7}, {"attention_blocks": 0},
                         {"attention_heads": 0}, {"dropout": -0.1}, {"dropout": 1.0},
                         {"architecture": "unknown"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                predictor(**settings)


if __name__ == "__main__":
    unittest.main()
