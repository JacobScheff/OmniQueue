"""Day isolation, evaluation masking, and saved test-set regression checks."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from ml.data import CLOSED, DOWN, TOKEN_LABELS, load_dataset, save_dataset
from ml.model import WaitPredictor
from ml.test import evaluate
from ml.train import Days, collate_days, epoch, split_dates, train


def day(tokens, rides=None):
    return {"timestamps": [1793548800 + 300 * index for index in range(len(tokens))],
            "tokens": tokens, "rides": rides or [{"id": "common", "name": "Common"}]}


class SplitTests(unittest.TestCase):
    def test_chronological_nonoverlapping_split_and_rounding(self):
        dates = [f"2026-10-{number:02}" for number in range(1, 28)]
        groups = split_dates(reversed(dates), .70, .15, .15)
        self.assertEqual([len(group) for group in groups], [19, 4, 4])
        self.assertEqual(sum((list(group) for group in groups), []), dates)
        self.assertEqual(groups, split_dates(dates, .70, .15, .15))
        self.assertEqual(split_dates(dates[:3], .70, .15, .15), (dates[:1], dates[1:2], dates[2:3]))
        self.assertEqual(split_dates(dates[:1], 1., 0., 0.), (dates[:1], [], []))

    def test_rejects_invalid_fractions_and_too_few_days(self):
        for fractions in ((.7, .2, .2), (0., .5, .5), (-.1, .5, .6),
                          (float("nan"), .15, .15), (float("inf"), .1, .1)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                split_dates(["a", "b", "c"], *fractions)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            split_dates(["a", "b"], .7, .15, .15)
        with self.assertRaisesRegex(ValueError, "unique"):
            split_dates(["a", "a", "b"], .7, .15, .15)

    def test_evaluation_weights_valid_targets_and_never_updates_model(self):
        model = WaitPredictor(1, embedding_size=2, hidden_size=4)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[1] = 2
        before = {key: value.clone() for key, value in model.state_dict().items()}
        days = Days([day([[1], [1], [DOWN], [CLOSED]]), day([[1], [1], [1]])],
                    [{"id": "common", "name": "Common"}])
        scores = []

        def check_evaluation(module, args):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(module.training)

        handle = model.register_forward_pre_hook(check_evaluation)
        try:
            for batch_size in (1, 2):
                model.train()
                scores.append(epoch(model, DataLoader(days, batch_size=batch_size, collate_fn=collate_days),
                                    torch.device("cpu")))
        finally:
            handle.remove()
        self.assertAlmostEqual(scores[0][0], scores[1][0], places=6)
        self.assertEqual(scores[0][1], .75)
        self.assertEqual(scores[1][1], .75)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(before[key], value)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_training_validation_and_test_keep_saved_days_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = SimpleNamespace(
                DATASET=root / "data.json.gz", CHECKPOINT=root / "model.pt", DEVICE="cpu", SEED=42,
                EPOCHS=2, BATCH_SIZE=3, HIDDEN_SIZE=8, EMBEDDING_SIZE=2, LEARNING_RATE=.001,
                TRAIN_SPLIT=.7, VAL_SPLIT=.15, TEST_SPLIT=.15,
            )
            days = {}
            for index in range(12):
                rides = [{"id": "common", "name": "Common"}, {"id": f"only-{index}", "name": str(index)}]
                days[f"2026-11-{index + 1:02}"] = day([[index + 1, 1]] * 4, rides)
            # Incomplete and unusably short days cannot be assigned to a split.
            days["2026-11-13"] = day([[1]])
            save_dataset(settings.DATASET, {"version": 1, "park_id": "park", "days": days,
                                           "incomplete_days": {"2026-11-14": day([[1], [1]])}})
            output = io.StringIO()
            with patch("ml.train.epoch", wraps=epoch) as calls, redirect_stdout(output):
                train(settings)
            self.assertEqual(calls.call_count, 4)
            self.assertEqual([len(call.args) for call in calls.call_args_list], [4, 3, 4, 3])
            checkpoint = torch.load(settings.CHECKPOINT, weights_only=True)
            expected = split_dates(sorted(days)[:12], .7, .15, .15)
            for field, dates in zip(("train_dates", "validation_dates", "test_dates"), expected):
                self.assertEqual(checkpoint[field], dates)
            self.assertEqual(len(calls.call_args_list[0].args[1].dataset), len(expected[0]))
            self.assertEqual(len(calls.call_args_list[1].args[1].dataset), len(expected[1]))
            # A ride found only in the holdouts cannot change the model's inputs.
            self.assertEqual({ride["id"] for ride in checkpoint["rides"]},
                             {"common"} | {f"only-{i}" for i in range(len(expected[0]))})
            self.assertIn("val NLL", output.getvalue())
            self.assertIsInstance(checkpoint["validation_loss"], float)
            self.assertIsInstance(checkpoint["validation_accuracy"], float)
            self.assertEqual(checkpoint["epoch"], 2)
            original_checkpoint = settings.CHECKPOINT.read_bytes()
            # New data and changed split config must not move the saved test set.
            dataset = load_dataset(settings.DATASET)
            dataset["days"]["2026-11-15"] = day([[1], [1]])
            save_dataset(settings.DATASET, dataset)
            settings.TRAIN_SPLIT, settings.VAL_SPLIT, settings.TEST_SPLIT = 1., 0., 0.
            with redirect_stdout(output):
                result = evaluate(settings)
            self.assertEqual(result["dates"], expected[2])
            self.assertTrue(0 <= result["horizons"][5]["accuracy"] <= 1)
            self.assertIn("Autoregressive greedy forecasts", output.getvalue())
            self.assertEqual(settings.CHECKPOINT.read_bytes(), original_checkpoint)
            # Reject old all-days checkpoints and corrupted overlapping splits.
            for change, message in (({"test_dates": checkpoint["train_dates"][:1]}, "overlap"),
                                    ({"test_dates": []}, "no held-out")):
                torch.save({**checkpoint, **change}, settings.CHECKPOINT)
                with self.assertRaisesRegex(ValueError, message):
                    evaluate(settings)
            old = dict(checkpoint)
            del old["test_dates"]
            torch.save(old, settings.CHECKPOINT)
            with self.assertRaisesRegex(ValueError, "Retrain"):
                evaluate(settings)
            torch.save(checkpoint, settings.CHECKPOINT)
            del dataset["days"][expected[2][0]]
            save_dataset(settings.DATASET, dataset)
            with self.assertRaisesRegex(ValueError, "missing from"):
                evaluate(settings)


if __name__ == "__main__":
    unittest.main()
