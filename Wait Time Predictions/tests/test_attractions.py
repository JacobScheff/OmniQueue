"""Excluded attractions never enter model tensors or visualizer snapshots."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from ml.attractions import EXCLUDED_ATTRACTIONS, is_excluded, modeling_day
from ml.data import MISSING, TOKEN_LABELS, save_dataset
from ml.model import WaitPredictor
from ml.run import predict
from ml.test import evaluate
from ml.train import train
from ml.visualize import PARK_ID, Visualizer


class AttractionTests(unittest.TestCase):
    def test_ids_names_and_duplicate_listings_are_excluded(self):
        for ride_id, name in EXCLUDED_ATTRACTIONS.items():
            with self.subTest(name=name):
                self.assertTrue(is_excluded({"id": ride_id, "name": "Renamed attraction"}))
                self.assertTrue(is_excluded({"id": "other-source-id", "name": name}))
        for name in ("Adventureland Treehouse inspired by Walt Disney’s Swiss Family Robinson",
                     "MINNIE’S HOUSE", "Walt Disney – A Magical Life", "Opera House",
                     "Goofy’s How-to-Play Yard", "Donald's Duck Pond"):
            self.assertTrue(is_excluded({"id": "alternate", "name": name}), name)
        for name in ("Mickey & Minnie's Runaway Railway", "Disneyland Railroad",
                     "Storybook Land Canal Boats", "Mark Twain Riverboat"):
            self.assertFalse(is_excluded({"id": "kept", "name": name}), name)

    def test_projection_keeps_columns_aligned_and_does_not_mutate_source(self):
        day = {"rides": [{"id": "a", "name": "A"}, {"id": "x", "name": "Treehouse"},
                         {"id": "b", "name": "B"}],
               "tokens": [[1, MISSING, 2], [3, 60, 4]], "timestamps": [0, 300]}
        before = copy.deepcopy(day)
        result = modeling_day(day)
        self.assertEqual(result["tokens"], [[1, 2], [3, 4]])
        self.assertEqual([ride["id"] for ride in result["rides"]], ["a", "b"])
        self.assertEqual(day, before)


class ExclusionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.settings = SimpleNamespace(
            DATASET=root / "data.json.gz", CHECKPOINT=root / "model.pt", DEVICE="cpu", SEED=42,
            EPOCHS=1, BATCH_SIZE=2, HIDDEN_SIZE=8, EMBEDDING_SIZE=4, LEARNING_RATE=.001,
            TRAIN_SPLIT=.7, VAL_SPLIT=.15, TEST_SPLIT=.15,
            MODEL_ARCHITECTURE="ride_gru_attention_v1", ATTENTION_BLOCKS=1, ATTENTION_HEADS=2,
            DROPOUT=0., HISTORY=None, PREDICT_DAY="2026-11-01", PREDICT_START_TIME="08:05",
            STEPS=2, SAMPLES=3, TEMPERATURE=1.,
        )
        excluded = [{"id": key, "name": name} for key, name in EXCLUDED_ATTRACTIONS.items()]
        self.kept = [{"id": "a", "name": "Space Mountain"}, {"id": "b", "name": "Jungle Cruise"}]
        rides = [self.kept[0], *excluded, self.kept[1]]
        self.days = {}
        for index in range(3):
            ticks = [1793548800 + index * 86400 + tick * 300 for tick in range(4)]
            self.days[f"2026-11-{index + 1:02}"] = {
                "timestamps": ticks, "expected_timestamps": ticks, "rides": rides,
                "tokens": [[2, *([MISSING] * len(excluded)), 4] for _ in ticks],
            }
        self.save()

    def tearDown(self):
        self.directory.cleanup()

    def save(self):
        save_dataset(self.settings.DATASET, {"version": 1, "park_id": PARK_ID,
                                            "days": self.days, "incomplete_days": {}})

    def test_training_forecasting_evaluation_and_map_share_exclusions(self):
        original = self.settings.DATASET.read_bytes()
        with redirect_stdout(io.StringIO()):
            train(self.settings)
            checkpoint = torch.load(self.settings.CHECKPOINT, weights_only=True)
            self.assertEqual(checkpoint["rides"], self.kept)
            self.assertEqual(checkpoint["model_config"]["num_rides"], 2)
            result = predict(self.settings, print_output=False)
            scores = evaluate(self.settings)
        self.assertEqual(scores["horizons"][5]["scored_targets"], 6)
        for snapshot in result["snapshots"]:
            self.assertEqual([ride["id"] for ride in snapshot["rides"]], ["a", "b"])
        app = Visualizer(self.settings)
        try:
            payload = app.day_payload("2026-11-01")
            self.assertEqual(payload["rides"], self.kept)
            self.assertEqual([ride["wait"] for ride in payload["snapshots"][0]["rides"]], [10, 20])
            forecast = app.forecast(self.settings)
            self.assertEqual([ride["id"] for ride in forecast["snapshots"][0]["rides"]], ["a", "b"])
        finally:
            app.close()
        self.assertEqual(self.settings.DATASET.read_bytes(), original)
        # Ignored observations cannot affect a forecast, even when they change or are missing.
        for day in self.days.values():
            for row in day["tokens"]:
                row[1:-1] = [60] * (len(row) - 2)
        self.save()
        self.assertEqual(predict(self.settings, print_output=False), result)

    def test_old_checkpoints_require_retraining_for_both_architectures(self):
        for architecture in ("park_gru_v1", "ride_gru_attention_v1"):
            with self.subTest(architecture=architecture):
                model_config = {"num_rides": 1, "embedding_size": 4, "hidden_size": 8,
                                "architecture": architecture, "attention_heads": 2}
                torch.save({"version": 1, "park_id": PARK_ID, "token_labels": TOKEN_LABELS,
                            "rides": [{"id": "old", "name": "Main Street Cinema"}],
                            "model_config": model_config,
                            "model_state": WaitPredictor(**model_config).state_dict()},
                           self.settings.CHECKPOINT)
                for operation in (predict, evaluate):
                    with self.assertRaisesRegex(ValueError, "excluded attractions.*ml/train.py"):
                        operation(self.settings)

    def test_training_reports_when_every_attraction_is_excluded(self):
        for day in self.days.values():
            day["rides"] = day["rides"][1:-1]
            day["tokens"] = [row[1:-1] for row in day["tokens"]]
        self.save()
        with self.assertRaisesRegex(ValueError, "no eligible attractions"):
            train(self.settings)


if __name__ == "__main__":
    unittest.main()
