"""Checkpoint publication survives transient locks without losing saved epochs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ml.train import save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_transient_lock_retries_then_publishes_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waits.pt"
            torch.save({"epoch": 47}, path)
            temporary = path.with_suffix(".pt.tmp")
            original_replace = Path.replace
            attempts = []

            def replace(source, target):
                attempts.append(source)
                if len(attempts) <= 2:
                    self.assertEqual(torch.load(path, weights_only=True)["epoch"], 47)
                    raise PermissionError("File is locked")
                return original_replace(source, target)

            with patch.object(Path, "replace", new=replace), patch("ml.train.time.sleep") as sleep:
                save_checkpoint({"epoch": 48, "weights": torch.tensor([1., 2.])}, path)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["epoch"], 48)
            torch.testing.assert_close(saved["weights"], torch.tensor([1., 2.]))
            self.assertFalse(temporary.exists())

    def test_persistent_lock_preserves_old_checkpoint_and_completed_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waits.pt"
            torch.save({"epoch": 47}, path)
            with patch.object(Path, "replace", side_effect=PermissionError("File is locked")) as replace, \
                    patch("ml.train.time.sleep") as sleep:
                with self.assertRaisesRegex(PermissionError, "completed checkpoint is preserved"):
                    save_checkpoint({"epoch": 48}, path)
            self.assertEqual(replace.call_count, 41)
            self.assertEqual(sleep.call_count, 40)
            self.assertEqual(torch.load(path, weights_only=True)["epoch"], 47)
            self.assertEqual(torch.load(path.with_suffix(".pt.tmp"), weights_only=True)["epoch"], 48)

    def test_other_replace_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waits.pt"
            with patch.object(Path, "replace", side_effect=OSError("Disk failure")), \
                    patch("ml.train.time.sleep") as sleep:
                with self.assertRaisesRegex(OSError, "Disk failure"):
                    save_checkpoint({"epoch": 48}, path)
            sleep.assert_not_called()
            self.assertEqual(torch.load(path.with_suffix(".pt.tmp"), weights_only=True)["epoch"], 48)


if __name__ == "__main__":
    unittest.main()
