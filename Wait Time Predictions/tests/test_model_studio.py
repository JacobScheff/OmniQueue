"""Named model jobs and evaluation of saved holdouts through the UI backend."""

import hashlib
from collections import deque
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

import torch

from ml.data import TOKEN_LABELS, load_dataset, save_dataset
from ml.availability import FORECAST_POLICY_VERSION
from ml.lab import ParkLab
from ml.model_library import model_name, split_preview
from ml.parks import settings_for
from ml.visualize import Visualizer


class ModelStudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.settings = settings_for("epcot", root=self.root, migrate=False, DEVICE="cpu")
        rides = [{"id": "ride", "name": "Ride"}]
        days = {f"2026-11-{index:02}": {"timestamps": [1793538000 + index * 86400 + step * 300 for step in range(4)],
                                      "rides": rides, "tokens": [[2], [3], [2], [4]], "timezone": "America/New_York"}
                for index in range(1, 7)}
        save_dataset(self.settings.DATASET, {"version": 1, "park_id": self.settings.PARK_ID, "days": days})
        self.app = Visualizer(self.settings)
        self.lab = ParkLab(device="cpu")
        self.lab.apps["epcot"] = self.app
        self.folder_patch = patch("ml.model_library.ModelLibrary.folder", side_effect=lambda app: self.root / "models" / app.park.slug)
        self.folder_patch.start()

    def tearDown(self):
        self.lab.close()
        self.folder_patch.stop()
        self.directory.cleanup()

    def finish(self):
        self.lab.tasks["epcot"]["future"].result(timeout=90)
        status = self.lab.status("epcot")
        self.assertEqual(status["state"], "complete", status["error"] + "\n" + "\n".join(status["log"][-15:]))
        return status

    def saved_run(self):
        identifier, path = self.lab.library.create(self.app)
        path.parent.mkdir(parents=True)
        torch.save({"version": 1, "park_id": self.app.park.id, "token_labels": TOKEN_LABELS,
                    "rides": [{"id": "ride", "name": "Ride"}], "run_name": "Keep me",
                    "train_dates": ["2026-11-01"], "validation_dates": ["2026-11-02"],
                    "test_dates": ["2026-11-03"]}, path)
        latest = path.with_name("best.latest.pt")
        latest.write_bytes(path.read_bytes())
        for checkpoint in (path, latest):
            stat = checkpoint.stat()
            checkpoint.with_suffix(".test.json").write_text(json.dumps({"checkpoint_mtime_ns": stat.st_mtime_ns,
                                                                      "forecast_policy_version": FORECAST_POLICY_VERSION,
                                                                      "checkpoint_size": stat.st_size}), encoding="utf-8")
        return identifier, path

    def test_delete_run_preserves_all_files_and_restore_does_not_reactivate(self):
        identifier, path = self.saved_run()
        originals = {file.name: file.read_bytes() for file in path.parent.iterdir()}
        data = self.settings.DATASET.read_bytes()
        self.lab.library.activate(self.app, identifier.replace("-best", "-latest"))
        result = self.lab.action({"park": "epcot", "action": "delete", "model": identifier})
        self.assertEqual(result["state"], "complete")
        self.assertFalse(path.parent.exists())
        self.assertFalse(self.app.settings.CHECKPOINT.exists())
        catalog = self.lab.catalog("epcot")
        self.assertEqual(catalog["models"], [])
        self.assertEqual(len(catalog["deleted"]), 1)
        deleted = catalog["deleted"][0]
        self.assertTrue(deleted["can_restore"])
        backup = self.lab.library.folder(self.app) / "deleted" / deleted["id"] / "files" / path.parent.name
        self.assertEqual({file.name: file.read_bytes() for file in backup.iterdir()}, originals)
        restarted = Visualizer(settings_for("epcot", root=self.root, migrate=False))
        try:
            self.lab.library.restore_active(restarted)
            self.assertEqual(restarted.settings.CHECKPOINT, self.app.settings.CHECKPOINT)
        finally:
            restarted.close()
        restored = self.lab.action({"park": "epcot", "action": "restore", "deleted_id": deleted["id"]})
        self.assertEqual(restored["model_id"], identifier)
        self.assertEqual({file.name: file.read_bytes() for file in path.parent.iterdir()}, originals)
        self.assertIsNotNone(self.lab.library.report(path))
        self.assertFalse(self.app.settings.CHECKPOINT.exists())
        self.assertEqual(self.lab.catalog("epcot")["deleted"], [])
        self.assertEqual(self.settings.DATASET.read_bytes(), data)

    def test_deleting_inactive_model_preserves_active_model(self):
        identifier, path = self.saved_run()
        active_id, active_path = self.saved_run()
        self.lab.library.activate(self.app, active_id)
        selection = (self.lab.library.folder(self.app) / "active-model.json").read_bytes()
        self.lab.action({"park": "epcot", "action": "delete", "model": identifier})
        self.assertFalse(path.exists())
        self.assertTrue(active_path.exists())
        self.assertEqual(self.app.settings.CHECKPOINT, active_path)
        self.assertEqual((self.lab.library.folder(self.app) / "active-model.json").read_bytes(), selection)

    def test_restore_collision_preserves_backup_and_existing_model(self):
        identifier, path = self.saved_run()
        result = self.lab.action({"park": "epcot", "action": "delete", "model": identifier})
        path.parent.mkdir(parents=True)
        path.write_bytes(b"keep existing model")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.lab.action({"park": "epcot", "action": "restore", "deleted_id": result["deleted_id"]})
        self.assertEqual(path.read_bytes(), b"keep existing model")
        deleted = self.lab.catalog("epcot")["deleted"][0]
        self.assertFalse(deleted["can_restore"])
        self.assertTrue((self.lab.library.folder(self.app) / "deleted" / deleted["id"] / "files" / path.parent.name / "best.pt").exists())

    def test_legacy_checkpoint_deletion_moves_report_but_not_other_slots(self):
        _, source = self.saved_run()
        folder = self.lab.library.folder(self.app)
        path = folder / "waits.pt"
        path.write_bytes(source.read_bytes())
        report = path.with_suffix(".test.json")
        report.write_bytes(b"test report")
        other = folder / "waits.latest.pt"
        other.write_bytes(b"other slot")
        result = self.lab.action({"park": "epcot", "action": "delete", "model": "best"})
        self.assertFalse(path.exists())
        self.assertFalse(report.exists())
        self.assertEqual(other.read_bytes(), b"other slot")
        self.assertTrue((folder / ".legacy-imported").exists())
        self.lab.action({"park": "epcot", "action": "restore", "deleted_id": result["deleted_id"]})
        self.assertEqual(path.read_bytes(), source.read_bytes())
        self.assertEqual(report.read_bytes(), b"test report")

    def test_delete_waits_for_running_jobs_and_forecasts(self):
        identifier, path = self.saved_run()
        self.lab.tasks["epcot"] = {"state": "running"}
        with self.assertRaisesRegex(ValueError, "in progress"):
            self.lab.action({"park": "epcot", "action": "delete", "model": identifier})
        self.lab.tasks.clear()
        future = Future()
        self.app.jobs["forecast"] = (None, future)
        with self.assertRaisesRegex(ValueError, "forecasts to finish"):
            self.lab.action({"park": "epcot", "action": "delete", "model": identifier})
        self.assertTrue(path.exists())
        future.set_result(None)
        self.lab.action({"park": "epcot", "action": "delete", "model": identifier})
        self.assertEqual(len(self.app.jobs), 0)

    def test_restore_rejects_other_park_and_manifest_path_traversal(self):
        identifier, path = self.saved_run()
        result = self.lab.action({"park": "epcot", "action": "delete", "model": identifier})
        manifest_path = self.lab.library.folder(self.app) / "deleted" / result["deleted_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        other = settings_for("disneyland", root=self.root, migrate=False).PARK_ID
        manifest_path.write_text(json.dumps({**manifest, "park_id": other}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different park"):
            self.lab.library.restore_deleted(self.app, result["deleted_id"])
        manifest["files"][0]["original"] = "../../outside"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid deleted model run"):
            self.lab.library.restore_deleted(self.app, result["deleted_id"])
        with self.assertRaises(ValueError):
            self.lab.library.restore_deleted(self.app, "../outside")
        self.assertFalse(path.exists())

    def test_multi_file_move_failure_rolls_back_without_losing_active_selection(self):
        _, source = self.saved_run()
        path = self.lab.library.folder(self.app) / "waits.pt"
        path.write_bytes(source.read_bytes())
        report = path.with_suffix(".test.json")
        report.write_bytes(b"report")
        self.lab.library.activate(self.app, "best")
        from os import replace

        def fail_report_move(source, destination):
            if Path(source) == report:
                raise OSError("simulated move failure")
            return replace(source, destination)

        with patch("ml.model_library.os.replace", side_effect=fail_report_move):
            with self.assertRaisesRegex(OSError, "simulated move failure"):
                self.lab.library.soft_delete(self.app, "best")
        self.assertEqual(path.read_bytes(), source.read_bytes())
        self.assertEqual(report.read_bytes(), b"report")
        self.assertEqual(self.app.settings.CHECKPOINT, path)
        self.assertEqual(json.loads((path.parent / "active-model.json").read_text(encoding="utf-8"))["model_id"], "best")

    def test_named_training_test_saved_dates_copy_and_activation_are_separate(self):
        previous = self.app.settings.CHECKPOINT
        result = self.lab.action({"park": "epcot", "action": "train", "name": "First model", "epochs": 1,
                                  "split": [60, 20, 20], "use_after_train": False})
        first = result["model_id"]
        with self.assertRaisesRegex(ValueError, "in progress"):
            self.lab.action({"park": "epcot", "action": "train", "name": "Overlap", "epochs": 1})
        status = self.finish()
        self.assertEqual(self.app.settings.CHECKPOINT, previous)
        path = self.lab.library.resolve(self.app, first)
        with self.assertRaisesRegex(ValueError, "Only queued or running"):
            self.lab.action({"park": "epcot", "action": "cancel", "job_id": status["id"]})
        self.assertTrue(path.exists())
        checkpoint = torch.load(path, weights_only=True)
        self.assertEqual(checkpoint["run_name"], "First model")
        self.assertEqual(len(checkpoint["test_dates"]), 1)
        preview = split_preview(self.app, [60, 20, 20])
        self.assertEqual(preview["seed"], checkpoint["split"]["seed"])
        for group, key in zip(preview["groups"], ("train_dates", "validation_dates", "test_dates")):
            self.assertEqual(group["dates"], checkpoint[key])
        self.assertIn("Epoch", self.lab.log("epcot").decode("utf-8"))
        self.assertEqual(status["current"], 1)
        catalog = self.lab.catalog("epcot")
        self.assertIsNone(catalog["active_model"])
        self.assertEqual(next(model for model in catalog["models"] if model["id"] == first)["test_days"], 1)

        # Growing the archive must not change the checkpoint's held-out dates.
        dataset = load_dataset(self.settings.DATASET)
        dataset["days"]["2026-11-07"] = dataset["days"]["2026-11-06"]
        save_dataset(self.settings.DATASET, dataset)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.lab.action({"park": "epcot", "action": "test", "model": first})
        self.finish()
        report = self.lab.report("epcot", first)["report"]
        self.assertEqual(report["dates"], checkpoint["test_dates"])
        self.assertNotIn("2026-11-07", report["dates"])
        self.assertFalse(set(report["dates"]) & set(checkpoint["train_dates"] + checkpoint["validation_dates"]))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertIn("5", report["horizons"])

        copied = self.lab.action({"park": "epcot", "action": "save", "model": first, "name": "Preferred copy"})["model_id"]
        self.assertNotEqual(first, copied)
        self.assertEqual(self.app.settings.CHECKPOINT, previous)
        self.lab.action({"park": "epcot", "action": "load", "model": copied})
        self.assertEqual(self.app.settings.CHECKPOINT, self.lab.library.resolve(self.app, copied))
        self.assertEqual(torch.load(self.app.settings.CHECKPOINT, weights_only=True)["run_name"], "Preferred copy")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        restarted = Visualizer(settings_for("epcot", root=self.root, migrate=False))
        try:
            self.lab.library.restore_active(restarted)
            self.assertEqual(restarted.settings.CHECKPOINT, self.app.settings.CHECKPOINT)
        finally:
            restarted.close()

    def test_split_preview_and_model_names_are_validated_before_jobs_start(self):
        preview = split_preview(self.app, [70, 15, 15])
        self.assertEqual([group["count"] for group in preview["groups"]], [4, 1, 1])
        self.assertFalse(split_preview(self.app, [80, 15, 15])["ready"])
        for percentages in ([], [True, 0, 0], [float("nan"), 0, 0]):
            with self.assertRaises(ValueError):
                split_preview(self.app, percentages)
        for name in ("", "\n", "x" * 81):
            with self.assertRaises(ValueError):
                model_name(name)
        with self.assertRaises(ValueError):
            self.lab.library.resolve(self.app, "../../another-park/waits.pt")
        self.assertEqual(self.lab.status("epcot")["state"], "idle")

    def test_stale_test_reports_are_not_attached_to_new_weights(self):
        path = self.root / "model.pt"
        path.write_bytes(b"first")
        stat = path.stat()
        report = {"checkpoint_mtime_ns": stat.st_mtime_ns, "checkpoint_size": stat.st_size,
                  "forecast_policy_version": FORECAST_POLICY_VERSION, "dates": ["2026-11-06"]}
        path.with_suffix(".test.json").write_text(json.dumps(report), encoding="utf-8")
        self.assertIsNotNone(self.lab.library.report(path))
        path.write_bytes(b"different weights")
        self.assertIsNone(self.lab.library.report(path))

    def test_reports_from_an_older_forecast_policy_need_reevaluation(self):
        path = self.root / "model.pt"
        path.write_bytes(b"same weights")
        stat = path.stat()
        report = {"checkpoint_mtime_ns": stat.st_mtime_ns, "checkpoint_size": stat.st_size}
        for version in (None, FORECAST_POLICY_VERSION - 1):
            if version is not None:
                report["forecast_policy_version"] = version
            path.with_suffix(".test.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertIsNone(self.lab.library.report(path))

    def test_full_log_download_preserves_lines_beyond_the_live_tail(self):
        task = {"state": "queued", "action": "refresh", "log": deque(maxlen=4000), "error": "",
                "_log_path": self.root / "jobs/full.log"}
        self.lab.tasks["epcot"] = task
        process = Mock()
        process.stdout = iter(f"line {index}\n" for index in range(4500))
        process.wait.return_value = 0
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        with patch("ml.lab.subprocess.Popen", return_value=process):
            self.lab.run(["preview"], task, self.app)
        self.assertEqual(len(self.lab.status("epcot")["log"]), 4000)
        full = self.lab.log("epcot").decode("utf-8").splitlines()
        self.assertEqual((len(full), full[0], full[-1]), (4500, "line 0", "line 4499"))

    def test_stopping_running_training_kills_the_child_and_discards_only_its_run(self):
        old_id, old_path = self.lab.library.create(self.app)
        old_path.parent.mkdir(parents=True)
        old_path.write_bytes(b"previously trained weights")
        self.lab.library.activate(self.app, old_id)
        active_file = self.lab.library.folder(self.app) / "active-model.json"
        prior_selection = active_file.read_bytes()
        real_popen = subprocess.Popen
        processes = []

        def training_child(command, **options):
            target = command[command.index("--checkpoint") + 1]
            script = ("from pathlib import Path; import sys,time; p=Path(sys.argv[1]); "
                      "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'partial'); "
                      "p.with_name('best.latest.pt').write_bytes(b'partial latest'); "
                      "p.with_name('best.pt.tmp').write_bytes(b'interrupted save'); "
                      "print('preview-ready',flush=True); time.sleep(30)")
            process = real_popen([sys.executable, "-u", "-c", script, target], **options)
            processes.append(process)
            return process

        try:
            with patch("ml.lab.subprocess.Popen", side_effect=training_child):
                self.lab.action({"park": "epcot", "action": "train", "name": "Discard this", "epochs": 20})
                task = self.lab.tasks["epcot"]
                deadline = time.monotonic() + 10
                while "preview-ready" not in self.lab.status("epcot")["log"]:
                    if time.monotonic() > deadline:
                        self.fail("The training child did not start")
                    time.sleep(.01)
                with self.assertRaisesRegex(ValueError, "no longer current"):
                    self.lab.action({"park": "epcot", "action": "cancel", "job_id": "stale-job"})
                self.lab.action({"park": "epcot", "action": "cancel", "job_id": task["id"]})
                task["future"].result(timeout=10)
            status = self.lab.status("epcot")
            self.assertEqual(status["state"], "cancelled")
            self.assertTrue(status["discarded"])
            self.assertIsNotNone(processes[0].poll())
            self.assertFalse(task["_checkpoint"].parent.exists())
            self.assertEqual(old_path.read_bytes(), b"previously trained weights")
            self.assertEqual(active_file.read_bytes(), prior_selection)
            self.assertEqual(self.app.settings.CHECKPOINT, old_path)
            self.assertIn("unfinished model was discarded", self.lab.log("epcot").decode("utf-8"))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_queued_training_can_be_discarded_without_starting_a_process(self):
        release, started = threading.Event(), threading.Event()

        def occupy_worker():
            started.set()
            release.wait(10)

        blocker = self.lab.worker.submit(occupy_worker)
        self.assertTrue(started.wait(5))
        try:
            with patch("ml.lab.subprocess.Popen", side_effect=AssertionError("Queued training must not start")):
                self.lab.action({"park": "epcot", "action": "train", "name": "Queued model", "epochs": 1})
                task = self.lab.tasks["epcot"]
                self.lab.action({"park": "epcot", "action": "cancel", "job_id": task["id"]})
                self.assertTrue(task["future"].cancelled())
                self.assertEqual(self.lab.status("epcot")["state"], "cancelled")
                self.assertFalse(task["_checkpoint"].parent.exists())
        finally:
            release.set()
            blocker.result(timeout=5)

    def test_discard_refuses_paths_outside_the_generated_park_run(self):
        identifier, path = self.lab.library.create(self.app)
        other = self.root / "other-model/best.pt"
        other.parent.mkdir()
        other.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "outside this park"):
            self.lab.library.discard_training_run(self.app, identifier, other)
        self.assertEqual(other.read_bytes(), b"keep")

    def test_stop_requested_during_process_start_cannot_leave_a_training_child_running(self):
        spawning, attach = threading.Event(), threading.Event()
        process = Mock()
        process.stdout = iter([])
        process.poll.return_value = None
        process.wait.return_value = -9
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)

        def spawn(command, **options):
            path = Path(command[command.index("--checkpoint") + 1])
            path.parent.mkdir(parents=True)
            path.write_bytes(b"partial")
            spawning.set()
            self.assertTrue(attach.wait(5))
            return process

        with patch("ml.lab.subprocess.Popen", side_effect=spawn):
            self.lab.action({"park": "epcot", "action": "train", "name": "Starting", "epochs": 1})
            try:
                self.assertTrue(spawning.wait(5))
                task = self.lab.tasks["epcot"]
                self.lab.action({"park": "epcot", "action": "cancel", "job_id": task["id"]})
                self.assertEqual(self.lab.status("epcot")["state"], "cancelling")
            finally:
                attach.set()
            task["future"].result(timeout=5)
        process.kill.assert_called_once()
        self.assertEqual(self.lab.status("epcot")["state"], "cancelled")
        self.assertFalse(task["_checkpoint"].parent.exists())


if __name__ == "__main__":
    unittest.main()
