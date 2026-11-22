"""Local park management jobs. Network/data work runs outside the HTTP threads."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from ml import config
from ml.parks import get_park, settings_for
from ml.model_library import ModelLibrary, model_name, split_preview


class ParkLab:
    def __init__(self, device="auto", samples=100, refresh_locations=False):
        self.device, self.samples = device, samples
        self.refresh_locations = refresh_locations
        self.apps, self.tasks = {}, {}
        self.lock = threading.RLock()
        self.library = ModelLibrary()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="park-data-model")

    def app(self, slug):
        from ml.visualize import Visualizer, load_locations

        park = get_park(slug)
        with self.lock:
            if park.slug not in self.apps:
                settings = settings_for(park, DEVICE=self.device, SAMPLES=self.samples)
                locations, warning = load_locations(settings.LOCATIONS, self.refresh_locations, park)
                self.apps[park.slug] = Visualizer(settings, locations, warning)
                self.library.restore_active(self.apps[park.slug])
            return self.apps[park.slug]

    def status(self, slug):
        park = get_park(slug)
        with self.lock:
            task = self.tasks.get(park.slug)
            if not task:
                return {"state": "idle", "log": []}
            return {**{key: list(value) if key == "log" else value
                       for key, value in task.items() if key != "future" and not key.startswith("_")},
                    "elapsed_seconds": max(0, int(task.get("finished_at", time.time()) - task.get("started_at", time.time())))}

    def catalog(self, slug, percentages=None):
        with self.lock:
            return self.library.catalog(self.app(slug), percentages)

    def report(self, slug, identifier):
        app = self.app(slug)
        path = self.library.resolve(app, identifier)
        return {"report": self.library.report(path)}

    def log(self, slug):
        with self.lock:
            task = self.tasks.get(get_park(slug).slug)
            path = task.get("_log_path") if task else None
        return path.read_bytes() if path and path.exists() else b"No job log yet."

    def _finish_cancel(self, task, app):
        try:
            self.library.discard_training_run(app, task["model_id"], task["_checkpoint"])
        except Exception as error:
            task.update(state="failed", error=f"Training stopped, but its unfinished model files could not be discarded: {error}",
                        finished_at=time.time(), discarded=False)
            raise
        message = "Training stopped. The unfinished model was discarded; the previously selected model is unchanged."
        task.update(state="cancelled", error="", discarded=True, finished_at=time.time())
        task["log"].append(message)
        task["_log_path"].parent.mkdir(parents=True, exist_ok=True)
        with task["_log_path"].open("a", encoding="utf-8") as log:
            log.write(message + "\n")

    @staticmethod
    def _stop_process(process):
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                if process.poll() is None:
                    raise

    def cancel_training(self, app, job_id):
        with self.lock:
            task = self.tasks.get(app.park.slug)
            if not task or not isinstance(job_id, str) or task["id"] != job_id:
                raise ValueError("This training job is no longer current. Refresh the job status.")
            if task["action"] != "train" or task["state"] not in ("queued", "running", "cancelling"):
                raise ValueError("Only queued or running training can be stopped and discarded")
            task["_cancel"].set()
            task["state"] = "cancelling"
            if task["future"].cancel():
                self._finish_cancel(task, app)
            else:
                self._stop_process(task.get("_process"))
            return {"state": task["state"], "message": "Stopping training and discarding its unfinished model."}

    def action(self, payload):
        park = get_park(payload.get("park", config.PARK))
        action = payload.get("action")
        if action not in ("refresh", "rebuild", "live", "train", "test", "load", "save", "cancel", "delete", "restore"):
            raise ValueError("Choose refresh, rebuild, live, train, test, load, save, cancel, delete, or restore")
        app = self.app(park.slug)
        if action == "cancel":
            return self.cancel_training(app, payload.get("job_id"))
        with self.lock:
            old = self.tasks.get(park.slug)
            if old and old["state"] in ("queued", "running", "cancelling"):
                raise ValueError("This park already has a data/model job in progress")
            if action in ("delete", "restore"):
                with app.lock:
                    if any(not future.done() for _, future in app.jobs.values()):
                        raise ValueError("Wait for this park's forecasts to finish before deleting or restoring models")
                    if action == "delete":
                        deleted = self.library.soft_delete(app, payload.get("model"))
                        message = f"Moved {deleted['name']} to Deleted. Restore it below whenever needed."
                        if deleted["was_active"]:
                            message += " No model is running; select a saved model and choose Use for forecasts."
                        result = {"state": "complete", "deleted_id": deleted["id"], "message": message}
                    else:
                        identifier = self.library.restore_deleted(app, payload.get("deleted_id"))
                        result = {"state": "complete", "model_id": identifier,
                                  "message": "Restored the model and its test reports. Choose Use for forecasts to activate it."}
                    app.jobs.clear()
                    app.checkpoint_cache = None
                    return result
            if action in ("load", "save"):
                if action == "save" and "name" in payload:
                    identifier = self.library.save_copy(app, payload.get("model", "best"), payload["name"])
                    return {"state": "complete", "model_id": identifier, "message": "Saved a new named copy. Select Use for forecasts to run it."}
                self.checkpoint_action(app, action, payload.get("model", "best"))
                return {"state": "complete", "model_id": payload.get("model", "best"),
                        "message": "This model now powers this park's forecasts." if action == "load" else "Saved a model copy."}
            module = "ml.train" if action == "train" else "ml.test" if action == "test" else "ml.data"
            command = [sys.executable, "-u", "-m", module, "--park", park.slug]
            identifier, checkpoint_path, name = None, None, action
            if action == "train":
                epochs = payload.get("epochs", config.EPOCHS)
                if type(epochs) is not int or not 1 <= epochs <= 10000:
                    raise ValueError("Epochs must be an integer between 1 and 10000")
                name = model_name(payload.get("name", f"{park.name} model"))
                preview = split_preview(app, payload.get("split"))
                if not preview["ready"]:
                    raise ValueError(preview["error"])
                if type(payload.get("use_after_train", True)) is not bool:
                    raise ValueError("Choose whether to use this model after training")
                identifier, checkpoint_path = self.library.create(app)
                command += ["--device", self.device, "--epochs", str(epochs), "--seed", str(app.settings.SEED), "--checkpoint", str(checkpoint_path),
                            "--dataset", str(app.settings.DATASET), "--run-name", name]
                for flag, value in zip(("--train-split", "--val-split", "--test-split"), preview["percentages"]):
                    command += [flag, str(value / 100)]
            elif action == "test":
                identifier = payload.get("model", "best")
                checkpoint_path = self.library.resolve(app, identifier)
                summary = self.library.summary(app, identifier, checkpoint_path)
                if not summary["test_ready"]:
                    raise ValueError(summary["error"])
                name = summary["name"]
                command += ["--device", self.device, "--checkpoint", str(checkpoint_path),
                            "--dataset", str(app.settings.DATASET), "--output", str(checkpoint_path.with_suffix(".test.json"))]
            elif action == "rebuild":
                command += ["--refresh"]
            elif action == "live":
                command += ["--live-only"]
            log_path = app.settings.DATASET.parent / "jobs" / f"{uuid4().hex}.log"
            task = {"id": uuid4().hex, "state": "queued", "action": action, "name": name,
                    "model_id": identifier, "log": deque(maxlen=4000), "error": "", "current": 0,
                    "total": payload.get("epochs", config.EPOCHS) if action == "train" else 0,
                    "_checkpoint": checkpoint_path, "_use_after_train": payload.get("use_after_train", True),
                    "_log_path": log_path, "_cancel": threading.Event()}
            self.tasks[park.slug] = task
            task["future"] = self.worker.submit(self.run, command, task, app)
            return {"state": "queued", "model_id": identifier}

    def run(self, command, task, app):
        with self.lock:
            if task.get("_cancel") and task["_cancel"].is_set():
                self._finish_cancel(task, app)
                return
            task["state"] = "running"
            task["started_at"] = time.time()
        try:
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            task["_log_path"].parent.mkdir(parents=True, exist_ok=True)
            with task["_log_path"].open("w", encoding="utf-8") as log, subprocess.Popen(command, cwd=config.ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **options) as process:
                with self.lock:
                    task["_process"] = process
                    if task.get("_cancel") and task["_cancel"].is_set():
                        self._stop_process(process)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with self.lock:
                        task["log"].append(line.rstrip())
                        epoch = re.search(r"^Epoch\s+(\d+):", line)
                        test_day = re.search(r"Evaluating test day (\d+)/(\d+)", line)
                        if epoch:
                            task["current"] = int(epoch[1])
                        if test_day:
                            task.update(current=int(test_day[1]), total=int(test_day[2]))
                code = process.wait()
            with self.lock:
                task.pop("_process", None)
                if task.get("_cancel") and task["_cancel"].is_set():
                    self._finish_cancel(task, app)
                    return
                if code == 0 and task["action"] == "train":
                    if not task["_checkpoint"].exists():
                        raise RuntimeError("Training did not save a model. See the log.")
                    if task["_use_after_train"]:
                        self.library.activate(app, task["model_id"])
                task["state"] = "complete" if code == 0 else "failed"
                task["finished_at"] = time.time()
                task["error"] = "" if code == 0 else f"Job exited with code {code}. See the log; saved progress is preserved."
        except Exception as error:
            with self.lock:
                task.pop("_process", None)
                task.update(state="failed", error=task.get("error") or str(error), finished_at=time.time())

    def checkpoint_action(self, app, action, model):
        import torch
        from ml.attractions import validate_checkpoint_rides
        from ml.model import build_model
        from ml.data import TOKEN_LABELS

        folder = settings_for(app.park, migrate=False).CHECKPOINT.parent
        path = self.library.resolve(app, model)
        source = app.settings.CHECKPOINT if action == "save" else path
        if not source.exists():
            raise ValueError("This park has no checkpoint in that slot. Train or select another model.")
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1 or checkpoint.get("token_labels") != TOKEN_LABELS:
            raise ValueError("Unsupported checkpoint or token vocabulary")
        if checkpoint.get("park_id") != app.park.id:
            raise ValueError("Checkpoint belongs to a different park")
        validate_checkpoint_rides(checkpoint["rides"])
        build_model(**checkpoint["model_config"]).load_state_dict(checkpoint["model_state"])
        if action == "save":
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "waits.saved.pt"
            if source != path:
                temporary = path.with_suffix(".pt.tmp")
                shutil.copy2(source, temporary)
                temporary.replace(path)
        else:
            self.library.activate(app, model)

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=True)
        for app in self.apps.values():
            app.close()
