"""Named park models, saved split previews, and checkpoint-specific test reports."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import shutil
from uuid import uuid4

from ml import config
from ml.availability import FORECAST_POLICY_VERSION
from ml.attractions import is_excluded
from ml.data import TOKEN_LABELS
from ml.parks import settings_for


def split_preview(app, percentages=None):
    from ml.train import split_dates

    values = [config.TRAIN_SPLIT * 100, config.VAL_SPLIT * 100, config.TEST_SPLIT * 100] if percentages is None else percentages
    if not isinstance(values, (list, tuple)) or len(values) != 3 or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Enter training, validation and test percentages")
    fractions = [value / 100 for value in values]
    dataset = app.dataset() if app.settings.DATASET.exists() else {"days": {}}
    dates = sorted(key for key, day in dataset["days"].items() if len(day["timestamps"]) >= 2)
    try:
        groups = split_dates(dates, *fractions, seed=app.settings.SEED)
    except ValueError as error:
        return {"available_days": len(dates), "percentages": values, "ready": False, "error": str(error)}
    return {"available_days": len(dates), "percentages": values, "ready": True,
            "method": "random_days", "seed": app.settings.SEED,
            "groups": [{"name": name, "count": len(group), "first_date": group[0] if group else None,
                        "last_date": group[-1] if group else None, "dates": group}
                       for name, group in zip(("train", "validation", "test"), groups)]}


def model_name(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80:
        raise ValueError("Give this model a name between 1 and 80 characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError("Model names must be a single line")
    return value.strip()


def best_model_id(identifier):
    if identifier == "latest":
        return "best"
    if isinstance(identifier, str) and identifier.startswith("run-") and identifier.endswith("-latest"):
        return identifier.removesuffix("-latest") + "-best"
    return identifier


def is_best_checkpoint(checkpoint):
    source = checkpoint.get("copied_from")
    best_epoch = checkpoint.get("best_epoch")
    return source == best_model_id(source) and (
        checkpoint.get("selection") == "final_epoch" or best_epoch is None or best_epoch == checkpoint.get("epoch")
    )


class ModelLibrary:
    def __init__(self):
        self.cache = {}

    @staticmethod
    def folder(app):
        return settings_for(app.park, migrate=False).CHECKPOINT.parent

    def paths(self, app):
        folder = self.folder(app)
        paths = {"best": folder / "waits.pt", "latest": folder / "waits.latest.pt", "saved": folder / "waits.saved.pt"}
        runs = folder / "runs"
        if runs.exists():
            for run in runs.iterdir():
                if not run.is_dir() or len(run.name) != 32 or any(char not in "0123456789abcdef" for char in run.name):
                    continue
                paths[f"run-{run.name}-best"] = run / "best.pt"
                paths[f"run-{run.name}-latest"] = run / "best.latest.pt"
        if app.settings.CHECKPOINT.resolve() not in {path.resolve() for path in paths.values()}:
            paths["active-custom"] = app.settings.CHECKPOINT
        return paths

    def resolve(self, app, identifier):
        path = self.paths(app).get(identifier)
        if path is None or not path.is_file():
            raise ValueError("Select an existing model saved for this park")
        return path

    def publication_origin(self, app, identifier, paths=None):
        """Resolve latest slots and named copies to the original run's best slot."""
        paths = self.paths(app) if paths is None else paths
        identifier = best_model_id(identifier)
        seen = {identifier}
        while (path := paths.get(identifier)) is not None and path.is_file():
            copied_from = self.summary(app, identifier, path)["copied_from"]
            parent = best_model_id(copied_from)
            if parent not in paths or parent in seen:
                break
            if not paths[parent].is_file() and (copied_from not in paths or not paths[copied_from].is_file()):
                break
            seen.add(parent)
            identifier = parent
        return identifier

    def create(self, app):
        run = uuid4().hex
        path = self.folder(app) / "runs" / run / "best.pt"
        return f"run-{run}-best", path

    def discard_training_run(self, app, identifier, checkpoint):
        """Remove only the UUID run created for this job, after its process exits."""
        if not isinstance(identifier, str) or not identifier.startswith("run-") or not identifier.endswith("-best"):
            raise ValueError("Refusing to discard a model outside a newly created training run")
        run_id = identifier[4:-5]
        if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
            raise ValueError("Invalid training run identifier")
        root = (self.folder(app) / "runs").resolve()
        target = (root / run_id).resolve()
        checkpoint = Path(checkpoint)
        if target.parent != root or checkpoint.name != "best.pt" or checkpoint.resolve() != target / "best.pt":
            raise ValueError("Refusing to discard a model outside this park's training run")
        if app.settings.CHECKPOINT.resolve().is_relative_to(target):
            raise ValueError("Refusing to discard the active model")
        if target.exists():
            shutil.rmtree(target)
        self.cache = {key: value for key, value in self.cache.items() if not Path(key[0]).is_relative_to(target)}

    def activate(self, app, identifier):
        path = self.resolve(app, identifier)
        # Custom CLI paths remain usable in this process; named park models persist.
        if identifier != "active-custom":
            folder = self.folder(app)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / f".active-{uuid4().hex}.tmp"
            try:
                temporary.write_text(json.dumps({"model_id": identifier}), encoding="utf-8")
                os.replace(temporary, folder / "active-model.json")
            finally:
                temporary.unlink(missing_ok=True)
        app.settings.CHECKPOINT = path

    def restore_active(self, app):
        try:
            selection = json.loads((self.folder(app) / "active-model.json").read_text(encoding="utf-8"))
            if selection.get("model_id", "") is None:
                app.settings.CHECKPOINT = self.folder(app) / ".no-active-model.pt"
                return
            app.settings.CHECKPOINT = self.resolve(app, selection["model_id"])
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def deactivate(self, app):
        folder = self.folder(app)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f".active-{uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps({"model_id": None}), encoding="utf-8")
            os.replace(temporary, folder / "active-model.json")
        finally:
            temporary.unlink(missing_ok=True)
        app.settings.CHECKPOINT = folder / ".no-active-model.pt"

    def _deletion_files(self, app, identifier):
        root = self.folder(app).resolve()
        path = self.resolve(app, identifier)
        if identifier == "active-custom":
            raise ValueError("Only models managed in this park's model folder can be moved to Deleted")
        if identifier in ("best", "latest", "saved"):
            sources = [path] + [item for item in (path.with_suffix(".test.json"), path.with_suffix(".pt.tmp")) if item.exists()]
            if any(item.resolve() != root / item.name for item in sources):
                raise ValueError("Model files must stay inside this park's model folder")
            return root, sources, "checkpoint"
        run_id = identifier.removeprefix("run-").rsplit("-", 1)[0]
        source = path.parent.resolve()
        if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id) or source != root / "runs" / run_id:
            raise ValueError("Model run must stay inside this park's model folder")
        return root, [source], "run"

    @staticmethod
    def _move_files(pairs):
        """Move a small validated set; put earlier moves back if one fails."""
        moved = []
        try:
            for source, destination in pairs:
                if destination.exists():
                    raise ValueError("A model already exists at the destination; nothing will be overwritten")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                moved.append((source, destination))
        except Exception:
            for source, destination in reversed(moved):
                os.replace(destination, source)
            raise

    def soft_delete(self, app, identifier):
        root, sources, scope = self._deletion_files(app, identifier)
        deleted_root = root / "deleted"
        if deleted_root.resolve() != deleted_root:
            raise ValueError("The Deleted folder must stay inside this park's model folder")
        summary = self.summary(app, identifier, self.resolve(app, identifier))
        active = app.settings.CHECKPOINT.resolve()
        active_id = next((key for key, value in self.paths(app).items() if value.resolve() == active), None)
        was_active = any(active == source.resolve() or (source.is_dir() and active.is_relative_to(source.resolve())) for source in sources)
        deleted_id = uuid4().hex
        entry = deleted_root / deleted_id
        entry.mkdir(parents=True, exist_ok=False)
        records = [{"original": source.relative_to(root).as_posix(), "stored": source.name} for source in sources]
        manifest = {"version": 1, "park_id": app.park.id, "model_id": identifier, "name": summary["name"],
                    "scope": scope, "deleted_at": datetime.now(timezone.utc).isoformat(), "files": records}
        (entry / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            if identifier == "best":
                # A deliberately deleted legacy import must not reappear on restart.
                (root / ".legacy-imported").touch()
            if was_active:
                self.deactivate(app)
            self._move_files([(source, entry / "files" / record["stored"]) for source, record in zip(sources, records)])
        except Exception:
            if was_active and active_id:
                self.activate(app, active_id)
            raise
        for source in sources:
            self.cache = {key: value for key, value in self.cache.items() if not Path(key[0]).is_relative_to(source)}
        return {"id": deleted_id, "name": summary["name"], "was_active": was_active}

    def _deleted_entry(self, app, deleted_id):
        if not isinstance(deleted_id, str) or len(deleted_id) != 32 or any(char not in "0123456789abcdef" for char in deleted_id):
            raise ValueError("Select a deleted model from this park")
        root = self.folder(app).resolve()
        entry = root / "deleted" / deleted_id
        if entry.resolve() != entry:
            raise ValueError("Deleted models must stay inside this park's Deleted folder")
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("park_id") != app.park.id or manifest.get("scope") not in ("run", "checkpoint"):
            raise ValueError("This deleted model belongs to a different park or format")
        files = manifest.get("files", [])
        if not isinstance(files, list) or not all(isinstance(record, dict) and isinstance(record.get("original"), str) and isinstance(record.get("stored"), str) for record in files):
            raise ValueError("Invalid deleted model files")
        identifier = manifest.get("model_id")
        if not isinstance(identifier, str):
            raise ValueError("Invalid deleted model identifier")
        if manifest["scope"] == "run":
            run_id = identifier.removeprefix("run-").rsplit("-", 1)[0]
            if identifier not in (f"run-{run_id}-best", f"run-{run_id}-latest") or len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id) or files != [{"original": f"runs/{run_id}", "stored": run_id}]:
                raise ValueError("Invalid deleted model run")
        else:
            original = {"best": "waits.pt", "latest": "waits.latest.pt", "saved": "waits.saved.pt"}.get(identifier)
            allowed = {original, str(Path(original).with_suffix(".test.json")), original + ".tmp"} if original else set()
            if not files or len(files) > 3 or len({record["original"] for record in files}) != len(files) or any(record["original"] not in allowed or record["stored"] != record["original"] for record in files):
                raise ValueError("Invalid deleted checkpoint files")
        pairs = []
        for record in files:
            source = entry / "files" / record["stored"]
            destination = root / record["original"]
            if source.resolve() != source or destination.resolve() != destination or not source.exists():
                raise ValueError("Deleted model files are missing or outside their expected folders")
            pairs.append((source, destination))
        return manifest, pairs

    def deleted_models(self, app):
        deleted = self.folder(app) / "deleted"
        results = []
        if deleted.exists():
            for entry in deleted.iterdir():
                try:
                    manifest, pairs = self._deleted_entry(app, entry.name)
                    conflict = any(destination.exists() for _, destination in pairs)
                    results.append({"id": entry.name, "name": manifest["name"], "deleted_at": manifest["deleted_at"],
                                    "scope": manifest["scope"], "can_restore": not conflict,
                                    "reason": "A model already exists in its original location. The backup is preserved." if conflict else ""})
                except (OSError, ValueError, TypeError, KeyError):
                    continue
        return sorted(results, key=lambda item: item["deleted_at"], reverse=True)

    def restore_deleted(self, app, deleted_id):
        manifest, pairs = self._deleted_entry(app, deleted_id)
        if any(destination.exists() for _, destination in pairs):
            raise ValueError("A model already exists in the original location. The deleted backup will not be overwritten or removed.")
        self._move_files(pairs)
        # The empty backup directory/manifest is retained as a deletion record.
        return manifest["model_id"]

    def summary(self, app, identifier, path):
        import torch

        stat = path.stat()
        signature = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get(signature)
        if cached is None:
            cached = {"ready": False, "test_ready": False, "error": "", "training_days": 0,
                      "validation_days": 0, "test_days": 0, "first_training_date": None,
                      "last_training_date": None, "epoch": None, "run_name": None,
                      "copied_from": None, "publishable": False, "selection": None}
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1 or checkpoint.get("token_labels") != TOKEN_LABELS:
                    raise ValueError("Unsupported model format")
                if checkpoint.get("park_id") != app.park.id:
                    raise ValueError("This model belongs to a different park")
                excluded = [ride["name"] for ride in checkpoint["rides"] if is_excluded(ride)]
                if excluded:
                    raise ValueError("Retraining required: this model includes excluded attractions (" + ", ".join(excluded[:3]) + ")")
                groups = [checkpoint.get(name, []) for name in ("train_dates", "validation_dates", "test_dates")]
                train, validation, test = groups
                cached.update(ready=True, training_days=len(train), validation_days=len(validation), test_days=len(test),
                              first_training_date=min(train) if train else None, last_training_date=max(train) if train else None,
                              epoch=checkpoint.get("epoch"), run_name=checkpoint.get("run_name"),
                              copied_from=checkpoint.get("copied_from") if isinstance(checkpoint.get("copied_from"), str) else None,
                              publishable=is_best_checkpoint(checkpoint),
                              selection=checkpoint.get("selection"),
                              test_ready=all(name in checkpoint for name in ("train_dates", "validation_dates", "test_dates"))
                                         and bool(test) and len(set(train + validation + test)) == sum(map(len, groups)))
                if not cached["test_ready"]:
                    cached["error"] = "No valid saved test split. Train a new model with a nonzero test percentage."
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
                cached["error"] = str(error) if isinstance(error, ValueError) else "This model cannot be loaded. Train a new model."
            self.cache = {key: value for key, value in self.cache.items() if key[0] != signature[0]}
            self.cache[signature] = cached
        fallback = {"best": "Previous best model", "latest": "Previous latest epoch", "saved": "Saved copy"}.get(identifier, "Unnamed model")
        report = self.report(path)
        return {**cached, "id": identifier, "name": cached["run_name"] or fallback,
                "can_delete": identifier != "active-custom", "delete_scope": "checkpoint" if identifier in ("best", "latest", "saved") else "run",
                "variant": "latest" if path.stem.endswith("latest") else "best",
                "active": path.resolve() == app.settings.CHECKPOINT.resolve(),
                "modified_at": stat.st_mtime, "has_report": report is not None}

    @staticmethod
    def report(path):
        report_path = path.with_suffix(".test.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            stat = path.stat()
            if (report.get("checkpoint_mtime_ns") == stat.st_mtime_ns
                    and report.get("checkpoint_size") == stat.st_size
                    and report.get("forecast_policy_version") == FORECAST_POLICY_VERSION):
                return report
        except (OSError, ValueError, TypeError):
            pass
        return None

    def catalog(self, app, percentages=None):
        models = [self.summary(app, identifier, path) for identifier, path in self.paths(app).items() if path.is_file()]
        models.sort(key=lambda model: model["modified_at"], reverse=True)
        return {"models": models, "active_model": next((model["id"] for model in models if model["active"]), None),
                "deleted": self.deleted_models(app),
                "preview": split_preview(app, percentages),
                "storage": {"Cached dataset (.json.gz)": str(app.settings.DATASET.resolve()),
                            "Raw history (.json.gz)": str((app.settings.DATASET.parent / "themeparks-history").resolve()),
                            "Models and test reports": str(self.folder(app).resolve()),
                            "Deleted models (recoverable)": str((self.folder(app) / "deleted").resolve()),
                            "Training and test logs": str((app.settings.DATASET.parent / "jobs").resolve())},
                "default_name": f"{app.park.name} · {datetime.now(app.park.tz):%b %d, %Y %H:%M}"}

    def save_copy(self, app, identifier, name):
        import torch
        from ml.train import save_checkpoint

        name = model_name(name)
        source = self.resolve(app, identifier)
        summary = self.summary(app, identifier, source)
        if not summary["ready"]:
            raise ValueError(summary["error"])
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        checkpoint.update(run_name=name, copied_from=identifier, copied_at=datetime.now(timezone.utc).isoformat())
        new_id, target = self.create(app)
        target.parent.mkdir(parents=True, exist_ok=False)
        save_checkpoint(checkpoint, target)
        return new_id
