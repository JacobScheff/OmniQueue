"""Explore cached park waits and model forecasts on a local interactive map.

Run ``python ml/visualize.py`` after exporting data and training a checkpoint.
The browser provides day/start controls, a five-minute timeline, and map overlays.
"""

import argparse
import json
import logging
import math
import pickle
import sys
import threading
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

# Allow direct execution from any working directory, like the other ML scripts.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.attractions import is_excluded, modeling_day
from ml.parks import PARKS, get_park
from ml.data import (
    CLOSED, DOWN, MISSING, OVERFLOW, REFURBISHMENT,
    TICK_SECONDS, TOKEN_LABELS, UNKNOWN, WAIT_STEP, interpolate_day, load_dataset,
)
from ml.run import park_timestamp, predict

ASSETS = Path(__file__).with_name("visualizer")
PARK_ID = "7340550b-c14d-4def-80bb-acdb51d49a66"
logger = logging.getLogger(__name__)


def load_locations(path, refresh=False, park="disneyland"):
    park = get_park(park)
    """Cache public attraction coordinates separately from the wait archive."""
    cached = {}
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            logger.warning("Could not read attraction location cache; fetching it again.")
    if cached and not refresh:
        return cached, ""
    try:
        import httpx

        response = httpx.get(f"https://api.themeparks.wiki/v1/entity/{park.id}/children", timeout=15,
                             headers={"User-Agent": "OmniQueueVisualizer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if payload.get("id") != park.id:
            raise ValueError("The location catalog belongs to a different park")
        locations = {}
        for entity in payload["children"]:
            point = entity.get("location") or {}
            lat, lon = point.get("latitude"), point.get("longitude")
            if (entity.get("entityType") == "ATTRACTION"
                    and isinstance(lat, (int, float)) and isinstance(lon, (int, float))
                    and math.isfinite(lat) and math.isfinite(lon)
                    and abs(lat - park.latitude) < .06 and abs(lon - park.longitude) < .06):
                locations[entity["id"]] = {"latitude": lat, "longitude": lon}
        if not locations:
            raise ValueError("The attraction catalog contains no park coordinates")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(locations, indent=2) + "\n", encoding="utf-8")
        return locations, ""
    except (OSError, ValueError, KeyError, httpx.HTTPError) as error:
        logger.warning("Attraction coordinates unavailable: %s", error)
        warning = ("Could not refresh attraction coordinates. Using the saved locations." if cached else
                   "Attraction coordinates could not load. Waits are available in the list; "
                   "connect to the internet and restart to place them on the map.")
        return cached, warning


def observed_ride(ride, token):
    result = {"id": ride["id"], "name": ride["name"], "wait": None}
    if 0 <= token < OVERFLOW:
        result.update(status="OPERATING", wait=token * WAIT_STEP)
    elif token == OVERFLOW:
        result.update(status="OPERATING", wait="300+")
    else:
        result["status"] = {DOWN: "DOWN", CLOSED: "CLOSED", REFURBISHMENT: "REFURBISHMENT",
                            UNKNOWN: "UNKNOWN", MISSING: "MISSING"}.get(token, "UNKNOWN")
    return result


class Visualizer:
    """Read snapshots on demand, with serialized inference and a small job cache."""

    def __init__(self, settings, locations=None, location_warning=""):
        self.settings = settings
        self.park = get_park(getattr(settings, "PARK", "disneyland"))
        self.locations = locations or {}
        self.location_warning = location_warning
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast")
        self.jobs = OrderedDict()
        self.next_job = 0
        self.checkpoint_cache = None

    def checkpoint_status(self):
        """Check model compatibility before offering forecasts; reload after a save."""
        import torch

        path = self.settings.CHECKPOINT
        if not path.exists():
            return {"ready": False, "state": "missing", "training_days": 0,
                    "message": "No trained model for this park. Open Data & models and select Start training a new model. Recorded waits are available."}
        try:
            stat = path.stat()
            signature = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
            with self.lock:
                if self.checkpoint_cache and self.checkpoint_cache[0] == signature:
                    return dict(self.checkpoint_cache[1])
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("version") != 1 or checkpoint.get("token_labels") != TOKEN_LABELS:
                raise ValueError("Unsupported model format")
            if checkpoint.get("park_id") != self.park.id:
                raise ValueError("This model belongs to a different park")
            excluded = [ride["name"] for ride in checkpoint["rides"] if is_excluded(ride)]
            dates = sorted(checkpoint.get("train_dates", []))
            result = {"ready": not excluded, "state": "needs_training" if excluded else "ready",
                      "run_name": checkpoint.get("run_name"),
                      "training_days": len(dates), "first_training_date": dates[0] if dates else None,
                      "last_training_date": dates[-1] if dates else None,
                      "excluded_attractions": excluded, "message": ""}
            if excluded:
                result["message"] = ("The saved model includes attractions now excluded from predictions: "
                                     + ", ".join(excluded[:3]) + (" and others" if len(excluded) > 3 else "")
                                     + ". Open Data & models and select Start training a new model to rebuild it. Your downloaded data is intact.")
            with self.lock:
                self.checkpoint_cache = (signature, result)
            return dict(result)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError):
            return {"ready": False, "state": "invalid", "training_days": 0,
                    "message": "This saved model cannot be used for this park. Open Data & models and select Start training a new model, or load another model. Recorded waits are available."}

    def dataset(self):
        if not self.settings.DATASET.exists():
            raise ValueError("No local dataset. Run python ml/data.py first, then reload this page.")
        dataset = load_dataset(self.settings.DATASET)
        if dataset.get("park_id") != self.park.id:
            raise ValueError("The cached dataset belongs to a different park")
        return dataset

    def days(self):
        dataset = self.dataset()
        return {**dataset.get("incomplete_days", {}), **dataset["days"]}

    def day(self, key):
        dataset = self.dataset()
        available = {**dataset.get("incomplete_days", {}), **dataset["days"]}
        if key not in available:
            raise ValueError("Choose a day in the local dataset")
        eligible = dataset.get("eligible_ride_ids")
        day = interpolate_day(modeling_day(available[key], set(eligible) if eligible is not None else None))
        expected = day.get("expected_timestamps", day["timestamps"])
        if not expected:
            raise ValueError("This day has no published schedule or observations")
        return day, expected

    def metadata(self):
        dataset = self.dataset() if self.settings.DATASET.exists() else {"days": {}}
        available = {**dataset.get("incomplete_days", {}), **dataset["days"]}
        dates = sorted(key for key, day in available.items() if day["timestamps"])
        days = [{"date": key, "observations": len(available[key]["timestamps"]),
                 "expected": len(available[key].get("expected_timestamps", available[key]["timestamps"])),
                 "complete": key in dataset["days"],
                 "reason": available[key].get("reason", "")} for key in dates]
        span = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days + 1 if dates else 0
        complete = sum(day["complete"] for day in days)
        checkpoint = self.checkpoint_status()
        return {
            "days": days,
            "coverage": {"first_date": dates[0] if dates else None,
                         "last_date": dates[-1] if dates else None,
                         "cached_days": len(dates), "complete_days": complete,
                         "partial_days": len(dates) - complete,
                         "span_days": span, "missing_days": span - len(dates),
                         "observations": sum(day["observations"] for day in days),
                         "updated_at": self.settings.DATASET.stat().st_mtime if self.settings.DATASET.exists() else None},
            "default_day": config.PREDICT_DAY if config.PREDICT_DAY in dates else (dates[-1] if dates else None),
            "default_start": config.PREDICT_START_TIME,
            "checkpoint_available": checkpoint["ready"],
            "checkpoint_status": checkpoint,
            "samples": self.settings.SAMPLES,
            "timezone": self.park.timezone,
            "park": self.park.metadata(),
            "checkpoint": self.settings.CHECKPOINT.name,
            "locations": self.locations,
            "location_warning": self.location_warning,
        }

    def day_payload(self, key):
        day, expected = self.day(key)
        # A continuous timeline also makes scheduled breaks explicit.
        timeline = list(range(expected[0], expected[-1] + TICK_SECONDS, TICK_SECONDS))
        if len(timeline) > 576:
            raise ValueError("An operating day cannot exceed 48 hours")
        interpolated = day.get("interpolated_ticks", {})
        snapshots = []
        for tick, row in zip(day["timestamps"], day["tokens"]):
            snapshots.append({"timestamp": tick,
                              "source": "interpolated" if str(tick) in interpolated else "observed",
                              "endpoints": interpolated.get(str(tick)),
                              "rides": [observed_ride(ride, token) for ride, token in zip(day["rides"], row)]})
        return {"date": key, "timestamps": timeline, "expected_timestamps": expected,
                "snapshots": snapshots, "rides": day["rides"], "reason": day.get("reason", ""),
                "latest_observation": day["timestamps"][-1] if day["timestamps"] else None}

    def start_forecast(self, key, start_time):
        day, expected = self.day(key)
        cutoff = park_timestamp(key, start_time, expected[0], self.park.tz)
        if cutoff <= expected[0] or cutoff > expected[-1] + TICK_SECONDS:
            raise ValueError("Start predictions after opening and no later than closing")
        checkpoint_path = self.settings.CHECKPOINT
        if not checkpoint_path.exists():
            raise ValueError("No trained checkpoint. Run python ml/train.py first. Recorded waits remain available.")
        checkpoint = self.checkpoint_status()
        if not checkpoint["ready"]:
            raise ValueError(checkpoint["message"])
        # Compute the whole remaining timeline once so scrubbing does not rerun inference.
        steps = max(1, (expected[-1] - cutoff) // TICK_SECONDS + 1)
        with self.lock:
            if self.settings.CHECKPOINT != checkpoint_path or not checkpoint_path.is_file():
                raise ValueError("The selected model changed. Select a saved model and generate the forecast again.")
            cache_key = (key, cutoff, self.settings.DATASET.stat().st_mtime_ns,
                         str(checkpoint_path.resolve()), checkpoint_path.stat().st_mtime_ns)
            for job_id, (saved_key, future) in self.jobs.items():
                if saved_key == cache_key:
                    return {"job": job_id}
            if sum(not future.done() for _, future in self.jobs.values()) >= 2:
                raise ValueError("A forecast is already queued. Wait for it to finish before starting another.")
            # Normalize to the selected five-minute bucket, just like run.py.
            start = datetime.fromtimestamp(cutoff, self.park.tz).strftime("%H:%M")
            settings = SimpleNamespace(**{**vars(self.settings), "HISTORY": None, "PREDICT_DAY": key,
                                          "PREDICT_START_TIME": start, "STEPS": steps})
            self.next_job += 1
            job_id = str(self.next_job)
            self.jobs[job_id] = (cache_key, self.executor.submit(self.forecast, settings))
            for old_id in list(self.jobs):
                if len(self.jobs) <= 6:
                    break
                if self.jobs[old_id][1].done():
                    del self.jobs[old_id]
            return {"job": job_id}

    @staticmethod
    def forecast(settings):
        result = predict(settings, print_output=False)
        snapshots = []
        for snapshot in result["snapshots"]:
            rides = []
            for ride in snapshot["rides"]:
                wait = ride.get("wait_minutes")
                status = ride.get("status") or ("OPERATING" if wait else "DOWN")
                rides.append({"id": ride["id"], "name": ride["name"], "status": status,
                              "wait": wait["p50"] if wait else None,
                              "p10": wait["p10"] if wait else None,
                              "p90": wait["p90"] if wait else None,
                              "down_probability": ride["down_probability"],
                              "closed_probability": ride.get("closed_probability", 0)})
            snapshots.append({"timestamp": int(datetime.fromisoformat(snapshot["time"]).timestamp()),
                              "source": "predicted", "forecast_method": result.get("forecast_method"), "rides": rides})
        return {"snapshots": snapshots, "as_of": result["as_of"],
                "prediction_start": result["prediction_start"], "samples": result["samples"],
                "forecast_method": result.get("forecast_method")}

    def job_payload(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("This forecast has expired. Generate it again.")
            future = self.jobs[job_id][1]
        if not future.done():
            return {"state": "running" if future.running() else "queued"}
        return {"state": "complete", **future.result()}

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class Handler(BaseHTTPRequestHandler):
    def allowed(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.send_error(403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            self.send_error(403)
            return False
        return True

    def reply(self, status, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def error_reply(self, error):
        if isinstance(error, (ValueError, FileNotFoundError)):
            self.reply(400, {"error": str(error)})
        else:
            logger.exception("Visualizer request failed")
            self.reply(500, {"error": f"The request failed: {error}"})

    def do_GET(self):
        if not self.allowed():
            return
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        try:
            if parts.path == "/api/parks":
                self.reply(200, {"parks": [park.metadata() for park in PARKS.values()],
                                 "default": getattr(self.server, "default_park", "disneyland")})
            elif parts.path == "/api/task":
                self.reply(200, self.server.lab.status(query.get("park", [self.server.default_park])[0]))
            elif parts.path == "/api/models":
                percentages = [float(query[key][0]) for key in ("train", "validation", "test")] if all(key in query for key in ("train", "validation", "test")) else None
                self.reply(200, self.server.lab.catalog(query.get("park", [self.server.default_park])[0], percentages))
            elif parts.path == "/api/ios-models":
                self.reply(200, self.server.lab.ios_models())
            elif parts.path == "/api/test-report":
                self.reply(200, self.server.lab.report(query.get("park", [self.server.default_park])[0], query.get("model", [""])[0]))
            elif parts.path == "/api/task-log":
                self.reply(200, self.server.lab.log(query.get("park", [self.server.default_park])[0]), "text/plain; charset=utf-8")
            elif parts.path == "/api/meta":
                self.reply(200, self.app_for(query).metadata())
            elif parts.path == "/api/day":
                self.reply(200, self.app_for(query).day_payload(query.get("date", [""])[0]))
            elif parts.path == "/api/forecast":
                self.reply(200, self.app_for(query).job_payload(query.get("job", [""])[0]))
            else:
                files = {"/": ("index.html", "text/html; charset=utf-8"),
                         "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                         "/style.css": ("style.css", "text/css; charset=utf-8")}
                if parts.path not in files:
                    self.send_error(404)
                    return
                name, content_type = files[parts.path]
                self.reply(200, (ASSETS / name).read_bytes(), content_type)
        except Exception as error:
            self.error_reply(error)

    def app_for(self, query):
        if getattr(self.server, "lab", None):
            return self.server.lab.app(query.get("park", [self.server.default_park])[0])
        return self.server.app

    def do_POST(self):
        if not self.allowed():
            return
        if urlsplit(self.path).path not in ("/api/forecast", "/api/action"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid forecast request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if urlsplit(self.path).path == "/api/action":
                self.reply(202, self.server.lab.action(payload))
                return
            if not all(isinstance(payload.get(k), str) for k in ("date", "start")):
                raise ValueError("Choose a day and prediction start time")
            self.reply(202, self.app_for(parse_qs(urlsplit(self.path).query)).start_forecast(payload["date"], payload["start"]))
        except Exception as error:
            self.error_reply(error)

    def log_message(self, message, *args):
        logger.info(message, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--park", choices=PARKS, default=config.PARK)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=config.DEVICE)
    parser.add_argument("--samples", type=int, default=config.SAMPLES)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--refresh-locations", action="store_true", help="Refresh cached public attraction coordinates")
    args = parser.parse_args()
    if not 1 <= args.samples <= 1000:
        parser.error("--samples must be between 1 and 1000")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from ml.lab import ParkLab
    lab = ParkLab(device=args.device, samples=args.samples, refresh_locations=args.refresh_locations)
    app = lab.app(args.park)
    if args.dataset:
        app.settings.DATASET = args.dataset.resolve()
    if args.checkpoint:
        app.settings.CHECKPOINT = args.checkpoint.resolve()
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
            server.app = app
            server.lab = lab
            server.default_park = args.park
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"OmniQueue wait-time map: {url}\nPress Ctrl+C to stop.", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the visualizer.", flush=True)
    finally:
        lab.close()


if __name__ == "__main__":
    main()
