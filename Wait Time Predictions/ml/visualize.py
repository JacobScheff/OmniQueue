"""Explore cached Disneyland waits and model forecasts on a local interactive map.

Run ``python ml/visualize.py`` after exporting data and training a checkpoint.
The browser provides day/start controls, a five-minute timeline, and map overlays.
"""

import argparse
import json
import logging
import math
import sys
import threading
import webbrowser
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

# Allow direct execution from any working directory, like the other ML scripts.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.attractions import modeling_day
from ml.data import (
    CLOSED, DOWN, MISSING, OVERFLOW, PARK_TIMEZONE, REFURBISHMENT,
    TICK_SECONDS, UNKNOWN, WAIT_STEP, interpolate_day, load_dataset,
)
from ml.run import park_timestamp, predict

ROOT = config.ROOT
ASSETS = Path(__file__).with_name("visualizer")
PARK_ID = "7340550b-c14d-4def-80bb-acdb51d49a66"
LOCATIONS_URL = f"https://api.themeparks.wiki/v1/entity/{PARK_ID}/children"
logger = logging.getLogger(__name__)


def load_locations(path, refresh=False):
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

        response = httpx.get(LOCATIONS_URL, timeout=15,
                             headers={"User-Agent": "OmniQueueVisualizer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if payload.get("id") != PARK_ID:
            raise ValueError("The location catalog belongs to a different park")
        locations = {}
        for entity in payload["children"]:
            point = entity.get("location") or {}
            lat, lon = point.get("latitude"), point.get("longitude")
            if (entity.get("entityType") == "ATTRACTION"
                    and isinstance(lat, (int, float)) and isinstance(lon, (int, float))
                    and math.isfinite(lat) and math.isfinite(lon)
                    and 33.79 < lat < 33.84 and -117.94 < lon < -117.89):
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
        self.locations = locations or {}
        self.location_warning = location_warning
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast")
        self.jobs = OrderedDict()
        self.next_job = 0

    def dataset(self):
        if not self.settings.DATASET.exists():
            raise ValueError("No local dataset. Run python ml/data.py first, then reload this page.")
        dataset = load_dataset(self.settings.DATASET)
        if dataset.get("park_id") != PARK_ID:
            raise ValueError("The cached dataset belongs to a different park")
        return dataset

    def days(self):
        dataset = self.dataset()
        return {**dataset.get("incomplete_days", {}), **dataset["days"]}

    def day(self, key):
        available = self.days()
        if key not in available:
            raise ValueError("Choose a day in the local dataset")
        day = interpolate_day(modeling_day(available[key]))
        expected = day.get("expected_timestamps", day["timestamps"])
        if not expected:
            raise ValueError("This day has no published schedule or observations")
        return day, expected

    def metadata(self):
        available = self.days()
        dates = sorted(available)
        return {
            "days": [{"date": key, "observations": len(available[key]["timestamps"]),
                      "expected": len(available[key].get("expected_timestamps", available[key]["timestamps"])),
                      "reason": available[key].get("reason", "")} for key in dates],
            "default_day": config.PREDICT_DAY if config.PREDICT_DAY in available else (dates[-1] if dates else None),
            "default_start": config.PREDICT_START_TIME,
            "checkpoint_available": self.settings.CHECKPOINT.exists(),
            "samples": self.settings.SAMPLES,
            "timezone": str(PARK_TIMEZONE),
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
        cutoff = park_timestamp(key, start_time, expected[0])
        if cutoff <= expected[0] or cutoff > expected[-1] + TICK_SECONDS:
            raise ValueError("Start predictions after opening and no later than closing")
        if not self.settings.CHECKPOINT.exists():
            raise ValueError("No trained checkpoint. Run python ml/train.py first. Recorded waits remain available.")
        # Compute the whole remaining timeline once so scrubbing does not rerun inference.
        steps = max(1, (expected[-1] - cutoff) // TICK_SECONDS + 1)
        cache_key = (key, cutoff, self.settings.DATASET.stat().st_mtime_ns,
                     self.settings.CHECKPOINT.stat().st_mtime_ns)
        with self.lock:
            for job_id, (saved_key, future) in self.jobs.items():
                if saved_key == cache_key:
                    return {"job": job_id}
            if sum(not future.done() for _, future in self.jobs.values()) >= 2:
                raise ValueError("A forecast is already queued. Wait for it to finish before starting another.")
            # Normalize to the selected five-minute bucket, just like run.py.
            start = datetime.fromtimestamp(cutoff, PARK_TIMEZONE).strftime("%H:%M")
            settings = SimpleNamespace(**vars(self.settings), HISTORY=None, PREDICT_DAY=key,
                                       PREDICT_START_TIME=start, STEPS=steps)
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
                              "down_probability": ride["down_probability"]})
            snapshots.append({"timestamp": int(datetime.fromisoformat(snapshot["time"]).timestamp()),
                              "source": "predicted", "rides": rides})
        return {"snapshots": snapshots, "as_of": result["as_of"],
                "prediction_start": result["prediction_start"], "samples": result["samples"]}

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
            if parts.path == "/api/meta":
                self.reply(200, self.server.app.metadata())
            elif parts.path == "/api/day":
                self.reply(200, self.server.app.day_payload(query.get("date", [""])[0]))
            elif parts.path == "/api/forecast":
                self.reply(200, self.server.app.job_payload(query.get("job", [""])[0]))
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

    def do_POST(self):
        if not self.allowed():
            return
        if self.path != "/api/forecast":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid forecast request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not all(isinstance(payload.get(k), str) for k in ("date", "start")):
                raise ValueError("Choose a day and prediction start time")
            self.reply(202, self.server.app.start_forecast(payload["date"], payload["start"]))
        except Exception as error:
            self.error_reply(error)

    def log_message(self, message, *args):
        logger.info(message, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET)
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=config.DEVICE)
    parser.add_argument("--samples", type=int, default=config.SAMPLES)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--refresh-locations", action="store_true", help="Refresh cached public attraction coordinates")
    args = parser.parse_args()
    if not 1 <= args.samples <= 1000:
        parser.error("--samples must be between 1 and 1000")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    settings = SimpleNamespace(DATASET=args.dataset.resolve(), CHECKPOINT=args.checkpoint.resolve(),
                               DEVICE=args.device, SAMPLES=args.samples,
                               SEED=config.SEED, TEMPERATURE=config.TEMPERATURE)
    locations, warning = load_locations(ROOT / "data" / "attraction_locations.json", args.refresh_locations)
    app = Visualizer(settings, locations, warning)
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
            server.app = app
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"OmniQueue wait-time map: {url}\nPress Ctrl+C to stop.", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the visualizer.", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
