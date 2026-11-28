"""Serving for the iOS app. Collection and training run independently."""

import logging
import os
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import time as epoch

import httpx

from api.collect import fetch_park_data
from ml.attractions import modeling_day
from ml.data import MISSING, OVERFLOW, TICK_SECONDS, collect_day, expected_ticks, interpolate_day
from ml.forecast_store import digest, encode
from ml.parks import PARKS, settings_for
from ml.visualize import observed_ride
from view import Archive


class ForecastUnavailable(ValueError):
    pass


class ForecastBusy(RuntimeError):
    pass


def legal_starts(day, ride_ids):
    day = interpolate_day(modeling_day(day, set(ride_ids)))
    if {ride["id"] for ride in day["rides"]} != set(ride_ids):
        return []
    ticks = day.get("expected_timestamps", day["timestamps"])
    if not ticks:
        return []
    rows = dict(zip(day["timestamps"], day["tokens"]))
    endpoints = day.get("interpolated_ticks", {})
    starts = []
    latest_required_endpoint = 0
    for tick in ticks:
        if tick - TICK_SECONDS in rows and latest_required_endpoint < tick:
            starts.append(tick)
        tokens = rows.get(tick)
        if tokens is None or MISSING in tokens:
            break
        if str(tick) in endpoints:
            latest_required_endpoint = max(latest_required_endpoint, endpoints[str(tick)][1])
    return starts


def display_day(day, eligible):
    display = interpolate_day(modeling_day(day, eligible))
    expected = display.get("expected_timestamps", display["timestamps"])
    timeline = list(range(expected[0], expected[-1] + TICK_SECONDS, TICK_SECONDS)) if expected else []
    if len(timeline) > 576:
        raise ValueError("An operating day cannot exceed 48 hours.")
    interpolated = display.get("interpolated_ticks", {})
    snapshots = []
    for tick, row in zip(display["timestamps"], display["tokens"]):
        snapshots.append({"timestamp": tick, "source": "interpolated" if str(tick) in interpolated else "observed",
                          "rides": [observed_ride(ride, token) for ride, token in zip(display["rides"], row)]})
    return {
        "timeline": timeline, "scheduled": expected,
        "snapshots": snapshots, "latest_observation": max(
            (frame["timestamp"] for frame in snapshots if frame["source"] == "observed"), default=None),
    }


class ForecastService:
    def __init__(self, store):
        self.store = store
        self.client = httpx.Client(timeout=6, headers={"User-Agent": "OmniQueue/1.0"})
        self.archives = {park.slug: Archive(store.db, park.id) for park in PARKS.values()}
        self.day_locks = {slug: Lock() for slug in PARKS}
        self.day_cache = {}
        self.inference = Lock()
        self.workspace = tempfile.TemporaryDirectory(prefix="omniqueue-forecast-")

    @lru_cache(maxsize=40)
    def schedule(self, park, month, window):
        return fetch_park_data(self.client, f"schedule/{month}", "schedule", park)

    def ticks(self, park, key):
        return expected_ticks(key, self.schedule(park, key.strftime("%Y/%m"), int(epoch()) // 3600), park.tz)

    def today(self, park, now):
        today = now.astimezone(park.tz).date()
        if now.astimezone(park.tz).hour < 6:
            previous = today - timedelta(days=1)
            ticks, close = self.ticks(park, previous)
            if ticks and ticks[0] <= now.timestamp() < close:
                return previous
        return today

    @lru_cache(maxsize=30)
    def locations(self, park, window):
        try:
            data = fetch_park_data(self.client, "children", "children", park)
            return {row["id"]: {"name": row["name"], **row["location"]}
                    for row in data if row.get("entityType") == "ATTRACTION" and row.get("location")
                    and isinstance(row["location"].get("latitude"), (float, int))
                    and isinstance(row["location"].get("longitude"), (float, int))
                    and abs(row["location"]["latitude"] - park.latitude) < .06
                    and abs(row["location"]["longitude"] - park.longitude) < .06}
        except (httpx.HTTPError, KeyError, ValueError):
            logging.warning("Attraction coordinates unavailable for %s", park.slug)
            return {}

    @lru_cache(maxsize=40)
    def model(self, slug, window):
        return self.store.active_model(slug)

    def recorded_day(self, park, key, *, recheck=False):
        ticks, close = self.ticks(park, key)
        if not ticks:
            return {"rides": [], "tokens": [], "timestamps": [], "expected_timestamps": [],
                    "closing_at": 0, "source": "firestore", "timezone": park.timezone}
        opening = datetime.fromtimestamp(ticks[0], timezone.utc)
        stop = datetime.fromtimestamp(ticks[-1] + TICK_SECONDS, timezone.utc)
        with self.day_locks[park.slug]:
            cache = self.day_cache.get(park.slug)
            if not recheck and cache and cache[0] == key and cache[1] == ticks and epoch() - cache[2] < 12:
                return cache[4]
            archive = self.archives[park.slug]
            query = archive.between(f"{park.id}_{opening:%Y%m%dT%H%MZ}", f"{park.id}_{stop:%Y%m%dT%H%MZ}")
            newest = next(iter(query.select(["sampled_at"]).order_by("__name__", direction="DESCENDING")
                               .limit(1).stream(timeout=5)), None)
            revision = newest.id if newest else None
            if (cache and cache[0] == key and cache[1] == ticks and cache[3] == revision
                    and epoch() - cache[5] < 300):
                day, read_at = cache[4], cache[5]
            else:
                snapshots = [doc.to_dict() for doc in query.stream(timeout=10)] if newest else []
                day = collect_day(ticks, snapshots, archive.catalog)
                day.update(timezone=park.timezone, source="firestore", closing_at=close)
                read_at = epoch()
            self.day_cache[park.slug] = (key, ticks, epoch(), revision, day, read_at)
            return day

    @lru_cache(maxsize=16)
    def archived_day(self, path):
        return self.store.read_json(path)

    def day(self, park, requested, now, *, recheck=False):
        today = self.today(park, now)
        key = requested or today
        if key > today:
            raise ForecastUnavailable("Choose today or an earlier operating day.")
        if key == today:
            return key, today, self.recorded_day(park, key, recheck=recheck)
        record = self.store.public_day(park.slug, key.isoformat())
        collected = record and record.get("source") == "firestore"
        day = self.archived_day(record["path"]) if collected else None
        if day is not None and day.get("source") != "firestore":
            day = None
        if day is None or not record.get("complete"):
            live = self.recorded_day(park, key, recheck=recheck)
            if day is None or len(live["timestamps"]) >= len(day["timestamps"]):
                day = live
        if not day["timestamps"]:
            raise ForecastUnavailable("No recorded waits are available for that date.")
        return key, today, day

    def dates(self, park, before, now, limit):
        today = self.today(park, now)
        upper = min(before, today) if before is not None else today
        seed = self.store.public_dates(park.slug, upper.isoformat(), limit)
        rows = {item["date"]: item for item in seed}
        for key in self.collector_dates(park, upper, limit, int(epoch()) // 300):
            rows.setdefault(key, {"date": key, "complete": None})
        return [rows[key] for key in sorted(rows, reverse=True)[:limit]]

    @lru_cache(maxsize=40)
    def collector_dates(self, park, before, limit, window):
        archive = self.archives[park.slug]
        dates = []
        for _ in range(limit):
            if before == date.min:
                break
            boundary = datetime.combine(before, time(), park.tz).astimezone(timezone.utc)
            _, close = self.ticks(park, before - timedelta(days=1))
            if close > boundary.timestamp():
                exclusive_bucket = (close + TICK_SECONDS - 1) // TICK_SECONDS * TICK_SECONDS
                boundary = datetime.fromtimestamp(exclusive_bucket, timezone.utc)
            sampled = archive.latest_before(boundary)
            if sampled is None:
                break
            key = self.today(park, sampled)
            if key >= before:
                break
            dates.append(key.isoformat())
            before = key
        return dates

    def day_payload(self, park, requested, now):
        key, today, day = self.day(park, requested, now, recheck=True)
        model = self.model(park.slug, int(epoch()) // 15)
        locations = self.locations(park, int(epoch()) // 21600)
        eligible = self.eligible_rides(day, model)
        display = display_day(day, eligible)
        starts = legal_starts(day, model["ride_ids"]) if model else []
        default_start = self.default_start(day, starts, key == today, now)
        return {**display, "date": key.isoformat(), "is_today": key == today,
                "forecast_starts": starts, "default_start": default_start, "model_available": bool(model),
                "model_version": model["version"] if model else None,
                "rides": [{**ride, "coordinate": {field: locations[ride["id"]][field]
                                                  for field in ("latitude", "longitude")}
                           if ride["id"] in locations else None}
                          for ride in day["rides"] if ride["id"] in eligible],
                "state": ("closed" if not display["scheduled"] else "waiting" if not display["snapshots"]
                          else "recorded")}

    @staticmethod
    def eligible_rides(day, model):
        # Keep modeled rides visible during closures; include additional attractions
        # once the collector has seen a numeric standby wait for them.
        ids = set(model["ride_ids"]) if model else set()
        ids.update(ride["id"] for index, ride in enumerate(day["rides"])
                   if any(0 <= row[index] <= OVERFLOW for row in day["tokens"]))
        return {ride["id"] for ride in modeling_day(day, ids)["rides"]}

    @staticmethod
    def default_start(day, starts, is_today, now):
        if not starts:
            return None
        if is_today:
            if now.timestamp() >= day.get("closing_at", float("inf")):
                return None
            actual = [tick for tick in day["timestamps"] if str(tick) not in day.get("interpolated_ticks", {})]
            candidate = actual[-1] + TICK_SECONDS if actual else None
            return candidate if candidate in starts else None
        # Replay near an hour after opening by default, with earlier choices available.
        return min(starts, key=lambda tick: abs(tick - day["expected_timestamps"][0] - 3600))

    def checkpoint(self, park, path, sha256):
        # The inference lock protects these per-park scratch files while PyTorch reads them.
        folder = Path(self.workspace.name) / park.slug
        local = folder / f"{sha256}.pt"
        if local.is_file():
            return local
        data = self.store.read(path)
        if data is None or digest(data) != sha256:
            raise ForecastUnavailable("The published model is temporarily unavailable.")
        folder.mkdir(exist_ok=True)
        pending = local.with_suffix(".tmp")
        pending.write_bytes(data)
        pending.replace(local)
        for old in folder.glob("*.pt"):
            if old != local:
                old.unlink()
        return local

    def forecast(self, park, requested, cutoff, now):
        key, today, day = self.day(park, requested, now)
        model = self.model(park.slug, int(epoch()) // 15)
        starts = legal_starts(day, model["ride_ids"]) if model else []
        default = self.default_start(day, starts, key == today, now)
        if cutoff is None and key != today:
            issued = self.store.first_issued(park.slug, key)
            if issued is not None:
                return issued
        cutoff = default if cutoff is None else cutoff
        if cutoff is None:
            raise ForecastUnavailable("There is no forecast start available for this operating day.")
        if cutoff not in day.get("expected_timestamps", []):
            raise ForecastUnavailable("Choose a five-minute forecast start during park hours.")
        issued_path = f"issued/{park.slug}/{key}/{cutoff}.json.gz"
        is_live = key == today and cutoff == default
        if not is_live:
            saved = self.store.read_json(issued_path)
            if saved is not None:
                return saved
        if not model or cutoff not in starts:
            raise ForecastUnavailable("This start time needs enough recorded waits and a published park model.")
        revision = digest(encode(day))
        path = f"forecasts/{park.slug}/{key}/{cutoff}/{model['version']}/{revision}.json.gz"
        saved = self.store.read_json(path)
        if saved is not None:
            return self.finish_forecast(saved, is_live, issued_path)
        if not self.inference.acquire(blocking=False):
            raise ForecastBusy("Another forecast is running. Try again shortly.")
        try:
            saved = self.store.read_json(path)
            if saved is None:
                from ml.run import predict
                settings = settings_for(park, migrate=False, PREDICT_DAY=key.isoformat(),
                                        DEVICE=os.environ.get("OMNIQUEUE_FORECAST_DEVICE", "cpu"),
                                        SAMPLES=model["samples"], SEED=model["seed"], TEMPERATURE=model["temperature"],
                                        CHECKPOINT=self.checkpoint(park, model["path"], model["sha256"]))
                result = predict(settings, print_output=False, day_override=day, start_timestamp=cutoff)
                snapshots = []
                for frame in result["snapshots"]:
                    rides = []
                    for ride in frame["rides"]:
                        wait = ride["wait_minutes"] or {}
                        rides.append({"id": ride["id"], "name": ride["name"],
                                      "status": ride.get("status", "OPERATING" if wait else "DOWN"),
                                      "wait": wait.get("p50"), "p10": wait.get("p10"), "p90": wait.get("p90")})
                    snapshots.append({"timestamp": int(datetime.fromisoformat(frame["time"]).timestamp()),
                                      "rides": rides})
                split = next((label for label in ("train", "validation", "test")
                              if key.isoformat() in model[f"{label}_dates"]), None)
                saved = {"snapshots": snapshots, "start": cutoff, "as_of": result["as_of"],
                         "model_name": model["name"], "model_version": model["version"], "split": split,
                         "date": key.isoformat()}
                self.store.write_json(path, saved)
            return self.finish_forecast(saved, is_live, issued_path)
        finally:
            self.inference.release()

    def finish_forecast(self, result, live, issued_path):
        if not live:
            return {**result, "kind": "replay"}
        issued = {**result, "kind": "live", "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.store.write_json(issued_path, issued)
        return issued
