"""Dataset storage, token encoding, and Firestore export for training and forecasts."""

import gzip
import json
import math
import os
import shutil
import sys
import tempfile
from bisect import bisect_left
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from zoneinfo import ZoneInfo

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config

TICK_SECONDS = 300
MISSING = -1
PARK_TIMEZONE = ZoneInfo("America/Los_Angeles")
WAIT_STEP = 5
MAX_WAIT = 300
OVERFLOW = MAX_WAIT // WAIT_STEP + 1
DOWN, CLOSED, REFURBISHMENT, UNKNOWN = range(OVERFLOW + 1, OVERFLOW + 5)
VOCAB_SIZE = UNKNOWN + 1
TOKEN_LABELS = [str(wait) for wait in range(0, MAX_WAIT + 1, WAIT_STEP)] + [
    f"{MAX_WAIT}+", "DOWN", "CLOSED", "REFURBISHMENT", "UNKNOWN",
]


def ride_token(ride):
    if ride["status"] == "OPERATING" and ride["wait"] is not None:
        wait = ride["wait"]
        if type(wait) is not int or not 0 <= wait < 65535:
            raise ValueError("An operating wait must be an integer from 0 through 65534 minutes")
        if wait == 13:
            wait = 5
        if wait > MAX_WAIT:
            return OVERFLOW
        return min(MAX_WAIT // WAIT_STEP, (wait + WAIT_STEP // 2) // WAIT_STEP)
    return {"DOWN": DOWN, "CLOSED": CLOSED, "REFURBISHMENT": REFURBISHMENT}.get(
        ride["status"], UNKNOWN,
    )


def load_dataset(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        dataset = json.load(source)
    if dataset.get("version") != 1:
        raise ValueError("Unsupported dataset version")
    return dataset


def save_dataset(path, dataset):
    """Replace atomically so an interrupted export preserves the previous file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as target:
            json.dump(dataset, target, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def time_features(timestamps, park_timezone=PARK_TIMEZONE):
    """Calendar features available for observed and future ticks, in park time."""
    rows = []
    for timestamp in timestamps:
        local = datetime.fromtimestamp(timestamp, park_timezone)
        cycles = (local.hour * 60 + local.minute, local.weekday(), local.timetuple().tm_yday - 1)
        rows.append([
            value
            for cycle, period in zip(cycles, (1440, 7, 365.25))
            for value in (math.sin(2 * math.pi * cycle / period), math.cos(2 * math.pi * cycle / period))
        ])
    return rows


def align_day(day, rides):
    """Catalog order can change. The checkpoint's attraction IDs define the slots."""
    import numpy as np
    import torch

    positions = {ride["id"]: index for index, ride in enumerate(day["rides"])}
    source = np.asarray(day["tokens"], dtype=np.int64)
    tokens = np.full((len(day["timestamps"]), len(rides)), UNKNOWN, dtype=np.int64)
    for column, ride in enumerate(rides):
        if ride["id"] in positions:
            tokens[:, column] = source[:, positions[ride["id"]]]
    return torch.from_numpy(tokens)


def history_from_json(path):
    """Accept the archive viewer's {snapshots: [{time, rides}]} JSON format."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshots = payload["snapshots"]
    if not snapshots:
        raise ValueError("History contains no snapshots")
    names = {ride["id"]: ride["name"] for snapshot in snapshots for ride in snapshot["rides"]}
    ride_ids = sorted(names)
    timestamps, tokens = [], []
    for snapshot in snapshots:
        moment = datetime.fromisoformat(snapshot["time"])
        if moment.tzinfo is None:
            raise ValueError("History timestamps must include a timezone")
        timestamps.append(int(moment.timestamp()) // TICK_SECONDS * TICK_SECONDS)
        by_id = {ride["id"]: ride_token(ride) for ride in snapshot["rides"]}
        tokens.append([by_id.get(ride_id, MISSING) for ride_id in ride_ids])
    result = {"timestamps": timestamps, "tokens": tokens,
              "rides": [{"id": ride_id, "name": names[ride_id]} for ride_id in ride_ids]}
    if "expected_timestamps" in payload:
        result["expected_timestamps"] = payload["expected_timestamps"]
    return result


def expected_ticks(day, schedule, park_timezone=PARK_TIMEZONE):
    """An operating day includes sessions opening that date, even past midnight."""
    from api.collect import OPERATING_HOURS

    ticks = set()
    closing = 0
    for session in schedule:
        if session["type"] not in OPERATING_HOURS:
            continue
        opens = datetime.fromisoformat(session["openingTime"])
        closes = datetime.fromisoformat(session["closingTime"])
        if opens.tzinfo is None or closes.tzinfo is None or closes <= opens:
            raise ValueError("Invalid scheduled operating hours")
        if opens.astimezone(park_timezone).date() != day:
            continue
        start, end = int(opens.timestamp()), int(closes.timestamp())
        closing = max(closing, end)
        ticks.update(range(start // TICK_SECONDS * TICK_SECONDS, end, TICK_SECONDS))
    return sorted(ticks), closing


def collect_day(ticks, snapshots, catalog):
    """Keep usable observations, including incomplete snapshots and days."""
    from view import decode

    rows = {}
    names = {}
    seen = set()
    expected = set(ticks)
    for snapshot in sorted(snapshots, key=lambda row: row["sampled_at"]):
        timestamp = int(snapshot["sampled_at"].timestamp()) // TICK_SECONDS * TICK_SECONDS
        if expected and timestamp not in expected:
            continue
        if timestamp in seen:
            # Conflicting observations cannot establish this bucket's history.
            rows.pop(timestamp, None)
            continue
        seen.add(timestamp)
        try:
            rides = decode(snapshot, catalog(snapshot["catalog_version"]))
            by_id = {ride["id"]: ride_token(ride) for ride in rides}
            if not by_id or len(by_id) != len(rides):
                raise ValueError("Empty or duplicate attraction catalog")
        except ValueError:
            continue
        rows[timestamp] = by_id
        names.update({ride["id"]: ride["name"] for ride in rides})
    ride_ids = sorted(names)
    timestamps = sorted(rows)
    return interpolate_day({
        "timestamps": timestamps,
        "expected_timestamps": ticks,
        "rides": [{"id": ride_id, "name": names[ride_id]} for ride_id in ride_ids],
        "tokens": [[rows[tick].get(ride_id, MISSING) for ride_id in ride_ids] for tick in timestamps],
    })


def interpolate_day(day):
    """Fill up to four missing scheduled ticks using their observed neighbors."""
    timestamps = day.get("timestamps", [])
    if not timestamps or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        return day  # Preserve invalid history for the caller's validation.
    expected = set(day.get("expected_timestamps", timestamps))
    # The unobserved tail of an in-progress day is not a collection gap.
    # Count earlier repairs too, so repeated refreshes keep the daily limit.
    missing = sorted(tick for tick in expected - set(timestamps) if tick <= timestamps[-1])
    interpolated = dict(day.get("interpolated_ticks", {}))
    if not missing or len(missing) + len(interpolated) >= 5:
        return day
    rows = dict(zip(timestamps, day["tokens"]))
    for tick in missing:
        index = bisect_left(timestamps, tick)
        if index == 0 or index == len(timestamps):
            continue  # Interpolation needs both endpoints; never extrapolate.
        left, right = timestamps[index - 1:index + 1]
        if any(point not in expected for point in range(left, right + TICK_SECONDS, TICK_SECONDS)):
            continue  # Do not bridge a scheduled break.
        span, offset = right - left, tick - left
        tokens = []
        for start, end in zip(rows[left], rows[right]):
            if MISSING in (start, end):
                token = MISSING
            elif 0 <= start < OVERFLOW and 0 <= end < OVERFLOW:
                # Tokens are five-minute waits. Round half up to a whole token.
                weighted = start * (span - offset) + end * offset
                token = (2 * weighted + span) // (2 * span)
            else:
                # Status and overflow tokens are categories, not numeric waits.
                token = start if 2 * offset < span else end
            tokens.append(token)
        rows[tick] = tokens
        interpolated[str(tick)] = [left, right]
    if len(rows) == len(timestamps):
        return day
    ordered = sorted(rows)
    return {**day, "timestamps": ordered, "tokens": [rows[tick] for tick in ordered],
            "interpolated_ticks": interpolated}


def validate_day(day, before=None):
    """Require every scheduled bucket and ride, optionally through a cutoff."""
    local_zone = ZoneInfo(day["timezone"]) if day.get("timezone") else PARK_TIMEZONE
    # Legacy complete days were already checked by the old exporter.
    expected = day.get("expected_timestamps", day["timestamps"])
    if not expected:
        raise ValueError("No published operating schedule is available for this day")
    required = [tick for tick in expected if before is None or tick < before]
    missing = sorted(set(required) - set(day["timestamps"]))
    if before is not None:
        # Replays cannot use an interpolation's future endpoint as history.
        missing = sorted(set(missing) | {
            int(tick) for tick, (_, right) in day.get("interpolated_ticks", {}).items()
            if int(tick) < before <= right
        })
    if missing:
        first = datetime.fromtimestamp(missing[0], local_zone)
        raise ValueError(f"Missing {len(missing)} of {len(required)} scheduled snapshots. First missing {first:%Y-%m-%d %H:%M %Z}")
    for tick, tokens in zip(day["timestamps"], day["tokens"]):
        if (before is None or tick < before) and any(token == MISSING for token in tokens):
            moment = datetime.fromtimestamp(tick, local_zone)
            raise ValueError(f"Attractions missing from snapshot at {moment:%Y-%m-%d %H:%M %Z}")


def complete_day(ticks, snapshots, catalog):
    day = collect_day(ticks, snapshots, catalog)
    validate_day(day)
    return day


def needs_refresh(day, now):
    """Refresh unfinished days once per bucket, including a final closing read."""
    checked = day.get("checked_at", 0)
    closing = day.get("closing_at", 0)
    return checked < closing and int(checked) // TICK_SECONDS < int(now) // TICK_SECONDS


def export_firestore(settings=config):
    print("Preparing dataset export...", flush=True)
    from api.collect import firestore_client, fetch_park_data
    from ml.parks import get_park
    park = get_park(getattr(settings, "PARK", "disneyland"))
    PARK_ID, PARK_TIMEZONE = park.id, park.tz

    first_day = date.fromisoformat(settings.START_DATE)
    now = datetime.now(timezone.utc).timestamp()
    last_day = date.fromisoformat(settings.END_DATE) if settings.END_DATE else datetime.now(PARK_TIMEZONE).date()
    if settings.DATASET.exists():
        print(f"Loading local cache from {settings.DATASET}", flush=True)
    dataset = load_dataset(settings.DATASET) if settings.DATASET.exists() else {
        "version": 1, "park_id": PARK_ID, "days": {}, "incomplete_days": {}, "discarded": {}, "catalogs": {}, "schedules": {},
    }
    incomplete = dataset.setdefault("incomplete_days", {})
    if dataset["park_id"] != PARK_ID:
        raise ValueError("Dataset belongs to a different park")
    if first_day > last_day:
        raise ValueError("Start date must be on or before the end date")
    if dataset.get("firestore_start_date"):
        first_day = max(first_day, date.fromisoformat(dataset["firestore_start_date"]))
        if first_day > last_day:
            print("Requested dates predate this park's Firebase collection start. No Firestore reads.", flush=True)
            return
    days = [first_day + timedelta(days=offset) for offset in range((last_day - first_day).days + 1)]
    repaired_cache = False
    for day in days:
        key = day.isoformat()
        if key not in incomplete:
            continue
        original = incomplete[key]
        repaired = interpolate_day(original)
        if repaired is original:
            continue
        incomplete[key] = repaired
        repaired_cache = True
        try:
            validate_day(repaired)
        except ValueError as error:
            repaired["reason"] = str(error)
            continue
        closing = repaired.get("closing_at", 0)
        if closing and repaired.get("checked_at", 0) >= closing:
            repaired.pop("reason", None)
            dataset["days"][key] = incomplete.pop(key)
    if repaired_cache:
        save_dataset(settings.DATASET, dataset)
    pending = [day for day in days if settings.REFRESH or (
        day.isoformat() not in dataset["days"] and (
            day.isoformat() not in incomplete or needs_refresh(incomplete[day.isoformat()], now)
        )
    )]
    if not pending:
        print(f"All {len(days)} requested days are cached in {settings.DATASET}. No Firestore reads.", flush=True)
        return

    print(f"Checking {len(pending)} days, with {len(days) - len(pending)} already cached.", flush=True)
    started = monotonic()
    completed = 0
    last_width = 0

    def progress(stage):
        nonlocal last_width
        elapsed = monotonic() - started
        remaining = elapsed / completed * (len(pending) - completed) if completed else None
        eta = f"{int(remaining) // 60:02}:{int(remaining) % 60:02}" if remaining is not None else "--:--"
        filled = 12 * completed // len(pending)
        line = (f"[{'#' * filled}{'-' * (12 - filled)}] {completed}/{len(pending)} days "
                f"{int(elapsed) // 60:02}:{int(elapsed) % 60:02} elapsed ETA {eta} | {stage}")
        if sys.stderr.isatty():
            width = max(1, shutil.get_terminal_size().columns - 1)
            line = line[:width]
            print("\r" + line.ljust(min(last_width, width)), end="", file=sys.stderr, flush=True)
            last_width = len(line)
        else:
            print(line, file=sys.stderr, flush=True)

    def message(text):
        nonlocal last_width
        if sys.stderr.isatty() and last_width:
            print(file=sys.stderr, flush=True)
            last_width = 0
        print(text, flush=True)

    import httpx
    from google.api_core.exceptions import GoogleAPICallError
    from google.api_core.retry import Retry
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import firestore
    from view import Archive

    # Firestore's streaming iterator can restart a failed query repeatedly.
    # A false retry predicate bounds this attempt to its RPC timeout.
    read_retry = Retry(predicate=lambda error: False)
    progress("Finding Firestore credentials")
    if settings.CREDENTIALS:
        db = firestore_client(settings.CREDENTIALS.read_text(encoding="utf-8-sig"))
    elif os.environ.get("FIRESTORE_SERVICE_ACCOUNT_JSON"):
        db = firestore_client(os.environ["FIRESTORE_SERVICE_ACCOUNT_JSON"])
    else:
        db = firestore.Client(project=settings.PROJECT)
    archive = Archive(db, park_id=PARK_ID)

    def catalog(version):
        if version not in dataset["catalogs"]:
            progress(f"{key} reading catalog")
            dataset["catalogs"][version] = archive.catalog(version, retry=read_retry, timeout=30)
        return dataset["catalogs"][version]

    refreshed_months = set()
    try:
        with httpx.Client(timeout=30, headers={"User-Agent": "OmniQueueDataset/1.0"}) as client:
            for day in pending:
                key, month = day.isoformat(), day.strftime("%Y/%m")
                if month not in refreshed_months:
                    progress(f"Fetching schedule {month}")
                    dataset["schedules"][month] = fetch_park_data(client, f"schedule/{month}", "schedule", park)
                    refreshed_months.add(month)
                ticks, closing = expected_ticks(day, dataset["schedules"][month], PARK_TIMEZONE)
                if ticks:
                    start = datetime.fromtimestamp(ticks[0], timezone.utc)
                    stop = datetime.fromtimestamp(ticks[-1] + TICK_SECONDS, timezone.utc)
                else:
                    start = datetime.combine(day, datetime.min.time(), PARK_TIMEZONE).astimezone(timezone.utc)
                    stop = datetime.combine(day + timedelta(days=1), datetime.min.time(), PARK_TIMEZONE).astimezone(timezone.utc)
                documents = archive.between(f"{PARK_ID}_{start:%Y%m%dT%H%MZ}", f"{PARK_ID}_{stop:%Y%m%dT%H%MZ}")
                progress(f"{key} Firestore 0/{len(ticks)} snapshots")
                snapshots = []
                for document in documents.stream(timeout=30, retry=read_retry):
                    snapshots.append(document.to_dict())
                    if len(snapshots) % 25 == 0:
                        progress(f"{key} Firestore {len(snapshots)}/{len(ticks)} snapshots")
                progress(f"{key} checking {len(snapshots)}/{len(ticks)} snapshots")
                result = collect_day(ticks, snapshots, catalog)
                from ml.attractions import modeling_day
                result = modeling_day(result, set(dataset["eligible_ride_ids"]) if "eligible_ride_ids" in dataset else None)
                result["timezone"] = str(PARK_TIMEZONE)
                result["source"] = "firestore"
                previous = dataset["days"].get(key) or incomplete.get(key)
                if not result["timestamps"] and previous and previous["timestamps"]:
                    completed += 1
                    message(f"{key}: no Firestore observations; kept cached history")
                    continue
                result["checked_at"] = datetime.now(timezone.utc).timestamp()
                result["closing_at"] = closing or datetime.combine(day + timedelta(days=1), datetime.min.time(), PARK_TIMEZONE).timestamp()
                reason = ""
                try:
                    validate_day(result)
                    if closing > result["checked_at"]:
                        reason = "Operating day is still in progress"
                except ValueError as error:
                    reason = str(error)
                if reason and previous and (key in dataset["days"] or len(previous["timestamps"]) > len(result["timestamps"])):
                    completed += 1
                    message(f"{key}: kept better cached observations; Firestore history is incomplete")
                    continue
                dataset["days"].pop(key, None)
                dataset.get("discarded", {}).pop(key, None)
                incomplete.pop(key, None)
                if reason:
                    result["reason"] = reason
                    incomplete[key] = result
                    outcome = f"{key} saved for prediction. {len(result['timestamps'])}/{len(ticks)} snapshots. {reason}"
                else:
                    dataset["days"][key] = result
                    outcome = f"{key} saved for training and prediction. {len(ticks)} ticks, {len(result['rides'])} rides"
                progress(f"{key} saving local cache")
                save_dataset(settings.DATASET, dataset)
                completed += 1
                progress(f"{key} checked")
                message(outcome)
    except (GoogleAPICallError, GoogleAuthError) as error:
        message("Firestore request failed. Earlier saved days remain cached.")
        recovery = "Check Firestore access, credentials, and the configured project."
        if "invalid_grant" in str(error):
            recovery = (
                "Google rejected the saved credentials. Renew application default credentials with\n"
                "gcloud auth application-default login\n"
                "or set CREDENTIALS in ml/config.py to a valid service-account JSON file."
            )
        raise RuntimeError(
            f"Firestore failed while checking {key} ({type(error).__name__}): {error}\n"
            f"{recovery}\nRerun data.py to retry this day."
        ) from error
    except Exception:
        message("Export stopped during the displayed stage. Earlier saved days remain cached.")
        raise
    finally:
        db.close()
    message(f"Dataset has {len(dataset['days'])} training days and {len(incomplete)} incomplete prediction days in {settings.DATASET}")


def set_firestore_start_date(settings, value):
    """Persist a park's historical-API/Firebase handoff in its local dataset."""
    from ml.parks import get_park
    park = get_park(settings.PARK)
    value = date.fromisoformat(value).isoformat()
    dataset = load_dataset(settings.DATASET) if settings.DATASET.exists() else {
        "version": 1, "park_id": park.id, "days": {}, "incomplete_days": {},
        "discarded": {}, "catalogs": {}, "schedules": {},
    }
    if dataset["park_id"] != park.id:
        raise ValueError("Dataset belongs to a different park")
    previous = dataset.get("firestore_start_date")
    if previous and previous != value:
        raise ValueError(f"Firebase collection already starts on {previous}; the handoff was not changed")
    dataset["firestore_start_date"] = value
    save_dataset(settings.DATASET, dataset)


def export(settings=None):
    """Refresh finalized API history, then the recent Firestore tail for one park."""
    from types import SimpleNamespace
    from ml.parks import settings_for
    from ml.backfill import backfill

    settings = settings or settings_for(config.PARK)
    key = os.environ.get("THEMEPARKS_API_KEY")
    if not key:
        key = Path(settings.KEY_FILE).read_text(encoding="utf-8-sig").strip()
    if not key:
        raise ValueError("The ThemeParks.wiki API key is empty")
    final = backfill(key, settings.DATASET, settings.RAW_DIR,
                     settings.START_DATE, settings.END_DATE,
                     park=getattr(settings, "PARK", "disneyland"), refresh=settings.REFRESH)
    today = datetime.now(settings.PARK_TIMEZONE).date()
    # Overlap the archive boundary: its operating day may close after midnight.
    first = final if final else today - timedelta(days=settings.LIVE_DAYS - 1)
    if settings.START_DATE:
        first = max(first, date.fromisoformat(settings.START_DATE))
    last = date.fromisoformat(settings.END_DATE) if settings.END_DATE else today
    if first <= last:
        live_settings = SimpleNamespace(**{**vars(settings), "START_DATE": first.isoformat()})
        export_firestore(live_settings)
    return final


def main():
    import argparse
    from ml.parks import PARKS, settings_for

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--park", choices=PARKS, default=config.PARK)
    parser.add_argument("--since", help="First local date; default: all available paid history")
    parser.add_argument("--until", help="Last local date; default: today")
    parser.add_argument("--refresh", action="store_true", help="Rebuild historical days and refetch raw API history")
    parser.add_argument("--history-only", action="store_true", help="Skip Firestore (no Firebase credentials needed)")
    parser.add_argument("--live-only", action="store_true", help="Only refresh recent Firestore observations")
    parser.add_argument("--firestore-start-date", help="Reserve this operating date and newer days for Firebase only")
    args = parser.parse_args()
    if args.history_only and args.live_only:
        parser.error("Choose only one of --history-only and --live-only")
    settings = settings_for(args.park)
    settings.START_DATE = args.since or settings.START_DATE
    settings.END_DATE = args.until or settings.END_DATE
    settings.REFRESH = settings.REFRESH or args.refresh
    try:
        if args.firestore_start_date:
            set_firestore_start_date(settings, args.firestore_start_date)
        if args.live_only:
            settings.START_DATE = settings.START_DATE or (datetime.now(settings.PARK_TIMEZONE).date() - timedelta(days=settings.LIVE_DAYS - 1)).isoformat()
            export_firestore(settings)
        elif args.history_only:
            from ml.backfill import backfill
            key = os.environ.get("THEMEPARKS_API_KEY") or settings.KEY_FILE.read_text(encoding="utf-8-sig").strip()
            backfill(key, settings.DATASET, settings.RAW_DIR, settings.START_DATE, settings.END_DATE,
                     park=args.park, refresh=settings.REFRESH)
        else:
            export(settings)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
