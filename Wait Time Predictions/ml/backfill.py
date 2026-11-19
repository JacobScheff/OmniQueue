"""Backfill the local training cache from ThemeParks.wiki change history."""

import argparse
import gzip
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.data import (
    MISSING, PARK_TIMEZONE, TICK_SECONDS, expected_ticks, load_dataset,
    ride_token, save_dataset, validate_day,
)

PARK_ID = "7340550b-c14d-4def-80bb-acdb51d49a66"
API = "https://api.themeparks.wiki/v1"
PARK_PATH = f"/entity/{PARK_ID}"


def moment(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("History timestamps must include a timezone")
    return result.timestamp()


def reconstruct(ticks, documents, attraction_ids):
    """Sample the latest state at or BEFORE each tick, including opening state.

    Documents are calendar days in park time; an operating day may use two.
    Carrying a value forward is part of the API's change-history semantics,
    not interpolation. Never use a later change to fill an earlier tick.
    """
    rows, names = {}, {}
    for document in documents:
        if document.get("id") != PARK_ID or document.get("entityType") != "PARK":
            raise ValueError("History response belongs to a different park")
        local_day = date.fromisoformat(document["range"]["from"])
        day_ticks = [tick for tick in ticks
                     if datetime.fromtimestamp(tick, PARK_TIMEZONE).date() == local_day]
        if not day_ticks:
            continue
        entities = document["entities"]
        if len({entity["id"] for entity in entities}) != len(entities):
            raise ValueError("Duplicate entities in history response")
        for entity in entities:
            ride_id = entity["id"]
            if ride_id not in attraction_ids or entity.get("entityType") != "ATTRACTION":
                continue
            opening = entity.get("opening", {})
            if opening.get("degraded"):
                raise ValueError(f"Incomplete opening state for {entity['name']}; retry later")
            history = sorted(entity["history"], key=lambda row: moment(row["time"]))
            if not opening.get("observedAt") and not history:
                continue  # No evidence this attraction existed in this range.
            names[ride_id] = entity["name"]
            state = opening if opening.get("observedAt") else None
            index = 0
            for tick in day_ticks:
                while index < len(history) and moment(history[index]["time"]) <= tick:
                    state = history[index]
                    index += 1
                token = MISSING
                if state is not None and moment(state["time"]) <= tick:
                    standby = (state.get("queue") or {}).get("STANDBY") or {}
                    token = ride_token({"status": state.get("status"), "wait": standby.get("waitTime")})
                rows.setdefault(tick, {})[ride_id] = token
    ride_ids = sorted(names)
    return {
        "timestamps": sorted(rows), "expected_timestamps": ticks,
        "rides": [{"id": ride_id, "name": names[ride_id]} for ride_id in ride_ids],
        "tokens": [[rows[tick].get(ride_id, MISSING) for ride_id in ride_ids] for tick in sorted(rows)],
        "source": "themeparks.wiki/history", "sampling": "last_state_at_or_before_tick",
        "attribution": "Powered by ThemeParks.wiki",
    }


def request_json(path, key):
    request = urllib.request.Request(API + path, headers={
        "X-API-Key": key, "User-Agent": "OmniQueueBackfill/1.0",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Do not print request headers, API keys, or account details.
        if error.code == 429:
            raise RuntimeError(f"API rate limit reached; retry after {error.headers.get('Retry-After', 'the budget resets')} seconds") from None
        raise RuntimeError(f"ThemeParks.wiki returned HTTP {error.code} for {path}") from None


def reencode_cached_history(dataset_path, raw_dir):
    """Rebuild imported tokens from raw minutes, without network access."""
    dataset_path, raw_dir = Path(dataset_path), Path(raw_dir)
    dataset = load_dataset(dataset_path)
    changed, skipped = 0, 0
    for group in ("days", "incomplete_days"):
        for day in dataset.get(group, {}).values():
            if day.get("source") != "themeparks.wiki/history":
                continue
            calendar_days = sorted({datetime.fromtimestamp(tick, PARK_TIMEZONE).date()
                                    for tick in day["timestamps"]})
            paths = [raw_dir / f"{value}.json.gz" for value in calendar_days]
            if not all(path.exists() for path in paths):
                skipped += 1
                continue
            documents = []
            for path in paths:
                with gzip.open(path, "rt", encoding="utf-8") as source:
                    documents.append(json.load(source))
            rebuilt = reconstruct(day["timestamps"], documents, {ride["id"] for ride in day["rides"]})
            if (rebuilt["timestamps"] != day["timestamps"]
                    or rebuilt["rides"] != day["rides"]):
                raise ValueError("Raw history does not match the cached attraction catalog/timestamps")
            if rebuilt["tokens"] != day["tokens"]:
                day["tokens"] = rebuilt["tokens"]
                changed += 1
    if changed:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = dataset_path.with_name(f"{dataset_path.name}.before-reencode-{stamp}.bak")
        shutil.copy2(dataset_path, backup)
        save_dataset(dataset_path, dataset)
    print(f"Reencoded {changed} imported days from raw history; {skipped} missing raw files. "
          "Refresh Firestore days with ml/data.py --refresh to reencode collected observations.", flush=True)


def backfill(key, dataset_path, raw_dir, since=None, until=None):
    plan = request_json("/me", key)
    coverage = request_json(PARK_PATH + "/history/coverage", key)
    summary = coverage["summary"]
    # Respect the server's dates rather than guessing the plan boundary.
    earliest = max(filter(None, [plan.get("historyEarliestDate"), summary["archiveFrom"]]))
    first = date.fromisoformat(since or earliest)
    if first.isoformat() < earliest:
        raise ValueError(f"Your key can retrieve history starting {earliest}")
    final = min(date.fromisoformat(summary["recordedTo"]),
                datetime.now(PARK_TIMEZONE).date() - timedelta(days=1))
    last = min(date.fromisoformat(until), final) if until else final
    if first > last:
        raise ValueError("No finalized history is available in this date range")
    print(f"Plan: {plan['tier']}. Backfill: {first} through {last}.", flush=True)
    dataset_path, raw_dir = Path(dataset_path), Path(raw_dir)
    existed = dataset_path.exists()
    dataset = load_dataset(dataset_path) if existed else {
        "version": 1, "park_id": PARK_ID, "days": {}, "incomplete_days": {},
        "discarded": {}, "catalogs": {}, "schedules": {},
    }
    if dataset["park_id"] != PARK_ID:
        raise ValueError("Dataset belongs to a different park")
    dataset.setdefault("incomplete_days", {})
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot the original cache before the first write of each run.
    backed_up = False
    attraction_ids = {row["id"] for row in coverage["entities"]
                      if row["entityType"] == "ATTRACTION" and row.get("stillListed")}
    refreshed_months = set()

    def history(day):
        path = raw_dir / f"{day}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as source:
                return json.load(source)
        payload = request_json(PARK_PATH + f"/history?date={day}", key)
        # Leave an interrupted download uncached.
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as target:
            json.dump(payload, target, separators=(",", ":"))
        os.replace(temporary, path)
        return payload

    imported = 0
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        name = day.isoformat()
        # Preserve both complete and incomplete locally collected observations.
        if name in dataset["days"] or (name in dataset["incomplete_days"]
                and dataset["incomplete_days"][name].get("source") != "themeparks.wiki/history"):
            print(f"{name}: kept existing observations", flush=True)
            continue
        month = day.strftime("%Y/%m")
        if month not in refreshed_months:
            schedule = request_json(PARK_PATH + f"/schedule/{month}", key)
            if schedule.get("id") != PARK_ID:
                raise ValueError("Schedule response belongs to a different park")
            dataset["schedules"][month] = schedule["schedule"]
            refreshed_months.add(month)
        ticks, closing = expected_ticks(day, dataset["schedules"][month])
        if not ticks:
            print(f"{name}: skipped; no published operating schedule", flush=True)
            continue
        calendar_days = sorted({datetime.fromtimestamp(tick, PARK_TIMEZONE).date() for tick in ticks})
        if calendar_days[-1] > final:
            print(f"{name}: skipped; overnight history is not finalized yet", flush=True)
            continue
        result = reconstruct(ticks, [history(value) for value in calendar_days], attraction_ids)
        result.update(checked_at=datetime.now(timezone.utc).timestamp(), closing_at=closing)
        reason = None
        try:
            if not result["rides"]:
                raise ValueError("No attractions have recorded history")
            validate_day(result)
        except ValueError as error:
            reason = str(error)
        dataset["incomplete_days"].pop(name, None)
        dataset.get("discarded", {}).pop(name, None)
        if reason:
            result["reason"] = reason
            dataset["incomplete_days"][name] = result
        else:
            dataset["days"][name] = result
            imported += 1
        if existed and not backed_up:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = dataset_path.with_name(f"{dataset_path.name}.before-backfill-{stamp}.bak")
            shutil.copy2(dataset_path, backup)
            print(f"Original cache backed up to {backup}", flush=True)
            backed_up = True
        save_dataset(dataset_path, dataset)
        print(f"{name}: {len(result['timestamps'])} ticks, {len(result['rides'])} attractions; "
              + (reason or "ready for training"), flush=True)
    print(f"Added {imported} training days. Cache now has {len(dataset['days'])} complete days "
          f"and {len(dataset['incomplete_days'])} incomplete days.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", help="First local operating date, YYYY-MM-DD")
    parser.add_argument("--until", help="Last local operating date, YYYY-MM-DD")
    parser.add_argument("--key-file", type=Path, default=config.ROOT / "credentials/themeparks-api-key.txt")
    parser.add_argument("--dataset", type=Path, default=config.DATASET)
    parser.add_argument("--raw-dir", type=Path, default=config.ROOT / "data/themeparks-history")
    parser.add_argument("--reencode-cache", action="store_true",
                        help="Rebuild imported tokens from local raw history; no API key or network needed")
    args = parser.parse_args()
    try:
        if args.reencode_cache:
            reencode_cached_history(args.dataset, args.raw_dir)
            return
        key = os.environ.get("THEMEPARKS_API_KEY") or args.key_file.read_text(encoding="utf-8-sig").strip()
        if not key:
            raise ValueError("The ThemeParks.wiki key file is empty")
        backfill(key, args.dataset, args.raw_dir, args.since, args.until)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
