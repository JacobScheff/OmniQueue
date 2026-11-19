"""Forecast waits with a trained checkpoint and settings in config.py."""

import argparse
import math
import sys
from bisect import bisect_left
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import torch

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.attractions import modeling_day, validate_checkpoint_rides
from ml.data import (
    CLOSED, DOWN, OVERFLOW, PARK_TIMEZONE, REFURBISHMENT, TICK_SECONDS, TOKEN_LABELS, UNKNOWN,
    WAIT_STEP, align_day, expected_ticks, history_from_json, interpolate_day, load_dataset, time_features, validate_day,
)
from ml.model import WaitPredictor, select_device


def summarize(paths, rides, timestamps):
    """Show unavailability when at least half of paths are DOWN or CLOSED."""
    paths = paths.cpu()
    snapshots = []
    for step, timestamp in enumerate(timestamps):
        rows = []
        for index, ride in enumerate(rides):
            tokens = paths[:, step, index]
            row = {**ride, "down_probability": (tokens == DOWN).float().mean().item()}
            row["closed_probability"] = (tokens == CLOSED).float().mean().item()
            if (tokens == REFURBISHMENT).all():
                row.update(status="REFURBISHMENT", wait_minutes=None)
            elif (tokens == UNKNOWN).all():
                row.update(status="UNKNOWN", wait_minutes=None)
            elif ((tokens == DOWN) | (tokens == CLOSED)).sum().item() * 2 >= len(tokens):
                status = "CLOSED" if row["closed_probability"] >= row["down_probability"] else "DOWN"
                row.update(status=status, wait_minutes=None,
                           operating_probability=(tokens <= OVERFLOW).float().mean().item())
            else:
                waits = tokens[tokens <= OVERFLOW].float()
                row["operating_probability"] = len(waits) / len(tokens)
                row["over_300_probability"] = (tokens == OVERFLOW).float().mean().item()
                # The overflow bucket is shown explicitly, never presented as an exact wait.
                if len(waits):
                    quantiles = waits.quantile(torch.tensor([0.1, 0.5, 0.9]), interpolation="nearest").long().tolist()
                    row["wait_minutes"] = {
                        name: "300+" if token == OVERFLOW else token * WAIT_STEP
                        for name, token in zip(("p10", "p50", "p90"), quantiles)
                    }
                else:
                    row["wait_minutes"] = None
            rows.append(row)
        snapshots.append({"time": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), "rides": rows})
    return snapshots


def print_predictions(result):
    observed = datetime.fromisoformat(result["as_of"]).astimezone(PARK_TIMEZONE)
    print(f"Forecast for {result['day']} | {result['samples']} samples", flush=True)
    print(f"Observed through {observed:%Y-%m-%d %H:%M %Z}. Wait percentiles apply when operating.", flush=True)
    for snapshot in result["snapshots"]:
        moment = datetime.fromisoformat(snapshot["time"]).astimezone(PARK_TIMEZONE)
        print(f"\n{moment:%Y-%m-%d %H:%M %Z}", flush=True)
        for ride in snapshot["rides"]:
            if ride.get("status"):
                print(f"  {ride['name']} | {ride['status']}", flush=True)
                continue
            wait = ride["wait_minutes"]
            down = f"DOWN {ride['down_probability']:.0%} | CLOSED {ride.get('closed_probability', 0):.0%}"
            if wait is None:
                print(f"  {ride['name']} | {down}", flush=True)
            else:
                print(f"  {ride['name']} | {wait['p50']} min | "
                      f"p10 {wait['p10']}, p90 {wait['p90']} | {down}", flush=True)


def park_timestamp(opening_date, clock_text, first_timestamp):
    """Resolve a park-local clock time on the selected operating day."""
    clock = time.fromisoformat(clock_text)
    if clock.tzinfo is not None:
        raise ValueError("Use a park-local time such as 18:00")
    moment = datetime.combine(date.fromisoformat(opening_date), clock, PARK_TIMEZONE)
    if moment.timestamp() < first_timestamp:
        moment += timedelta(days=1)
    return math.ceil(moment.timestamp() / TICK_SECONDS) * TICK_SECONDS


def forecast_boundary(day, opening_date, start_time):
    """Use observed tokens strictly before the first predicted five-minute tick."""
    timestamps = day["timestamps"]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("History needs strictly increasing timestamps")
    expected = day.get("expected_timestamps", timestamps)
    if not expected:
        raise ValueError("No published schedule or observations are available for this day")
    if start_time is None:
        if not timestamps:
            raise ValueError("No observations are available before the prediction start")
        cutoff = timestamps[-1] + TICK_SECONDS
    else:
        cutoff = park_timestamp(opening_date, start_time, expected[0])
    validate_day(day, before=cutoff)
    count = bisect_left(timestamps, cutoff)
    if not count:
        raise ValueError("Choose a prediction start after the day's first observed snapshot")
    if timestamps[count - 1] + TICK_SECONDS != cutoff:
        raise ValueError("The selected start needs an observed snapshot in the preceding five-minute bucket")
    return count, cutoff


def predict(settings=config, target_time=None, *, print_output=True):
    if (settings.STEPS is not None and settings.STEPS < 1) or settings.SAMPLES < 1 or settings.TEMPERATURE <= 0:
        raise ValueError("Steps, samples, and temperature must be positive")
    torch.manual_seed(settings.SEED)
    device = select_device(settings.DEVICE)
    checkpoint = torch.load(settings.CHECKPOINT, map_location=device, weights_only=True)
    if checkpoint.get("version") != 1 or checkpoint["token_labels"] != TOKEN_LABELS:
        raise ValueError("Unsupported checkpoint or token vocabulary")
    validate_checkpoint_rides(checkpoint["rides"])
    model = WaitPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if settings.HISTORY:
        day = history_from_json(settings.HISTORY)
        key = settings.PREDICT_DAY or datetime.fromtimestamp(day["timestamps"][0], PARK_TIMEZONE).date().isoformat()
        if "expected_timestamps" not in day:
            cache = load_dataset(settings.DATASET) if settings.DATASET.exists() else {}
            cached_day = cache.get("days", {}).get(key) or cache.get("incomplete_days", {}).get(key)
            if cached_day:
                day["expected_timestamps"] = cached_day.get("expected_timestamps", cached_day["timestamps"])
            else:
                schedule = cache.get("schedules", {}).get(date.fromisoformat(key).strftime("%Y/%m"), [])
                day["expected_timestamps"], _ = expected_ticks(date.fromisoformat(key), schedule)
    else:
        dataset = load_dataset(settings.DATASET)
        if dataset["park_id"] != checkpoint["park_id"]:
            raise ValueError("Dataset and checkpoint belong to different parks")
        available = {**dataset.get("incomplete_days", {}), **dataset["days"]}
        if not available:
            raise ValueError("Dataset contains no cached days. Run data.py first.")
        key = settings.PREDICT_DAY or max(available)
        if key not in available:
            raise ValueError(f"Dataset has no cached day for {key}. Run data.py to fetch it.")
        day = available[key]
    day = interpolate_day(modeling_day(day))
    count, cutoff = forecast_boundary(day, key, settings.PREDICT_START_TIME)
    timestamps = day["timestamps"][:count]
    # Later observations supply the replay's end time, while history contains
    # only the tokens available before the forecast starts.
    history_day = {**day, "timestamps": timestamps, "tokens": day["tokens"][:count]}
    tokens = align_day(history_day, checkpoint["rides"])
    steps = settings.STEPS
    if target_time is not None:
        target = park_timestamp(key, target_time, day["timestamps"][0])
        if target < cutoff:
            start = datetime.fromtimestamp(cutoff, PARK_TIMEZONE)
            raise ValueError(f"Requested time must be at or after the prediction start, {start:%H:%M %Z}")
        steps = (target - cutoff) // TICK_SECONDS + 1
    elif steps is None:
        end = day.get("expected_timestamps", day["timestamps"])[-1]
        steps = (end - cutoff) // TICK_SECONDS + 1
        if steps <= 0:
            steps = 24
    future = [cutoff + TICK_SECONDS * step for step in range(steps)]
    features = torch.tensor(time_features(timestamps[1:] + future), dtype=torch.float32, device=device)
    paths = model.forecast(tokens.to(device), features,
                           steps=steps, samples=settings.SAMPLES, temperature=settings.TEMPERATURE,
                           history_timestamps=timestamps)
    if target_time is not None:
        paths = paths[:, -1:]
        future = future[-1:]
    result = {"as_of": datetime.fromtimestamp(timestamps[-1], timezone.utc).isoformat(),
              "day": key, "prediction_start": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
              "samples": settings.SAMPLES, "snapshots": summarize(paths, checkpoint["rides"], future)}
    if print_output:
        print_predictions(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Print ride predictions at a park-local time.")
    parser.add_argument("time", metavar="HH:MM", help="Target time, generated from PREDICT_START_TIME in config.py")
    args = parser.parse_args()
    try:
        predict(target_time=args.time)
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
