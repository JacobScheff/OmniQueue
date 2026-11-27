"""Evaluate direct or autoregressive forecasts on held-out operating days."""

import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.attractions import modeling_day, validate_checkpoint_rides
from ml.data import (
    CLOSED, DOWN, OVERFLOW, REFURBISHMENT, TICK_SECONDS, TOKEN_LABELS, UNKNOWN,
    WAIT_STEP, align_day, load_dataset, time_features, validate_day,
)
from ml.model import build_model, select_device
from ml.availability import (
    FORECAST_POLICY_VERSION, advance_ages, decode_availability, decode_availability_sequence,
)

HORIZON_MINUTES = (5, 15, 30, 60, 120, 240, 360, 480, 600)


def empty_transitions():
    return {name: {"actual_events": 0, "predicted_events": 0, "matched_events": 0,
                   "missed_events": 0, "false_alarms": 0, "errors": []}
            for name in ("closing", "reopening")}


def score_transitions(origins, predictions, targets, observed=None):
    """Compare the next operating <-> DOWN/CLOSED transition in each rollout.

    Paths have shape [origins, steps, rides]. Only a continuous observed prefix
    is scored; missing/unknown/refurbishment states cannot establish event times.
    Repeated forecast origins are separate opportunities, not unique outages.
    """
    initial_valid = (origins >= 0) & (origins <= CLOSED)
    valid = (targets >= 0) & (targets <= CLOSED)
    if observed is not None:
        valid &= observed
    valid = valid.to(torch.int64).cumprod(dim=1).bool() & initial_valid.unsqueeze(1)
    initial_closed = (origins == DOWN) | (origins == CLOSED)
    actual_closed = (targets == DOWN) | (targets == CLOSED)
    predicted_closed = (predictions == DOWN) | (predictions == CLOSED)
    actual_events = valid & (actual_closed != initial_closed.unsqueeze(1))
    predicted_events = (valid & (predictions >= 0) & (predictions <= CLOSED)
                        & (predicted_closed != initial_closed.unsqueeze(1)))
    steps = predictions.shape[1]
    offsets = torch.arange(1, steps + 1, device=predictions.device).view(1, steps, 1)
    actual_first = torch.where(actual_events, offsets, steps + 1).amin(dim=1)
    predicted_first = torch.where(predicted_events, offsets, steps + 1).amin(dim=1)
    results = empty_transitions()
    for name, closed in (("closing", False), ("reopening", True)):
        eligible = initial_valid & valid[:, 0] & (initial_closed == closed)
        actual = eligible & (actual_first <= steps)
        predicted = eligible & (predicted_first <= steps)
        matched = actual & predicted
        results[name] = {
            "actual_events": actual.sum().item(), "predicted_events": predicted.sum().item(),
            "matched_events": matched.sum().item(), "missed_events": (actual & ~predicted).sum().item(),
            "false_alarms": (predicted & ~actual).sum().item(),
            "errors": ((predicted_first[matched] - actual_first[matched]) * TICK_SECONDS / 60).tolist(),
        }
    return results


def numeric_wait_errors(predictions, targets):
    """Signed minute errors for numeric wait pairs, excluding the 300+ bucket."""
    valid = ((predictions >= 0) & (predictions < OVERFLOW)
             & (targets >= 0) & (targets < OVERFLOW))
    return ((predictions[valid] - targets[valid]) * WAIT_STEP).to(device="cpu", dtype=torch.float64)


@torch.inference_mode()
def forecast_day(model, day, rides, device, batch_size, *, transition_totals=None):
    """Branch from each observed prefix and predict for up to ten hours.

    The prefix pass saves a separate GRU state at each origin. Batching those
    independent states avoids re-encoding every prefix for every forecast.
    V3 queries each saved origin directly; older models consume generated tokens.
    Future observations are read only when collecting scoring targets.
    """
    from zoneinfo import ZoneInfo
    from ml.data import PARK_TIMEZONE
    park_timezone = ZoneInfo(day["timezone"]) if day.get("timezone") else PARK_TIMEZONE
    model.eval()
    validate_day(day)
    timestamps = day["timestamps"]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Day timestamps must be strictly increasing")
    tokens = align_day(day, rides).to(device)
    # A forecast must stay inside one continuously observed operating session.
    remaining = [0] * len(timestamps)
    for index in range(len(timestamps) - 2, -1, -1):
        if timestamps[index + 1] - timestamps[index] == TICK_SECONDS:
            remaining[index] = remaining[index + 1] + 1
    observed_features = torch.tensor(time_features(timestamps[1:], park_timezone), dtype=torch.float32, device=device)
    origins, states, first_logits, locks, absences, outage_ages = [], [], [], [], [], []
    hidden = None
    ages = torch.zeros(len(rides), dtype=torch.long, device=device)
    seen = torch.zeros(len(rides), dtype=torch.bool, device=device)
    interpolated = day.get("interpolated_ticks", {})
    observed = torch.tensor([str(tick) not in interpolated for tick in timestamps], device=device)
    for index in range(len(timestamps) - 1):
        if index and timestamps[index] - timestamps[index - 1] != TICK_SECONDS:
            ages.zero_()
        ages = advance_ages(ages, tokens[index])
        logits, hidden = model(tokens[index:index + 1].unsqueeze(0),
                               observed_features[index:index + 1].unsqueeze(0), hidden)
        seen = seen | (tokens[index] != UNKNOWN)
        if not remaining[index]:
            continue
        # Interpolation cannot expose an endpoint later than this origin.
        if any(int(tick) <= timestamps[index] < right for tick, (_, right) in interpolated.items()):
            continue
        origins.append(index)
        states.append(hidden)
        first_logits.append(logits[0, 0])
        locks.append(tokens[index] == REFURBISHMENT)
        absences.append(~seen)
        outage_ages.append(ages)
    results = []
    for start in range(0, len(origins), batch_size):
        selected = origins[start:start + batch_size]
        origin_indices = torch.tensor(selected, device=device)
        lengths = torch.tensor([remaining[index] for index in selected], device=device)
        hidden = torch.cat(states[start:start + batch_size], dim=1)
        origin_states = hidden
        origin_tokens = tokens[origin_indices]
        previous = origin_tokens
        logits = torch.stack(first_logits[start:start + batch_size])
        locked = torch.stack(locks[start:start + batch_size])
        absent = torch.stack(absences[start:start + batch_size])
        ages = torch.stack(outage_ages[start:start + batch_size])
        observed_ages = ages
        survival = torch.ones_like(ages, dtype=torch.float32)
        steps = min(max(HORIZON_MINUTES) // 5, max(remaining[index] for index in selected))
        paths = []
        for step in range(1, steps + 1):
            # Direct forecasts use exactly the same probabilities, conditional
            # median waits and causal availability decoder as the map.
            if hasattr(model, "forecast_probabilities"):
                probabilities = model.direct_probabilities(logits.unsqueeze(1), observed_ages, [[step]])[:, 0]
                prediction = decode_availability(probabilities, previous)
            else:
                prediction, survival = model.greedy_prediction(logits, ages, survival)
            prediction[locked] = REFURBISHMENT
            prediction[absent] = UNKNOWN
            previous = prediction
            ages = advance_ages(ages, prediction)
            survival = torch.where(ages > 0, survival, 1.)
            if transition_totals is not None:
                paths.append(prediction.cpu())
            minutes = step * 5
            if minutes in HORIZON_MINUTES:
                eligible = lengths >= step
                if eligible.any():
                    indices = origin_indices[eligible]
                    results.append((minutes, indices.cpu(), prediction[eligible].cpu(),
                                    tokens[indices + step].cpu()))
            if step < steps:
                future = [timestamps[index] + (step + 1) * TICK_SECONDS for index in selected]
                features = torch.tensor(time_features(future, park_timezone), dtype=torch.float32, device=device).unsqueeze(1)
                if hasattr(model, "horizon_logits"):
                    logits = model.horizon_logits(origin_states, origin_tokens, features, [[step + 1]])
                else:
                    logits, hidden = model(prediction.unsqueeze(1), features, hidden)
                logits = logits[:, 0]
        if transition_totals is not None:
            offsets = torch.arange(1, steps + 1, device=device)
            indices = (origin_indices[:, None] + offsets).clamp_max(len(tokens) - 1)
            known_ticks = ((offsets <= lengths[:, None]) & observed[indices]
                           & observed[origin_indices, None]).unsqueeze(-1)
            events = score_transitions(tokens[origin_indices].cpu(), torch.stack(paths, dim=1),
                                       tokens[indices].cpu(), known_ticks.cpu())
            for name, values in events.items():
                for field, value in values.items():
                    if field == "errors":
                        transition_totals[name][field].extend(value)
                    else:
                        transition_totals[name][field] += value
    return results


def _prepare_test_day(day, rides):
    """Prepare independent origin metadata without reading future ride targets."""
    from zoneinfo import ZoneInfo
    from ml.data import PARK_TIMEZONE
    validate_day(day)
    timestamps = day["timestamps"]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Day timestamps must be strictly increasing")
    timezone = ZoneInfo(day["timezone"]) if day.get("timezone") else PARK_TIMEZONE
    tokens = align_day(day, rides)
    remaining = [0] * len(timestamps)
    for index in range(len(timestamps) - 2, -1, -1):
        if timestamps[index + 1] - timestamps[index] == TICK_SECONDS:
            remaining[index] = remaining[index + 1] + 1
    interpolated = day.get("interpolated_ticks", {})
    observed = torch.tensor([str(tick) not in interpolated for tick in timestamps])
    ages = torch.zeros(len(rides), dtype=torch.long)
    origins, outage_ages = [], []
    for index in range(len(timestamps) - 1):
        if index and timestamps[index] - timestamps[index - 1] != TICK_SECONDS:
            ages = torch.zeros_like(ages)
        ages = advance_ages(ages, tokens[index])
        if remaining[index] and not any(int(tick) <= timestamps[index] < right
                                        for tick, (_, right) in interpolated.items()):
            origins.append(index)
            outage_ages.append(ages)
    return {"tokens": tokens, "timestamps": timestamps, "timezone": timezone,
            "features": torch.tensor(time_features(timestamps[1:], timezone), dtype=torch.float32),
            "remaining": remaining, "observed": observed, "origins": origins, "ages": outage_ages}


@torch.inference_mode()
def forecast_days(model, days, rides, device, batch_size, *, horizon_batch_size=8):
    """Evaluate a bounded group of independent days together.

    V3 encodes all days in one causal, padded forward pass. Origin states are
    then batched across days, and independent future leads share model queries.
    Availability decoding stays causal across leads. Only scoring reads future
    targets, and each origin retains its own session and interpolation limits.
    Older architectures retain their existing autoregressive evaluator.
    """
    if batch_size < 1 or horizon_batch_size < 1:
        raise ValueError("Evaluation origin and horizon batch sizes must be positive")
    if not hasattr(model, "forecast_probabilities"):
        results = []
        for day in days:
            transitions = empty_transitions()
            forecasts = forecast_day(model, day, rides, device, batch_size, transition_totals=transitions)
            results.append((forecasts, transitions))
        return results
    from torch.nn.utils.rnn import pad_sequence
    model.eval()
    prepared = [_prepare_test_day(day, rides) for day in days]
    results = [([], empty_transitions()) for _ in days]
    # Round-robin origins keep multiple days represented in each GPU query,
    # rather than draining one day's forecast origins before the next day.
    records = [(row, day["origins"][index], day["ages"][index])
               for index in range(max((len(day["origins"]) for day in prepared), default=0))
               for row, day in enumerate(prepared) if index < len(day["origins"])]
    if not records:
        return results
    rows, origins, outage_ages = zip(*records)
    outage_ages = torch.stack(outage_ages).to(device)
    inputs = pad_sequence([day["tokens"][:-1] for day in prepared], batch_first=True, padding_value=UNKNOWN).to(device)
    features = pad_sequence([day["features"] for day in prepared], batch_first=True).to(device)
    immediate, states = model(inputs, features, return_state_sequence=True)
    rows_device = torch.tensor(rows, device=device)
    origins_device = torch.tensor(origins, device=device)
    origin_states = states[:, rows_device, origins_device].contiguous()
    first_logits = immediate[rows_device, origins_device]
    origin_tokens = inputs[rows_device, origins_device]
    del immediate, states, inputs, features
    lengths = torch.tensor([prepared[row]["remaining"][origin] for row, origin in zip(rows, origins)])
    steps = min(max(HORIZON_MINUTES) // 5, int(lengths.max()))
    for start in range(0, len(origins), batch_size):
        end = min(start + batch_size, len(origins))
        selected_rows, selected_origins = rows[start:end], origins[start:end]
        count = end - start
        batch_steps = min(steps, int(lengths[start:end].max()))
        # Calendar features are known at the origin, including leads after a
        # day's final observation. Such leads are generated but never scored.
        calendar = torch.tensor([
            time_features([prepared[row]["timestamps"][origin] + step * TICK_SECONDS
                           for step in range(1, batch_steps + 1)], prepared[row]["timezone"])
            for row, origin in zip(selected_rows, selected_origins)
        ], dtype=torch.float32, device=device)
        tokens = origin_tokens[start:end]
        previous = tokens.cpu()
        locked = previous == REFURBISHMENT
        absent = torch.stack([(prepared[row]["tokens"][:origin + 1] == UNKNOWN).all(0)
                              for row, origin in zip(selected_rows, selected_origins)])
        paths = []
        for lead_start in range(0, batch_steps, horizon_batch_size):
            lead_end = min(lead_start + horizon_batch_size, batch_steps)
            leads = torch.arange(lead_start + 1, lead_end + 1, device=device)
            logits = model.horizon_logits(origin_states[:, start:end], tokens,
                                           calendar[:, lead_start:lead_end], leads)
            if lead_start == 0:
                logits[:, 0] = first_logits[start:end]
            probabilities = model.direct_probabilities(logits, outage_ages[start:end], leads).cpu()
            prediction = decode_availability_sequence(probabilities, previous)
            prediction.masked_fill_(locked.unsqueeze(1), REFURBISHMENT)
            prediction.masked_fill_(absent.unsqueeze(1), UNKNOWN)
            previous = prediction[:, -1]
            paths.append(prediction)
        paths = torch.cat(paths, dim=1)
        for row in sorted(set(selected_rows)):
            selected = torch.tensor([index for index in range(count) if selected_rows[index] == row])
            indices = torch.tensor([selected_origins[index] for index in selected.tolist()])
            day = prepared[row]
            forecasts, transitions = results[row]
            day_lengths = lengths[start:end][selected]
            for minutes in HORIZON_MINUTES:
                step = minutes // 5
                eligible = day_lengths >= step
                if step <= batch_steps and eligible.any():
                    eligible_origins = indices[eligible]
                    forecasts.append((minutes, eligible_origins, paths[selected[eligible], step - 1],
                                      day["tokens"][eligible_origins + step]))
            offsets = torch.arange(1, batch_steps + 1)
            targets = (indices[:, None] + offsets).clamp_max(len(day["tokens"]) - 1)
            observed = ((offsets <= day_lengths[:, None]) & day["observed"][targets]
                        & day["observed"][indices, None]).unsqueeze(-1)
            events = score_transitions(day["tokens"][indices], paths[selected], day["tokens"][targets], observed)
            for name, values in events.items():
                for field, value in values.items():
                    if field == "errors":
                        transitions[name][field].extend(value)
                    else:
                        transitions[name][field] += value
    return results


def summarize_wait_errors(errors):
    """Pool individual ride/tick errors, including exact matches in all averages."""
    errors = torch.as_tensor(errors, dtype=torch.float64, device="cpu").flatten()
    count = errors.numel()
    if not count:
        return {"count": 0, "absolute_minutes": None, "signed_minutes": None,
                "mse_minutes_squared": None, "rmse_minutes": None, "overpredictions": None,
                "underpredictions": None, "exact_matches": None}

    def distribution(values):
        q1, median, q3 = values.quantile(torch.tensor([.25, .5, .75], dtype=torch.float64)).tolist()
        return {"mean": values.mean().item(), "min": values.min().item(),
                "q1": q1, "median": median, "q3": q3, "max": values.max().item(),
                "std": values.std(correction=0).item()}

    def direction(values):
        return {"count": values.numel(), "fraction": values.numel() / count,
                "mean_magnitude_minutes": values.abs().mean().item() if values.numel() else None}

    exact = (errors == 0).sum().item()
    mse = errors.square().mean().item()
    return {"count": count, "absolute_minutes": distribution(errors.abs()),
            "signed_minutes": distribution(errors), "mse_minutes_squared": mse, "rmse_minutes": mse ** .5,
            "overpredictions": direction(errors[errors > 0]),
            "underpredictions": direction(errors[errors < 0]),
            "exact_matches": {"count": exact, "fraction": exact / count}}


def horizon_label(minutes):
    amount, unit = (minutes / 60, "hour") if minutes >= 60 else (minutes, "minute")
    return f"{amount:g} {unit}{'' if amount == 1 else 's'}"


def print_horizon(minutes, result):
    accuracy = f"{result['accuracy']:.1%}" if result["accuracy"] is not None else "N/A"
    stats = result["wait_errors"]
    if stats["count"]:
        error = (f"MAE {stats['absolute_minutes']['mean']:.2f} min | "
                 f"MSE {stats['mse_minutes_squared']:.2f} min^2 | "
                 f"RMSE {stats['rmse_minutes']:.2f} min | bias {stats['signed_minutes']['mean']:+.2f} min")
    else:
        error = "MAE N/A | MSE N/A | RMSE N/A | bias N/A"
    print(f"{horizon_label(minutes):>10}: accuracy {accuracy} | {error} | {result['forecast_count']:,} forecasts | "
          f"numeric pairs {stats['count']:,}/{stats['actual_numeric_wait_count']:,}", flush=True)


def print_transitions(totals):
    print(f"\nClosure/reopening timing (next transition per ride/forecast origin, up to {horizon_label(max(HORIZON_MINUTES))}; "
          "DOWN or CLOSED = unavailable):", flush=True)
    results = {}
    for name, counts in totals.items():
        stats = summarize_wait_errors(counts["errors"])
        exact = sum(error == 0 for error in counts["errors"])
        accuracy = exact / counts["actual_events"] if counts["actual_events"] else None
        results[name] = {key: value for key, value in counts.items() if key != "errors"}
        results[name].update(exact_events=exact, accuracy=accuracy, timing_errors=stats)
        accuracy_text = f"{accuracy:.1%}" if accuracy is not None else "N/A"
        timing = (f"timing MAE {stats['absolute_minutes']['mean']:.2f} min | "
                  f"bias {stats['signed_minutes']['mean']:+.2f} min" if stats["count"] else
                  "timing MAE N/A | bias N/A")
        print(f"  {name.capitalize()}: exact {accuracy_text} ({exact:,}/{counts['actual_events']:,}) | "
              f"{timing} | matched {counts['matched_events']:,}/{counts['actual_events']:,} | "
              f"missed {counts['missed_events']:,} | false alarms {counts['false_alarms']:,}", flush=True)
    print("Timing errors use matched next transitions; positive bias = late, negative = early. "
          "Unknown/refurbishment/interpolated observations and session breaks end timing coverage.", flush=True)
    return results


def evaluate(settings=config):
    if settings.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be positive")
    day_batch_size = getattr(settings, "TEST_DAY_BATCH_SIZE", config.TEST_DAY_BATCH_SIZE)
    horizon_batch_size = getattr(settings, "TEST_HORIZON_BATCH_SIZE", config.TEST_HORIZON_BATCH_SIZE)
    if day_batch_size < 1 or horizon_batch_size < 1:
        raise ValueError("Evaluation day and horizon batch sizes must be positive")
    started = time.perf_counter()
    device = select_device(settings.DEVICE)
    checkpoint = torch.load(settings.CHECKPOINT, map_location=device, weights_only=True)
    if hasattr(settings, "PARK"):
        from ml.parks import get_park
        if checkpoint.get("park_id") != get_park(settings.PARK).id:
            raise ValueError("Checkpoint belongs to a different park")
    if checkpoint.get("version") != 1 or checkpoint.get("token_labels") != TOKEN_LABELS:
        raise ValueError("Unsupported checkpoint or token vocabulary")
    validate_checkpoint_rides(checkpoint["rides"])
    fields = ("train_dates", "validation_dates", "test_dates")
    if any(field not in checkpoint for field in fields):
        raise ValueError("This checkpoint has no saved day splits. Retrain with ml/train.py before testing")
    partitions = [checkpoint[field] for field in fields]
    seen = set()
    for selected in partitions:
        if len(set(selected)) != len(selected) or seen.intersection(selected):
            raise ValueError("Checkpoint train/validation/test dates overlap; retrain before testing")
        seen.update(selected)
    dates = checkpoint["test_dates"]
    if not dates:
        raise ValueError("This checkpoint has no held-out test days. Set TEST_SPLIT > 0 and retrain")
    dataset = load_dataset(settings.DATASET)
    if dataset["park_id"] != checkpoint["park_id"]:
        raise ValueError("Dataset and checkpoint belong to different parks")
    missing = [day for day in dates if day not in dataset["days"]]
    if missing:
        raise ValueError(f"Saved test days are missing from the complete-day cache: {', '.join(missing)}")
    if any(len(dataset["days"][day]["timestamps"]) < 2 for day in dates):
        raise ValueError("Every saved test day needs at least two snapshots")
    model = build_model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Test: {len(dates)} held-out days ({', '.join(dates)})", flush=True)
    strategy = "Direct multi-horizon" if hasattr(model, "horizon_logits") else "Autoregressive greedy"
    print(f"{strategy} forecasts from every eligible five-minute origin; "
          "only observations at or before each origin enter its forecast.", flush=True)
    if hasattr(model, "forecast_probabilities"):
        print(f"Parallel evaluation: up to {day_batch_size} days, {settings.BATCH_SIZE} forecast origins "
              f"and {horizon_batch_size} future leads per model query.", flush=True)
    else:
        day_batch_size = 1
    transitions = empty_transitions()
    totals = {minutes: {"correct": 0, "scored": 0, "forecasts": 0, "numeric_actual": 0, "errors": []}
              for minutes in HORIZON_MINUTES}
    for start in range(0, len(dates), day_batch_size):
        selected_dates = dates[start:start + day_batch_size]
        print(f"Evaluating test days {start + 1}-{start + len(selected_dates)}/{len(dates)} together", flush=True)
        batches = forecast_days(model, [modeling_day(dataset["days"][key]) for key in selected_dates],
                                checkpoint["rides"], device, settings.BATCH_SIZE,
                                horizon_batch_size=horizon_batch_size)
        for offset, (forecasts, events) in enumerate(batches, 1):
            key = selected_dates[offset - 1]
            print(f"Evaluating test day {start + offset}/{len(dates)}: {key} (complete)", flush=True)
            for name, values in events.items():
                for field, value in values.items():
                    if field == "errors":
                        transitions[name][field].extend(value)
                    else:
                        transitions[name][field] += value
            for minutes, origins, predictions, targets in forecasts:
                total = totals[minutes]
                valid = (targets >= 0) & (targets <= DOWN)
                total["correct"] += ((predictions == targets) & valid).sum().item()
                total["scored"] += valid.sum().item()
                total["forecasts"] += len(origins)
                total["numeric_actual"] += ((targets >= 0) & (targets < OVERFLOW)).sum().item()
                total["errors"].append(numeric_wait_errors(predictions, targets))
    results = {}
    for minutes, total in totals.items():
        wait_errors = summarize_wait_errors(torch.cat(total["errors"]) if total["errors"] else [])
        wait_errors["actual_numeric_wait_count"] = total["numeric_actual"]
        results[minutes] = {
            "accuracy": total["correct"] / total["scored"] if total["scored"] else None,
            "scored_targets": total["scored"], "forecast_count": total["forecasts"], "wait_errors": wait_errors,
        }
        print_horizon(minutes, results[minutes])
    print("Accuracy scores wait/overflow and DOWN tokens. Minute errors require two numeric wait tokens "
          "(0-300 min); numeric pairs shows coverage of actual numeric waits. "
          "Forecasts stop at session breaks and the end of each day.", flush=True)
    transitions = print_transitions(transitions)
    print(f"Test completed in {time.perf_counter() - started:.1f}s", flush=True)
    return {"forecast_policy_version": FORECAST_POLICY_VERSION,
            "dates": dates, "horizons": results, "transitions": transitions}


def main():
    import argparse
    from ml.parks import PARKS, settings_for
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--park", choices=PARKS, default=config.PARK)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=config.DEVICE)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate this saved model's own test dates")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, help="Save test metrics as JSON")
    parser.add_argument("--day-batch-size", type=int, default=config.TEST_DAY_BATCH_SIZE,
                        help="Maximum independent days encoded together")
    parser.add_argument("--horizon-batch-size", type=int, default=config.TEST_HORIZON_BATCH_SIZE,
                        help="Maximum direct future leads per model query")
    args = parser.parse_args()
    try:
        settings = settings_for(args.park, DEVICE=args.device, TEST_DAY_BATCH_SIZE=args.day_batch_size,
                                TEST_HORIZON_BATCH_SIZE=args.horizon_batch_size)
        if args.checkpoint:
            settings.CHECKPOINT = args.checkpoint.resolve()
        if args.dataset:
            settings.DATASET = args.dataset.resolve()
        evaluated_stat = settings.CHECKPOINT.stat()
        results = evaluate(settings)
        if args.output:
            import json
            import os
            import tempfile
            from datetime import datetime, timezone

            stat = settings.CHECKPOINT.stat()
            if (stat.st_mtime_ns, stat.st_size) != (evaluated_stat.st_mtime_ns, evaluated_stat.st_size):
                raise ValueError("The model changed during testing. Run the test again for the saved model.")
            report = {"version": 1, "park_id": settings.PARK_ID,
                      "evaluated_at": datetime.now(timezone.utc).isoformat(),
                      "checkpoint_mtime_ns": stat.st_mtime_ns, "checkpoint_size": stat.st_size,
                      **results}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=args.output.parent, suffix=".tmp")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                    json.dump(report, target, allow_nan=False)
                os.replace(temporary, args.output)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            print(f"Test report saved to {args.output}", flush=True)
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
