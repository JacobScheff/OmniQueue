"""Evaluate autoregressive forecasts on the model's held-out operating days."""

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.data import (
    CLOSED, DOWN, OVERFLOW, REFURBISHMENT, TICK_SECONDS, TOKEN_LABELS, UNKNOWN,
    WAIT_STEP, align_day, load_dataset, time_features, validate_day,
)
from ml.model import WaitPredictor, select_device

HORIZON_MINUTES = (5, 15, 30, 60, 120, 240)


def numeric_wait_errors(predictions, targets):
    """Signed minute errors for numeric wait pairs, excluding the 300+ bucket."""
    valid = ((predictions >= 0) & (predictions < OVERFLOW)
             & (targets >= 0) & (targets < OVERFLOW))
    return ((predictions[valid] - targets[valid]) * WAIT_STEP).to(device="cpu", dtype=torch.float64)


@torch.inference_mode()
def forecast_day(model, day, rides, device, batch_size):
    """Branch from each observed prefix, then feed predictions back for up to 4h.

    The prefix pass saves a separate GRU state at each origin. Batching those
    independent states avoids re-encoding every prefix for every forecast.
    Future observations are read only when collecting scoring targets.
    """
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
    observed_features = torch.tensor(time_features(timestamps[1:]), dtype=torch.float32, device=device)
    origins, states, first_logits, locks, absences = [], [], [], [], []
    hidden = None
    seen = torch.zeros(len(rides), dtype=torch.bool, device=device)
    interpolated = day.get("interpolated_ticks", {})
    for index in range(len(timestamps) - 1):
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
    results = []
    for start in range(0, len(origins), batch_size):
        selected = origins[start:start + batch_size]
        origin_indices = torch.tensor(selected, device=device)
        lengths = torch.tensor([remaining[index] for index in selected], device=device)
        hidden = torch.cat(states[start:start + batch_size], dim=1)
        logits = torch.stack(first_logits[start:start + batch_size])
        locked = torch.stack(locks[start:start + batch_size])
        absent = torch.stack(absences[start:start + batch_size])
        steps = min(max(HORIZON_MINUTES) // 5, max(remaining[index] for index in selected))
        for step in range(1, steps + 1):
            # Match forecast()'s supported output classes and context locks,
            # using deterministic most-likely tokens instead of sampling paths.
            logits = logits.clone()
            logits[..., [CLOSED, REFURBISHMENT, UNKNOWN]] = -torch.inf
            prediction = logits.argmax(dim=-1)
            prediction[locked] = REFURBISHMENT
            prediction[absent] = UNKNOWN
            minutes = step * 5
            if minutes in HORIZON_MINUTES:
                eligible = lengths >= step
                if eligible.any():
                    indices = origin_indices[eligible]
                    results.append((minutes, indices.cpu(), prediction[eligible].cpu(),
                                    tokens[indices + step].cpu()))
            if step < steps:
                future = [timestamps[index] + (step + 1) * TICK_SECONDS for index in selected]
                features = torch.tensor(time_features(future), dtype=torch.float32, device=device).unsqueeze(1)
                logits, hidden = model(prediction.unsqueeze(1), features, hidden)
                logits = logits[:, 0]
    return results


def summarize_wait_errors(errors):
    """Pool individual ride/tick errors, including exact matches in all averages."""
    errors = torch.as_tensor(errors, dtype=torch.float64, device="cpu").flatten()
    count = errors.numel()
    if not count:
        return {"count": 0, "absolute_minutes": None, "signed_minutes": None,
                "rmse_minutes": None, "overpredictions": None,
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
    return {"count": count, "absolute_minutes": distribution(errors.abs()),
            "signed_minutes": distribution(errors), "rmse_minutes": errors.square().mean().sqrt().item(),
            "overpredictions": direction(errors[errors > 0]),
            "underpredictions": direction(errors[errors < 0]),
            "exact_matches": {"count": exact, "fraction": exact / count}}


def print_horizon(minutes, result):
    accuracy = f"{result['accuracy']:.1%}" if result["accuracy"] is not None else "N/A"
    stats = result["wait_errors"]
    if stats["count"]:
        error = (f"MAE {stats['absolute_minutes']['mean']:.2f} min | "
                 f"RMSE {stats['rmse_minutes']:.2f} min | bias {stats['signed_minutes']['mean']:+.2f} min")
    else:
        error = "MAE N/A | RMSE N/A | bias N/A"
    print(f"{minutes:3} min: accuracy {accuracy} | {error} | {result['forecast_count']:,} forecasts | "
          f"numeric pairs {stats['count']:,}/{stats['actual_numeric_wait_count']:,}", flush=True)


def evaluate(settings=config):
    if settings.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be positive")
    device = select_device(settings.DEVICE)
    checkpoint = torch.load(settings.CHECKPOINT, map_location=device, weights_only=True)
    if checkpoint.get("version") != 1 or checkpoint.get("token_labels") != TOKEN_LABELS:
        raise ValueError("Unsupported checkpoint or token vocabulary")
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
    model = WaitPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Test: {len(dates)} held-out days ({', '.join(dates)})", flush=True)
    print("Autoregressive greedy forecasts from every eligible five-minute origin; "
          "only observations at or before each origin enter its forecast.", flush=True)
    totals = {minutes: {"correct": 0, "scored": 0, "forecasts": 0, "numeric_actual": 0, "errors": []}
              for minutes in HORIZON_MINUTES}
    for key in dates:
        for minutes, origins, predictions, targets in forecast_day(
                model, dataset["days"][key], checkpoint["rides"], device, settings.BATCH_SIZE):
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
    return {"dates": dates, "horizons": results}


def main():
    try:
        evaluate()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
