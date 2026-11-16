"""Evaluate the saved model on its original held-out operating days."""

import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.data import OVERFLOW, PARK_TIMEZONE, TOKEN_LABELS, WAIT_STEP, load_dataset
from ml.model import WaitPredictor, select_device
from ml.train import Days, collate_days, epoch


def numeric_wait_errors(predictions, targets):
    """Signed minute errors for numeric wait pairs, excluding the 300+ bucket."""
    valid = ((predictions >= 0) & (predictions < OVERFLOW)
             & (targets >= 0) & (targets < OVERFLOW))
    return ((predictions[valid] - targets[valid]) * WAIT_STEP).to(device="cpu", dtype=torch.float64)


def large_wait_errors(predictions, targets, days, rides):
    """Map numeric errors over 40 minutes to the target tick and checkpoint ride."""
    valid = ((predictions >= 0) & (predictions < OVERFLOW)
             & (targets >= 0) & (targets < OVERFLOW)
             & (((predictions - targets) * WAIT_STEP).abs() > 40))
    locations = valid.nonzero().tolist()
    predicted = predictions[valid].tolist()
    actual = targets[valid].tolist()
    records = []
    for (batch, step, column), prediction, target in zip(locations, predicted, actual):
        ride = rides[column]
        source_day = days[batch]
        name = next((row["name"] for row in source_day["rides"] if row["id"] == ride["id"]), ride["name"])
        # Inputs at index step predict the NEXT observation, not their own time.
        timestamp = source_day["timestamps"][step + 1]
        records.append({
            "ride_id": ride["id"], "ride_name": name,
            "time": datetime.fromtimestamp(timestamp, PARK_TIMEZONE).isoformat(),
            "predicted_minutes": prediction * WAIT_STEP, "actual_minutes": target * WAIT_STEP,
            "error_minutes": (prediction - target) * WAIT_STEP,
        })
    return records


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


def print_wait_errors(stats):
    count, actual = stats["count"], stats["actual_numeric_wait_count"]
    coverage = f"{count / actual:.1%}" if actual else "N/A"
    print(f"Numeric wait pairs: {count:,}/{actual:,} actual numeric waits ({coverage}). "
          "Both tokens must be waits from 0 to 300 minutes; overflow/status tokens are excluded.", flush=True)
    if not count:
        print("Wait-time errors: N/A (no comparable numeric wait pairs).", flush=True)
        return
    absolute, signed = stats["absolute_minutes"], stats["signed_minutes"]
    print(f"Wait-time MAE {absolute['mean']:.2f} min | RMSE {stats['rmse_minutes']:.2f} min", flush=True)
    print("Absolute errors (min): " + ", ".join(
        f"{label} {absolute[key]:.2f}" for key, label in
        (("min", "min"), ("q1", "Q1"), ("median", "median"), ("q3", "Q3"),
         ("max", "max"), ("std", "population std dev"))), flush=True)
    print(f"Signed errors (prediction - actual, min): mean bias {signed['mean']:+.2f}, "
          f"population std dev {signed['std']:.2f}", flush=True)
    for key, label in (("overpredictions", "Overpredictions"), ("underpredictions", "Underpredictions")):
        group = stats[key]
        mean = group["mean_magnitude_minutes"]
        magnitude = f"{mean:.2f} min" if mean is not None else "N/A"
        print(f"{label}: {group['count']:,} ({group['fraction']:.1%}), mean magnitude {magnitude}", flush=True)
    exact = stats["exact_matches"]
    print(f"Exact wait matches: {exact['count']:,} ({exact['fraction']:.1%})", flush=True)


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
    test_days = [dataset["days"][day] for day in dates]
    loader = DataLoader(Days(test_days, checkpoint["rides"]),
                        batch_size=settings.BATCH_SIZE, shuffle=False,
                        pin_memory=device.type == "cuda", collate_fn=collate_days)
    model = WaitPredictor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    errors = []
    actual_numeric_wait_count = 0
    large_errors = []
    day_offset = 0

    def collect_errors(predictions, targets):
        nonlocal actual_numeric_wait_count, day_offset
        errors.append(numeric_wait_errors(predictions, targets))
        actual_numeric_wait_count += ((targets >= 0) & (targets < OVERFLOW)).sum().item()
        batch_days = test_days[day_offset:day_offset + len(predictions)]
        large_errors.extend(large_wait_errors(predictions, targets, batch_days, checkpoint["rides"]))
        day_offset += len(predictions)

    loss, accuracy = epoch(model, loader, device, on_predictions=collect_errors)
    if not math.isfinite(loss):
        raise ValueError("Test produced a nonfinite loss")
    print(f"Test: {len(dates)} held-out days ({', '.join(dates)})", flush=True)
    print(f"Test NLL {loss:.4f}, token accuracy {accuracy:.1%}", flush=True)
    print("Accuracy is exact next-five-minute token accuracy using observed history; "
          "wait/overflow and DOWN targets only.", flush=True)
    wait_errors = summarize_wait_errors(torch.cat(errors) if errors else [])
    wait_errors["actual_numeric_wait_count"] = actual_numeric_wait_count
    print_wait_errors(wait_errors)
    print(f"Wait-time errors > 40 min: {len(large_errors):,}", flush=True)
    for record in large_errors:
        local = datetime.fromisoformat(record["time"]).astimezone(PARK_TIMEZONE)
        print(f"  {local:%Y-%m-%d %H:%M %Z} | {record['ride_name']} | "
              f"predicted {record['predicted_minutes']} min | actual {record['actual_minutes']} min | "
              f"error {record['error_minutes']:+d} min", flush=True)
    return {"loss": loss, "accuracy": accuracy, "dates": dates, "wait_errors": wait_errors,
            "large_wait_errors": large_errors}


def main():
    try:
        evaluate()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
