"""Train on whole operating days, with chronological validation/test holdouts."""

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
from ml.attractions import is_excluded
from ml.data import (
    DOWN, TICK_SECONDS, TOKEN_LABELS, UNKNOWN,
    align_day, load_dataset, time_features,
)
from ml.model import WaitPredictor, select_device, token_loss


class Days(Dataset):
    """Each sequence starts with the first observation of its operating day."""

    def __init__(self, days, rides):
        self.days = []
        for day in days:
            tokens = align_day(day, rides)
            if len(tokens) < 2:
                continue
            features = torch.tensor(time_features(day["timestamps"]), dtype=torch.float32)
            timestamps = np.asarray(day["timestamps"], dtype=np.int64)
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError("Day timestamps must be strictly increasing")
            targets = tokens[1:].clone()
            # Retain earlier history across scheduled breaks, while excluding
            # transitions whose target is more than five minutes away.
            targets[np.diff(timestamps) != TICK_SECONDS] = UNKNOWN
            self.days.append((tokens[:-1], features[1:], targets))

    def __len__(self):
        return len(self.days)

    def __getitem__(self, index):
        return self.days[index]


def collate_days(days):
    """Pad shorter days. UNKNOWN targets are excluded from the loss."""
    inputs, features, targets = zip(*days)
    return (
        pad_sequence(inputs, batch_first=True, padding_value=UNKNOWN),
        pad_sequence(features, batch_first=True),
        pad_sequence(targets, batch_first=True, padding_value=UNKNOWN),
    )


def split_dates(dates, train_fraction, val_fraction, test_fraction):
    """Allocate whole days in chronological order, keeping enabled splits nonempty."""
    fractions = (train_fraction, val_fraction, test_fraction)
    if (any(not math.isfinite(value) or value < 0 for value in fractions)
            or train_fraction <= 0 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9)):
        raise ValueError("TRAIN_SPLIT, VAL_SPLIT, and TEST_SPLIT must be nonnegative, sum to 1, and include training")
    dates = sorted(dates)
    if len(set(dates)) != len(dates):
        raise ValueError("Operating dates must be unique")
    minimums = [int(value > 0) for value in fractions]
    if len(dates) < sum(minimums):
        raise ValueError(f"Need at least {sum(minimums)} complete days with two snapshots for the configured splits")
    targets = [len(dates) * value for value in fractions]
    counts = [max(minimum, math.floor(target)) for minimum, target in zip(minimums, targets)]
    while sum(counts) > len(dates):
        index = max((i for i in range(3) if counts[i] > minimums[i]),
                    key=lambda i: counts[i] - targets[i])
        counts[index] -= 1
    while sum(counts) < len(dates):
        index = max((i for i in range(3) if fractions[i] > 0),
                    key=lambda i: targets[i] - counts[i])
        counts[index] += 1
    train_end, val_end = counts[0], counts[0] + counts[1]
    return dates[:train_end], dates[train_end:val_end], dates[val_end:]


def epoch(model, loader, device, optimizer=None):
    """Score next-tick targets; only training enables gradients and weight updates."""
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for tokens, features, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits, _ = model(tokens, features)
            loss, count = token_loss(logits, targets)
            if training and count.item():
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.detach() * count
        total_count += count
        predictions = logits.argmax(dim=-1)
        total_correct += ((predictions == targets) & (targets <= DOWN)).sum()
    count = total_count.item()
    if count == 0:
        raise ValueError("These days contain no wait or DOWN targets")
    return total_loss.item() / count, total_correct.item() / count


def save_checkpoint(checkpoint, path):
    """Publish a complete checkpoint, allowing brief Windows file locks to clear."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    for attempt in range(41):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if attempt == 40:
                raise PermissionError(
                    f"Could not replace checkpoint {path} after 10 seconds of retries. "
                    f"The completed checkpoint is preserved at {temporary}; "
                    "close programs holding either file open and move it to the checkpoint path "
                    "before starting another training run."
                ) from error
            time.sleep(0.25)


def train(settings=config):
    if min(settings.EPOCHS, settings.BATCH_SIZE, settings.HIDDEN_SIZE, settings.EMBEDDING_SIZE) < 1:
        raise ValueError("Epochs, batch size, and model sizes must be positive")
    if settings.LEARNING_RATE <= 0:
        raise ValueError("Learning rate must be positive")
    random.seed(settings.SEED)
    np.random.seed(settings.SEED)
    torch.manual_seed(settings.SEED)
    device = select_device(settings.DEVICE)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if not settings.DATASET.exists():
        raise ValueError("No local dataset. Set the export dates in config.py and run python data.py first.")
    dataset = load_dataset(settings.DATASET)
    dates = sorted(day for day, row in dataset["days"].items() if len(row["timestamps"]) >= 2)
    if not dates:
        raise ValueError("Dataset contains no complete days with at least two snapshots")
    train_dates, validation_dates, test_dates = split_dates(
        dates, settings.TRAIN_SPLIT, settings.VAL_SPLIT, settings.TEST_SPLIT,
    )
    training = [dataset["days"][day] for day in train_dates]
    excluded_ids = {ride["id"] for day in training for ride in day["rides"] if is_excluded(ride)}
    ride_by_id = {ride["id"]: ride for day in training for ride in day["rides"]
                  if ride["id"] not in excluded_ids}
    rides = [ride_by_id[key] for key in sorted(ride_by_id)]
    if not rides:
        raise ValueError("Dataset contains no eligible attractions after exclusions")
    train_days = Days(training, rides)
    if not train_days:
        raise ValueError("Dataset needs an operating day with at least two snapshots")
    train_loader = DataLoader(train_days, batch_size=settings.BATCH_SIZE, shuffle=True,
                              pin_memory=device.type == "cuda", collate_fn=collate_days)
    validation_loader = None
    if validation_dates:
        validation_loader = DataLoader(
            Days([dataset["days"][day] for day in validation_dates], rides),
            batch_size=settings.BATCH_SIZE, shuffle=False,
            pin_memory=device.type == "cuda", collate_fn=collate_days,
        )
    model_config = {
        "num_rides": len(rides), "embedding_size": settings.EMBEDDING_SIZE,
        "hidden_size": settings.HIDDEN_SIZE,
        "architecture": getattr(settings, "MODEL_ARCHITECTURE", config.MODEL_ARCHITECTURE),
        "attention_blocks": getattr(settings, "ATTENTION_BLOCKS", config.ATTENTION_BLOCKS),
        "attention_heads": getattr(settings, "ATTENTION_HEADS", config.ATTENTION_HEADS),
        "dropout": getattr(settings, "DROPOUT", config.DROPOUT),
    }
    model = WaitPredictor(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE)
    description = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Training on {device} ({description}), {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    print(f"Whole-day chronological split: {len(train_dates)} train, "
          f"{len(validation_dates)} validation, {len(test_dates)} test; {len(rides)} rides", flush=True)
    for label, selected in (("Train", train_dates), ("Validation", validation_dates), ("Test", test_dates)):
        if selected:
            print(f"{label} dates: {', '.join(selected)}", flush=True)
    settings.CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, settings.EPOCHS + 1):
        training_loss, accuracy = epoch(model, train_loader, device, optimizer)
        if not math.isfinite(training_loss):
            raise ValueError("Training produced a nonfinite loss")
        validation_loss, validation_accuracy = None, None
        metrics = f"Epoch {number:3}: train NLL {training_loss:.4f}, token accuracy {accuracy:.1%}"
        if validation_loader is not None:
            validation_loss, validation_accuracy = epoch(model, validation_loader, device)
            if not math.isfinite(validation_loss):
                raise ValueError("Validation produced a nonfinite loss")
            metrics += f" | val NLL {validation_loss:.4f}, token accuracy {validation_accuracy:.1%}"
        print(metrics, flush=True)
        checkpoint = {
            "version": 1, "model_config": model_config, "model_state": model.state_dict(),
            "rides": rides, "token_labels": TOKEN_LABELS, "history": "full_day",
            "park_id": dataset["park_id"], "train_dates": train_dates,
            "validation_dates": validation_dates, "test_dates": test_dates,
            "split": {"method": "chronological_days", "train": settings.TRAIN_SPLIT,
                      "validation": settings.VAL_SPLIT, "test": settings.TEST_SPLIT},
            "epoch": number, "training_loss": training_loss, "training_accuracy": accuracy,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
        }
        save_checkpoint(checkpoint, settings.CHECKPOINT)
    print(f"Final checkpoint saved to {settings.CHECKPOINT}")


def main():
    try:
        train()
    except (ValueError, KeyError, FileNotFoundError, PermissionError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
