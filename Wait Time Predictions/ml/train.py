"""Train the wait predictor on every day in the dataset using settings in config.py."""

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config
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


def epoch(model, loader, device, optimizer):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    for tokens, features, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, _ = model(tokens, features)
        loss, count = token_loss(logits, targets)
        if count.item():
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.detach() * count
        total_count += count
        total_correct += ((logits.argmax(dim=-1) == targets) & (targets <= DOWN)).sum()
    count = total_count.item()
    if count == 0:
        raise ValueError("These days contain no wait or DOWN targets")
    return total_loss.item() / count, total_correct.item() / count


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
    dates = sorted(dataset["days"])
    if not dates:
        raise ValueError("Dataset contains no complete days")
    training = [dataset["days"][day] for day in dates]
    ride_by_id = {ride["id"]: ride for day in training for ride in day["rides"]}
    rides = [ride_by_id[key] for key in sorted(ride_by_id)]
    train_days = Days(training, rides)
    if not train_days:
        raise ValueError("Dataset needs an operating day with at least two snapshots")
    train_loader = DataLoader(train_days, batch_size=settings.BATCH_SIZE, shuffle=True,
                              pin_memory=device.type == "cuda", collate_fn=collate_days)
    model_config = {"num_rides": len(rides), "embedding_size": settings.EMBEDDING_SIZE, "hidden_size": settings.HIDDEN_SIZE}
    model = WaitPredictor(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE)
    description = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Training on {device} ({description}), {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    print(f"{len(train_days)} full-day training sequences, {len(rides)} rides", flush=True)
    settings.CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, settings.EPOCHS + 1):
        training_loss, accuracy = epoch(model, train_loader, device, optimizer)
        if not math.isfinite(training_loss):
            raise ValueError("Training produced a nonfinite loss")
        print(f"Epoch {number:3}: train NLL {training_loss:.4f}, token accuracy {accuracy:.1%}", flush=True)
        checkpoint = {
            "version": 1, "model_config": model_config, "model_state": model.state_dict(),
            "rides": rides, "token_labels": TOKEN_LABELS, "history": "full_day",
            "park_id": dataset["park_id"], "train_dates": dates,
            "epoch": number, "training_loss": training_loss,
        }
        temporary = settings.CHECKPOINT.with_suffix(settings.CHECKPOINT.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(settings.CHECKPOINT)
    print(f"Final checkpoint saved to {settings.CHECKPOINT}")


def main():
    try:
        train()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
