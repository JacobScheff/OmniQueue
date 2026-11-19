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
    CLOSED, DOWN, REFURBISHMENT, TICK_SECONDS, TOKEN_LABELS, UNKNOWN, VOCAB_SIZE,
    align_day, load_dataset, time_features,
)
from ml.model import build_model, select_device, token_loss
from ml.availability import advance_ages, fit_breakdown_hazards


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
            inputs = tokens[:-1].clone()
            # An interpolated value may depend on a later endpoint. Do not let
            # it enter prefix encodings or pretend it is an observed target.
            interpolated = day.get("interpolated_ticks", {})
            for index, timestamp in enumerate(day["timestamps"]):
                if str(timestamp) in interpolated:
                    if index < len(inputs):
                        inputs[index] = UNKNOWN
                    if index:
                        targets[index - 1] = UNKNOWN
            # Retain earlier history across scheduled breaks, while excluding
            # transitions whose target is more than five minutes away.
            targets[np.diff(timestamps) != TICK_SECONDS] = UNKNOWN
            self.days.append((inputs, features[1:], targets))

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


def rollout_loss(model, tokens, features, targets, logits, states, *, steps, origins,
                 ordinal_weight=0.0):
    """Branch from observed prefixes and train on generated inputs, never targets.

    Prefix states share the one-step pass's graph. Sampling is discrete, while
    recurrent gradients continue through the generated sequence. An entire
    rollout must fit before padding or a scheduled break. Validation uses fixed
    origins and deterministic production decoding, so selection is repeatable.
    """
    steps = min(steps, tokens.shape[1])
    valid = ((targets >= 0) & (targets <= (CLOSED if model.predict_closed else DOWN))).any(-1)
    eligible = valid.unfold(1, steps, 1).all(-1)
    rows, starts = [], []
    for row in range(tokens.shape[0]):
        choices = eligible[row].nonzero().flatten()
        if not len(choices):
            continue
        count = min(origins, len(choices))
        if model.training:
            chosen = choices[torch.randperm(len(choices), device=tokens.device)[:count]]
        else:
            chosen = choices[torch.linspace(0, len(choices) - 1, count, device=tokens.device).long()]
        rows.append(torch.full_like(chosen, row))
        starts.append(chosen)
    if not rows:
        return logits[..., 0].sum() * 0
    rows, starts = torch.cat(rows), torch.cat(starts)
    hidden = states[:, rows, starts].contiguous()
    current = logits[rows, starts]
    last = tokens[rows, starts]
    locked = last == REFURBISHMENT
    absent = ~(tokens != UNKNOWN).long().cumsum(1)[rows, starts].bool()
    ticks = torch.arange(tokens.shape[1], device=tokens.device).view(1, -1, 1)
    last_nondown = torch.where(tokens != DOWN, ticks, -1).cummax(1).values
    ages = torch.where(last == DOWN, starts[:, None] - last_nondown[rows, starts], 0)
    survival = torch.ones_like(ages, dtype=logits.dtype)
    total = logits[..., 0].sum() * 0
    for offset in range(steps):
        loss, _ = token_loss(current, targets[rows, starts + offset],
                             predict_closed=model.predict_closed, ordinal_weight=ordinal_weight)
        total = total + loss
        if offset + 1 == steps:
            break
        with torch.no_grad():
            if model.training:
                probabilities = model.prediction_logits(current, ages).softmax(-1)
                prediction = torch.multinomial(probabilities.reshape(-1, VOCAB_SIZE), 1).reshape_as(last)
            else:
                prediction, survival = model.greedy_prediction(current, ages, survival)
            prediction[locked] = REFURBISHMENT
            prediction[absent] = UNKNOWN
            ages = advance_ages(ages, prediction)
            survival = torch.where(ages > 0, survival, 1.)
        current, hidden = model(prediction.unsqueeze(1), features[rows, starts + offset + 1].unsqueeze(1), hidden)
        current = current[:, 0]
    return total / steps


def horizon_loss(model, tokens, features, targets, states, *, horizons, origins, ordinal_weight=0.0):
    """Supervise lead queries using only the saved state at each observed origin."""
    valid = ((targets >= 0) & (targets <= (CLOSED if model.predict_closed else DOWN))).any(-1)
    rows, starts, leads = [], [], []
    for lead in horizons:
        if lead > tokens.shape[1]:
            continue
        eligible = valid.unfold(1, lead, 1).all(-1)
        for row in range(tokens.shape[0]):
            choices = eligible[row].nonzero().flatten()
            if not len(choices):
                continue
            count = min(origins, len(choices))
            if model.training:
                choices = choices[torch.randperm(len(choices), device=tokens.device)[:count]]
            else:
                choices = choices[torch.linspace(0, len(choices) - 1, count, device=tokens.device).long()]
            rows.append(torch.full_like(choices, row))
            starts.append(choices)
            leads.append(torch.full_like(choices, lead))
    if not rows:
        return states[..., 0].sum() * 0
    rows, starts, leads = torch.cat(rows), torch.cat(starts), torch.cat(leads)
    end = starts + leads - 1
    logits = model.horizon_logits(states[:, rows, starts].contiguous(), tokens[rows, starts],
                                  features[rows, end].unsqueeze(1), leads.unsqueeze(1))[:, 0]
    return token_loss(logits, targets[rows, end], predict_closed=model.predict_closed,
                      ordinal_weight=ordinal_weight)[0]


def epoch(model, loader, device, optimizer=None, *, ordinal_weight=0.0, rollout_weight=0.0,
          rollout_steps=12, rollout_origins=2, horizon_weight=0.0,
          horizons=(1, 3, 6, 12, 24, 48, 96, 144), horizon_origins=8, metrics=None):
    """Score observed-prefix and future targets; only training updates weights."""
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total_nll = torch.zeros((), device=device)
    total_rollout = torch.zeros((), device=device)
    total_horizon = torch.zeros((), device=device)
    for tokens, features, targets in loader:
        tokens = tokens.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            direct = horizon_weight and hasattr(model, "horizon_logits")
            logits, states = (model(tokens, features, return_state_sequence=True)
                              if rollout_weight or direct else model(tokens, features))
            nll, count = token_loss(logits, targets, predict_closed=model.predict_closed)
            loss = (token_loss(logits, targets, predict_closed=model.predict_closed, ordinal_weight=ordinal_weight)[0]
                    if ordinal_weight else nll)
            generated_loss = logits.new_zeros(())
            if rollout_weight:
                generated_loss = rollout_loss(model, tokens, features, targets, logits, states,
                                              steps=rollout_steps, origins=rollout_origins,
                                              ordinal_weight=ordinal_weight)
                loss = loss + rollout_weight * generated_loss
            direct_loss = logits.new_zeros(())
            if direct:
                direct_loss = horizon_loss(model, tokens, features, targets, states, horizons=horizons,
                                           origins=horizon_origins, ordinal_weight=ordinal_weight)
                loss = loss + horizon_weight * direct_loss
            if training and count.item():
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.detach() * count
        total_nll += nll.detach() * count
        total_rollout += generated_loss.detach() * count
        total_horizon += direct_loss.detach() * count
        total_count += count
        predictions = logits.argmax(dim=-1)
        limit = CLOSED if model.predict_closed else DOWN
        total_correct += ((predictions == targets) & (targets >= 0) & (targets <= limit)).sum()
    count = total_count.item()
    if count == 0:
        raise ValueError("These days contain no wait or availability targets")
    if metrics is not None:
        metrics.update(nll=total_nll.item() / count, rollout_loss=total_rollout.item() / count,
                       horizon_loss=total_horizon.item() / count)
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
    objective = {
        "ordinal_weight": getattr(settings, "ORDINAL_WEIGHT", config.ORDINAL_WEIGHT),
        "rollout_weight": getattr(settings, "ROLLOUT_WEIGHT", config.ROLLOUT_WEIGHT),
        "rollout_steps": getattr(settings, "ROLLOUT_STEPS", config.ROLLOUT_STEPS),
        "rollout_origins": getattr(settings, "ROLLOUT_ORIGINS", config.ROLLOUT_ORIGINS),
        "horizon_weight": getattr(settings, "HORIZON_WEIGHT", config.HORIZON_WEIGHT),
        "horizons": getattr(settings, "HORIZON_STEPS", config.HORIZON_STEPS),
        "horizon_origins": getattr(settings, "HORIZON_ORIGINS", config.HORIZON_ORIGINS),
    }
    warmup = getattr(settings, "ROLLOUT_WARMUP_EPOCHS", config.ROLLOUT_WARMUP_EPOCHS)
    patience = getattr(settings, "EARLY_STOPPING_PATIENCE", config.EARLY_STOPPING_PATIENCE)
    if (any(not math.isfinite(objective[key]) or objective[key] < 0
            for key in ("ordinal_weight", "rollout_weight", "horizon_weight"))
            or min(objective["rollout_steps"], objective["rollout_origins"], objective["horizon_origins"]) < 1
            or not objective["horizons"] or any(type(lead) is not int or lead < 1 for lead in objective["horizons"])
            or min(warmup, patience) < 0):
        raise ValueError("Loss weights must be finite and nonnegative; rollout sizes positive; warmup/patience nonnegative")
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
        "predict_closed": True,
        "breakdown_hazards": fit_breakdown_hazards(training, rides),
    }
    if model_config["architecture"] == "ride_horizon_v3" and not objective["horizon_weight"]:
        raise ValueError("ride_horizon_v3 requires HORIZON_WEIGHT > 0; use ride_transition_v2 for one-step training")
    model = build_model(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE)
    description = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Training on {device} ({description}), {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    print(f"Whole-day chronological split: {len(train_dates)} train, "
          f"{len(validation_dates)} validation, {len(test_dates)} test; {len(rides)} rides", flush=True)
    for label, selected in (("Train", train_dates), ("Validation", validation_dates), ("Test", test_dates)):
        if selected:
            print(f"{label} dates: {', '.join(selected)}", flush=True)
    settings.CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch = math.inf, 0
    latest = settings.CHECKPOINT.with_name(settings.CHECKPOINT.stem + ".latest" + settings.CHECKPOINT.suffix)
    for number in range(1, settings.EPOCHS + 1):
        train_metrics, validation_metrics = {}, {}
        training_objective = {**objective, "rollout_weight": objective["rollout_weight"] * min(1., number / max(1, warmup))}
        training_loss, accuracy = epoch(model, train_loader, device, optimizer,
                                       **training_objective, metrics=train_metrics)
        if not math.isfinite(training_loss):
            raise ValueError("Training produced a nonfinite loss")
        validation_loss, validation_accuracy = None, None
        metrics = f"Epoch {number:3}: train NLL {train_metrics['nll']:.4f}, token accuracy {accuracy:.1%}"
        if validation_loader is not None:
            validation_loss, validation_accuracy = epoch(model, validation_loader, device,
                                                         **objective, metrics=validation_metrics)
            if not math.isfinite(validation_loss):
                raise ValueError("Validation produced a nonfinite loss")
            metrics += (f" | val NLL {validation_metrics['nll']:.4f}, token accuracy {validation_accuracy:.1%}"
                        f" | forecast objective {validation_loss:.4f}")
        score = validation_loss if validation_loader is not None else training_loss
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, number
        metrics += f" | best {best_epoch}"
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
            "objective": {**objective, "rollout_warmup_epochs": warmup},
            "training_nll": train_metrics["nll"], "validation_nll": validation_metrics.get("nll"),
            "validation_horizon_loss": validation_metrics.get("horizon_loss"),
            "selection": "validation_forecast_objective" if validation_loader is not None else "final_epoch",
            "best_epoch": best_epoch, "best_score": best_score,
            "seed": settings.SEED,
        }
        if validation_loader is not None:
            save_checkpoint(checkpoint, latest)
        if improved or validation_loader is None:
            save_checkpoint(checkpoint, settings.CHECKPOINT)
        if validation_loader is not None and patience and number - best_epoch >= patience:
            print(f"Early stopping after {patience} epochs without validation improvement.", flush=True)
            break
    print(f"Selected checkpoint saved to {settings.CHECKPOINT}")
    if validation_loader is not None:
        print(f"Latest epoch saved to {latest}")


def main():
    try:
        train()
    except (ValueError, KeyError, FileNotFoundError, PermissionError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
