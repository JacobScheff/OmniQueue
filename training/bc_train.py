#!/usr/bin/env python3
"""Phase 2: behavioral cloning from the C++ heuristic router."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import _park_sim
import config
from model import ParkRouterModel
from training.checkpoint import default_model, save_checkpoint
from training.features import (
    ENV_DYNAMIC_FEAT_DIM,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
    build_action_mask,
    masked_cross_entropy,
)


@dataclass
class BCConfig:
    """Behavioral cloning run configuration.

    Runtime parameters are set via CLI; hyperparameters default to ``config.py``.
    """

    seed: int = 42
    bc_days: int = 1
    device: str = "cpu"
    # Hyperparameters — edit config.py to change defaults
    epochs: int = config.BC_EPOCHS
    batch_size: int = config.BC_BATCH_SIZE
    lr: float = config.BC_LR
    save_dir: str = config.BC_SAVE_DIR
    save_every: int = config.BC_SAVE_EVERY


@dataclass
class WaveSample:
    guest: np.ndarray  # (G, guest_feat)
    ride: np.ndarray  # (G, R, ride_feat)
    env: np.ndarray  # (env_feat,) — shared park context from first party
    actions: np.ndarray  # (G,)


class BCWaveDataset(Dataset):
    """Each item is a co-timed routing wave (G >= 1 parties)."""

    def __init__(self, waves: list[WaveSample]):
        self.waves = waves

    def __len__(self) -> int:
        return len(self.waves)

    def __getitem__(self, idx: int) -> WaveSample:
        return self.waves[idx]


def _collate_waves(
    batch: list[WaveSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad waves in a minibatch to a common guest count G_max."""
    max_g = max(w.guest.shape[0] for w in batch)
    bsz = len(batch)
    guest = torch.zeros(bsz, max_g, GUEST_FEAT_DIM, dtype=torch.float32)
    ride = torch.zeros(bsz, max_g, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM, dtype=torch.float32)
    env = torch.zeros(bsz, ENV_DYNAMIC_FEAT_DIM, dtype=torch.float32)
    actions = torch.zeros(bsz, max_g, dtype=torch.long)
    padding = torch.zeros(bsz, max_g, dtype=torch.bool)

    for i, wave in enumerate(batch):
        g = wave.guest.shape[0]
        guest[i, :g] = torch.from_numpy(wave.guest)
        ride[i, :g] = torch.from_numpy(wave.ride)
        env[i] = torch.from_numpy(wave.env)
        actions[i, :g] = torch.from_numpy(wave.actions)
        padding[i, :g] = True
    return guest, ride, env, actions, padding


def samples_to_waves(samples) -> list[WaveSample]:
    """Group flat BC samples by wave_id into multi-party waves."""
    buckets: dict[int, list] = defaultdict(list)
    for sample in samples:
        buckets[int(sample.wave_id)].append(sample)

    waves: list[WaveSample] = []
    for wave_id in sorted(buckets.keys()):
        group = buckets[wave_id]
        guests = np.stack([np.asarray(s.obs.guest, dtype=np.float32) for s in group], axis=0)
        rides = np.stack([np.asarray(s.obs.ride, dtype=np.float32) for s in group], axis=0)
        env = np.asarray(group[0].obs.env, dtype=np.float32)
        actions = np.asarray([s.action for s in group], dtype=np.int64)
        waves.append(WaveSample(guest=guests, ride=rides, env=env, actions=actions))
    return waves


def collect_samples(num_days: int, seed: int) -> list:
    print(f"Collecting BC data from {num_days} heuristic day(s)...", flush=True)
    t0 = time.perf_counter()
    samples = _park_sim.collect_bc_dataset(num_days, seed)
    print(f"Collected {len(samples)} samples in {time.perf_counter() - t0:.1f}s", flush=True)
    return samples


def train(cfg: BCConfig) -> None:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    samples = collect_samples(cfg.bc_days, cfg.seed)
    if not samples:
        raise RuntimeError("No BC samples collected.")

    waves = samples_to_waves(samples)
    print(
        f"Grouped into {len(waves)} co-timed waves "
        f"(mean G={np.mean([w.guest.shape[0] for w in waves]):.1f})",
        flush=True,
    )
    dataset = BCWaveDataset(waves)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_waves,
    )

    model: ParkRouterModel = default_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        batches = 0
        for guest, ride, env, action, padding in loader:
            guest = guest.to(device)
            ride = ride.to(device)
            env = env.to(device)
            action = action.to(device)
            padding = padding.to(device)

            logits, _ = model(guest, ride, env, guest_padding_mask=padding)
            action_mask = build_action_mask(guest, ride, env)
            action_mask = action_mask & padding.unsqueeze(-1)
            loss = masked_cross_entropy(logits, action, action_mask, padding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1
            global_step += 1

            if global_step % cfg.save_every == 0:
                ckpt = save_dir / f"bc_step_{global_step}.pt"
                save_checkpoint(ckpt, model, optimizer, global_step, {"phase": "bc", "epoch": epoch})
                print(f"Saved checkpoint: {ckpt}", flush=True)

        avg_loss = epoch_loss / max(1, batches)
        print(f"Epoch {epoch + 1}/{cfg.epochs}  loss={avg_loss:.4f}", flush=True)

    final_path = save_dir / "bc_final.pt"
    save_checkpoint(final_path, model, optimizer, global_step, {"phase": "bc", "epochs": cfg.epochs})
    print(f"Training complete. Final checkpoint: {final_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral cloning for ParkRouterModel. "
        "Hyperparameters (epochs, lr, batch-size, etc.) are set in config.py."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc-days", type=int, default=1, help="Heuristic days to mine for labels")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = BCConfig(seed=args.seed, bc_days=args.bc_days, device=args.device)
    train(cfg)


if __name__ == "__main__":
    main()
