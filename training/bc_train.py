#!/usr/bin/env python3
"""Phase 2: behavioral cloning from the C++ heuristic router."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import _park_sim
import config
from training.checkpoint import default_model, save_checkpoint


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


class BCDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        obs = sample.obs
        guest = torch.from_numpy(np.asarray(obs.guest, dtype=np.float32))
        ride = torch.from_numpy(np.asarray(obs.ride, dtype=np.float32))
        env = torch.from_numpy(np.asarray(obs.env, dtype=np.float32))
        action = torch.tensor(sample.action, dtype=torch.long)
        return guest, ride, env, action


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

    dataset = BCDataset(samples)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = default_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        batches = 0
        for guest, ride, env, action in loader:
            guest = guest.unsqueeze(1).to(device)
            ride = ride.to(device)
            env = env.to(device)
            action = action.to(device)

            logits, _ = model(guest, ride, env)
            loss = F.cross_entropy(logits[:, 0, :], action)

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
