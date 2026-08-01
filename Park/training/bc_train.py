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
_PARENT = ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import _park_sim
import Park.config as config
from Park.model import ParkRouterModel, obs_flat_to_tensors
from Park.training.checkpoint import default_model, save_checkpoint
from Park.training.features import build_action_mask, masked_cross_entropy


@dataclass
class BCConfig:
    """Behavioral cloning run configuration.

    Runtime parameters are set via CLI; hyperparameters default to ``config.py``.
    """

    seed: int = 42
    bc_days: int = 1
    device: str = "cpu"
    epochs: int = config.BC_EPOCHS
    batch_size: int = config.BC_BATCH_SIZE
    lr: float = config.BC_LR
    save_dir: str = config.BC_SAVE_DIR
    save_every: int = config.BC_SAVE_EVERY


class BCFlatDataset(Dataset):
    """Each item is one single-party routing decision."""

    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        self.obs = obs
        self.actions = actions

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self.obs[idx], int(self.actions[idx])


def collect_arrays(num_days: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    print(f"Collecting BC data from {num_days} heuristic day(s)...", flush=True)
    t0 = time.perf_counter()
    samples = _park_sim.collect_bc_dataset(num_days, seed)
    print(f"Collected {len(samples)} samples in {time.perf_counter() - t0:.1f}s", flush=True)
    obs = np.stack(
        [np.asarray(s.obs.flat(), dtype=np.float32) for s in samples],
        axis=0,
    )
    actions = np.asarray([int(s.action) for s in samples], dtype=np.int64)
    return obs, actions


def train(cfg: BCConfig) -> None:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs, actions = collect_arrays(cfg.bc_days, cfg.seed)
    if obs.shape[0] == 0:
        raise RuntimeError("No BC samples collected.")

    dataset = BCFlatDataset(obs, actions)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model: ParkRouterModel = default_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        batches = 0
        for obs_b, action_b in loader:
            obs_b = obs_b.to(device=device, dtype=torch.float32)
            action_b = action_b.to(device=device, dtype=torch.long)
            guest, ride, env = obs_flat_to_tensors(obs_b)
            logits, _ = model(guest, ride, env)
            action_mask = build_action_mask(guest, ride, env)
            loss = masked_cross_entropy(logits, action_b, action_mask)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite BC loss at step {global_step}: {float(loss.detach().cpu())}"
                )

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
