#!/usr/bin/env python3
"""Minimal PPO training loop against the C++ FleetEnv (_fleet_sim).

No heuristic / BC stage — policy starts from random weights and learns from
simulator rewards (completions + pending-wait shaping).

Supports multiple parallel envs: each update collects one episode from each
env with batched policy inference across live envs.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import _fleet_sim
import fleet.config as config
from fleet.model import VehicleRouter, forward_with_mask
from fleet.simulator import make_env_config


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class PPOConfig:
    seed: int = 42
    total_updates: int = 20
    num_envs: int = config.PPO_NUM_ENVS
    device: str = "cpu"
    learning_rate: float = config.PPO_LEARNING_RATE
    anneal_lr: bool = config.PPO_ANNEAL_LR
    gamma: float = config.PPO_GAMMA
    gae_lambda: float = config.PPO_GAE_LAMBDA
    num_minibatches: int = config.PPO_NUM_MINIBATCHES
    update_epochs: int = config.PPO_UPDATE_EPOCHS
    clip_coef: float = config.PPO_CLIP_COEF
    ent_coef: float = config.PPO_ENT_COEF
    vf_coef: float = config.PPO_VF_COEF
    max_grad_norm: float = config.PPO_MAX_GRAD_NORM
    target_kl: float = config.PPO_TARGET_KL
    max_steps: int = config.PPO_MAX_STEPS_PER_EPISODE
    save_dir: str = config.PPO_SAVE_DIR
    save_every: int = config.PPO_SAVE_EVERY
    log_every: int = config.PPO_LOG_EVERY
    num_vehicles: int = config.PPO_NUM_VEHICLES
    num_requests: int = config.PPO_NUM_REQUESTS
    num_intersections: int = config.PPO_NUM_INTERSECTIONS
    horizon_sec: int = config.PPO_HORIZON_SEC


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    # Aggregate metrics across envs
    mean_return: float
    mean_steps: float
    mean_wait: float
    mean_completion: float
    mean_completed: float
    total_steps: int
    rollout_sec: float


class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = VehicleRouter(use_graph_encoder=False)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        logits, values = forward_with_mask(self.model, obs)
        logits = logits[:, 0, :]
        values = values[:, 0, 0]
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values


def _make_env(cfg: PPOConfig, seed: int) -> _fleet_sim.FleetEnv:
    return _fleet_sim.FleetEnv(
        seed,
        make_env_config(
            num_intersections=cfg.num_intersections,
            num_vehicles=cfg.num_vehicles,
            num_requests=cfg.num_requests,
            horizon_sec=cfg.horizon_sec,
        ),
    )


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    *,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(rewards)
    advantages = torch.zeros(n, dtype=torch.float32, device=device)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    return advantages, advantages + values_t


def _collect_parallel(
    agent: Agent,
    cfg: PPOConfig,
    device: torch.device,
    base_seed: int,
) -> RolloutBatch:
    """Run ``num_envs`` episodes with batched policy forwards over live envs."""
    n_envs = max(1, cfg.num_envs)
    envs = [_make_env(cfg, base_seed + i) for i in range(n_envs)]

    obs_list: list[np.ndarray | None] = [None] * n_envs
    for i, env in enumerate(envs):
        obs = env.reset(base_seed + i)
        flat = np.asarray(obs.flat(), dtype=np.float32)
        if flat.shape[0] != config.FLAT_OBS_DIM:
            raise RuntimeError(
                f"FLAT_OBS_DIM mismatch: python={config.FLAT_OBS_DIM} "
                f"native={flat.shape[0]} (rebuild _fleet_sim)"
            )
        obs_list[i] = flat

    # Per-env trajectory buffers
    ep_obs: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
    ep_actions: list[list[int]] = [[] for _ in range(n_envs)]
    ep_logprobs: list[list[float]] = [[] for _ in range(n_envs)]
    ep_rewards: list[list[float]] = [[] for _ in range(n_envs)]
    ep_values: list[list[float]] = [[] for _ in range(n_envs)]
    ep_dones: list[list[float]] = [[] for _ in range(n_envs)]
    ep_return = [0.0] * n_envs
    ep_steps = [0] * n_envs
    ep_metrics = [e.metrics for e in envs]
    alive = [True] * n_envs

    t0 = time.perf_counter()

    while any(alive):
        live = [i for i in range(n_envs) if alive[i] and ep_steps[i] < cfg.max_steps]
        if not live:
            break

        batch_obs = np.stack([obs_list[i] for i in live], axis=0)
        obs_t = torch.as_tensor(batch_obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, logprobs, _, values = agent.get_action_and_value(obs_t)

        actions_np = actions.detach().cpu().numpy()
        logprobs_np = logprobs.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()

        for row, i in enumerate(live):
            a = int(actions_np[row])
            result = envs[i].step(a)

            ep_obs[i].append(obs_list[i].copy())
            ep_actions[i].append(a)
            ep_logprobs[i].append(float(logprobs_np[row]))
            ep_values[i].append(float(values_np[row]))
            ep_rewards[i].append(float(result.reward))
            ep_return[i] += float(result.reward)
            ep_steps[i] += 1
            done = bool(result.done) or ep_steps[i] >= cfg.max_steps
            ep_dones[i].append(1.0 if done else 0.0)
            ep_metrics[i] = result.metrics

            if done:
                alive[i] = False
            elif result.has_obs:
                obs_list[i] = np.asarray(result.obs.flat(), dtype=np.float32)
            else:
                alive[i] = False

    # GAE per episode, then concatenate (do not stitch across env boundaries).
    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    all_logprobs: list[float] = []
    all_adv: list[torch.Tensor] = []
    all_ret: list[torch.Tensor] = []

    for i in range(n_envs):
        if not ep_obs[i]:
            raise RuntimeError(f"Env {i} collected zero decision steps.")
        adv, ret = _compute_gae(
            ep_rewards[i],
            ep_values[i],
            ep_dones[i],
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=device,
        )
        all_obs.extend(ep_obs[i])
        all_actions.extend(ep_actions[i])
        all_logprobs.extend(ep_logprobs[i])
        all_adv.append(adv)
        all_ret.append(ret)

    return RolloutBatch(
        obs=torch.tensor(np.stack(all_obs), dtype=torch.float32, device=device),
        actions=torch.tensor(all_actions, dtype=torch.long, device=device),
        logprobs=torch.tensor(all_logprobs, dtype=torch.float32, device=device),
        advantages=torch.cat(all_adv, dim=0),
        returns=torch.cat(all_ret, dim=0),
        mean_return=float(np.mean(ep_return)),
        mean_steps=float(np.mean(ep_steps)),
        mean_wait=float(np.mean([m.mean_wait() for m in ep_metrics])),
        mean_completion=float(np.mean([m.completion_rate() for m in ep_metrics])),
        mean_completed=float(np.mean([m.requests_completed for m in ep_metrics])),
        total_steps=int(sum(ep_steps)),
        rollout_sec=time.perf_counter() - t0,
    )


def _ppo_update(
    agent: Agent,
    optimizer: optim.Optimizer,
    batch: RolloutBatch,
    cfg: PPOConfig,
) -> dict[str, float]:
    n = batch.obs.shape[0]
    mb_size = max(1, n // max(1, cfg.num_minibatches))
    indices = np.arange(n)

    adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    last = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}

    for _ in range(cfg.update_epochs):
        np.random.shuffle(indices)
        early_stop = False
        for start in range(0, n, mb_size):
            mb_idx = indices[start : start + mb_size]
            mb_obs = batch.obs[mb_idx]
            mb_actions = batch.actions[mb_idx]
            mb_logprobs = batch.logprobs[mb_idx]
            mb_adv = adv[mb_idx]
            mb_returns = batch.returns[mb_idx]

            _, new_logprob, entropy, new_value = agent.get_action_and_value(
                mb_obs, mb_actions
            )
            log_ratio = new_logprob - mb_logprobs
            ratio = log_ratio.exp()

            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(
                ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
            last = {
                "pg_loss": float(pg_loss.item()),
                "v_loss": float(v_loss.item()),
                "entropy": float(entropy_loss.item()),
                "approx_kl": float(approx_kl),
                "clipfrac": float(clipfrac),
            }
            if cfg.target_kl > 0 and approx_kl > cfg.target_kl:
                early_stop = True
                break
        if early_stop:
            break
    return last


def train(cfg: PPOConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"PPO start: updates={cfg.total_updates} envs={cfg.num_envs} device={device} "
        f"fleet={cfg.num_vehicles}v/{cfg.num_requests}r "
        f"graph={cfg.num_intersections}n horizon={cfg.horizon_sec}s"
    )
    _log(
        f"native FLAT_OBS_DIM={_fleet_sim.FLAT_OBS_DIM} "
        f"python FLAT_OBS_DIM={config.FLAT_OBS_DIM} NUM_ACTIONS={_fleet_sim.NUM_ACTIONS}"
    )

    global_steps = 0
    for update in range(cfg.total_updates):
        if cfg.anneal_lr:
            frac = 1.0 - update / max(1, cfg.total_updates)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * frac

        seed = cfg.seed + update * cfg.num_envs
        batch = _collect_parallel(agent, cfg, device, seed)
        stats = _ppo_update(agent, optimizer, batch, cfg)
        global_steps += batch.total_steps

        if cfg.log_every > 0 and (update + 1) % cfg.log_every == 0:
            _log(
                f"up={update + 1}/{cfg.total_updates} "
                f"steps/env={batch.mean_steps:.0f} total_steps={batch.total_steps} "
                f"return={batch.mean_return:.2f} "
                f"completed={batch.mean_completed:.1f} "
                f"completion={batch.mean_completion:.1%} "
                f"mean_wait={batch.mean_wait:.1f}s "
                f"pg={stats['pg_loss']:.3f} v={stats['v_loss']:.3f} "
                f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
                f"rollout={batch.rollout_sec:.1f}s"
            )

        if cfg.save_every > 0 and (update + 1) % cfg.save_every == 0:
            path = save_dir / f"ppo_up{update + 1}.pt"
            torch.save(
                {
                    "model": agent.model.state_dict(),
                    "update": update + 1,
                    "global_steps": global_steps,
                    "config": {
                        "num_envs": cfg.num_envs,
                        "num_vehicles": cfg.num_vehicles,
                        "num_requests": cfg.num_requests,
                        "num_intersections": cfg.num_intersections,
                        "horizon_sec": cfg.horizon_sec,
                    },
                },
                path,
            )
            _log(f"saved {path}")

    final = save_dir / "ppo_final.pt"
    torch.save({"model": agent.model.state_dict(), "update": cfg.total_updates}, final)
    _log(f"done. saved {final} (total decision steps={global_steps})")


def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser(description="Fleet PPO (C++ sim, no heuristics)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Alias for --updates (kept for CLI compatibility).",
    )
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--num-envs", type=int, default=config.PPO_NUM_ENVS)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-vehicles", type=int, default=config.PPO_NUM_VEHICLES)
    p.add_argument("--num-requests", type=int, default=config.PPO_NUM_REQUESTS)
    p.add_argument("--num-intersections", type=int, default=config.PPO_NUM_INTERSECTIONS)
    p.add_argument("--horizon-sec", type=int, default=config.PPO_HORIZON_SEC)
    p.add_argument("--lr", type=float, default=config.PPO_LEARNING_RATE)
    p.add_argument("--save-dir", type=str, default=config.PPO_SAVE_DIR)
    args = p.parse_args()
    updates = args.episodes if args.episodes is not None else args.updates
    return PPOConfig(
        seed=args.seed,
        total_updates=updates,
        num_envs=args.num_envs,
        device=args.device,
        learning_rate=args.lr,
        num_vehicles=args.num_vehicles,
        num_requests=args.num_requests,
        num_intersections=args.num_intersections,
        horizon_sec=args.horizon_sec,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    train(parse_args())
