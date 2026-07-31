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
    num_requests: int = config.PPO_NUM_REQUESTS
    horizon_sec: int = config.PPO_HORIZON_SEC
    # Inclusive ranges sampled once per simulation-day / episode.
    vehicles_range: tuple[int, int] = (
        config.PPO_VEHICLES_MIN,
        config.PPO_VEHICLES_MAX,
    )
    intersections_range: tuple[int, int] = (
        config.PPO_INTERSECTIONS_MIN,
        config.PPO_INTERSECTIONS_MAX,
    )
    avg_streets_range: tuple[int, int] = (
        config.PPO_AVG_STREETS_MIN,
        config.PPO_AVG_STREETS_MAX,
    )
    checkpoint: str | None = None


def _clamp_range(lo: int, hi: int, *, minimum: int = 1) -> tuple[int, int]:
    lo = max(minimum, int(lo))
    hi = max(lo, int(hi))
    return lo, hi


def _sample_int(rng: random.Random, lo_hi: tuple[int, int]) -> int:
    lo, hi = _clamp_range(lo_hi[0], lo_hi[1])
    return rng.randint(lo, hi)


def _sample_env_scale(cfg: PPOConfig, rng: random.Random) -> dict[str, int]:
    """Sample fleet / graph scale for one simulation-day episode."""
    vehicles_range = _clamp_range(
        cfg.vehicles_range[0],
        min(cfg.vehicles_range[1], config.MAX_VEHICLES),
    )
    intersections_range = _clamp_range(
        cfg.intersections_range[0],
        min(cfg.intersections_range[1], config.MAX_NODES),
    )
    num_vehicles = _sample_int(rng, vehicles_range)
    # Day-total demand (not concurrent): prefer num_vehicles * 2 > num_requests.
    num_requests = min(cfg.num_requests, max(1, num_vehicles * 2 - 1))
    return {
        "num_vehicles": num_vehicles,
        "num_intersections": _sample_int(rng, intersections_range),
        "avg_streets_per_intersection": _sample_int(rng, cfg.avg_streets_range),
        "num_requests": num_requests,
    }


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


def _make_env(cfg: PPOConfig, seed: int, scale: dict[str, int]) -> _fleet_sim.FleetEnv:
    return _fleet_sim.FleetEnv(
        seed,
        make_env_config(
            num_intersections=scale["num_intersections"],
            num_vehicles=scale["num_vehicles"],
            num_requests=scale["num_requests"],
            horizon_sec=cfg.horizon_sec,
            avg_streets_per_intersection=scale["avg_streets_per_intersection"],
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
    # Distinct RNG per update so scale sampling is reproducible from cfg.seed.
    scale_rng = random.Random(base_seed ^ 0xC0FFEE)
    scales = [_sample_env_scale(cfg, scale_rng) for _ in range(n_envs)]
    envs = [_make_env(cfg, base_seed + i, scales[i]) for i in range(n_envs)]

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


def _fleet_config_dict(cfg: PPOConfig) -> dict:
    v_lo, v_hi = _clamp_range(*cfg.vehicles_range)
    n_lo, n_hi = _clamp_range(*cfg.intersections_range)
    s_lo, s_hi = _clamp_range(*cfg.avg_streets_range, minimum=2)
    return {
        "num_envs": cfg.num_envs,
        "num_requests": cfg.num_requests,
        "horizon_sec": cfg.horizon_sec,
        "vehicles_range": [v_lo, v_hi],
        "intersections_range": [n_lo, n_hi],
        "avg_streets_range": [s_lo, s_hi],
        # Midpoints kept for older rollout helpers that expect fixed scale.
        "num_vehicles": (v_lo + v_hi) // 2,
        "num_intersections": (n_lo + n_hi) // 2,
        "avg_streets_per_intersection": (s_lo + s_hi) // 2,
    }


def _apply_saved_fleet_config(cfg: PPOConfig, saved: dict) -> None:
    """Optionally restore env scale ranges from a checkpoint.

    Only applies explicit ``*_range`` fields. Legacy fixed ``num_vehicles`` /
    ``num_intersections`` are ignored so older checkpoints do not freeze the
    new per-episode randomization. CLI / current defaults always win for
    request baseline and horizon.
    """
    if "vehicles_range" in saved:
        lo, hi = saved["vehicles_range"]
        cfg.vehicles_range = _clamp_range(int(lo), int(hi))
    if "intersections_range" in saved:
        lo, hi = saved["intersections_range"]
        cfg.intersections_range = _clamp_range(int(lo), int(hi))
    if "avg_streets_range" in saved:
        lo, hi = saved["avg_streets_range"]
        cfg.avg_streets_range = _clamp_range(int(lo), int(hi), minimum=2)


def _checkpoint_payload(
    agent: Agent,
    optimizer: optim.Optimizer,
    *,
    update: int,
    global_steps: int,
    cfg: PPOConfig,
) -> dict:
    return {
        "model": agent.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "global_steps": global_steps,
        "config": _fleet_config_dict(cfg),
    }


def _load_training_checkpoint(
    path: Path,
    agent: Agent,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, dict]:
    """Load model (+ optimizer if present). Returns (start_update, global_steps, saved_cfg)."""
    if not path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' state_dict: {path}")

    agent.model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        _log(f"loaded optimizer state from {path}")
    else:
        _log(f"checkpoint has no optimizer state; continuing with fresh Adam")

    start_update = int(ckpt.get("update", 0))
    global_steps = int(ckpt.get("global_steps", 0))
    saved_cfg = ckpt.get("config") or {}
    return start_update, global_steps, saved_cfg


def train(cfg: PPOConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    start_update = 0
    global_steps = 0
    if cfg.checkpoint:
        start_update, global_steps, saved_cfg = _load_training_checkpoint(
            Path(cfg.checkpoint), agent, optimizer, device
        )
        _apply_saved_fleet_config(cfg, saved_cfg)
        _log(
            f"resumed from {cfg.checkpoint} "
            f"(update={start_update}, global_steps={global_steps})"
        )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    end_update = start_update + cfg.total_updates

    v_lo, v_hi = _clamp_range(*cfg.vehicles_range)
    n_lo, n_hi = _clamp_range(*cfg.intersections_range)
    s_lo, s_hi = _clamp_range(*cfg.avg_streets_range, minimum=2)
    _log(
        f"PPO start: updates={start_update}->{end_update} (+{cfg.total_updates}) "
        f"envs={cfg.num_envs} device={device} "
        f"fleet={v_lo}-{v_hi}v "
        f"day_requests=min({cfg.num_requests},2v-1) "
        f"graph={n_lo}-{n_hi}n streets_deg={s_lo}-{s_hi} "
        f"horizon={cfg.horizon_sec}s"
    )
    _log(
        f"native FLAT_OBS_DIM={_fleet_sim.FLAT_OBS_DIM} "
        f"python FLAT_OBS_DIM={config.FLAT_OBS_DIM} NUM_ACTIONS={_fleet_sim.NUM_ACTIONS}"
    )

    for update in range(start_update, end_update):
        if cfg.anneal_lr:
            frac = 1.0 - update / max(1, end_update)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * frac

        seed = cfg.seed + update * cfg.num_envs
        batch = _collect_parallel(agent, cfg, device, seed)
        stats = _ppo_update(agent, optimizer, batch, cfg)
        global_steps += batch.total_steps
        finished = update + 1

        if cfg.log_every > 0 and finished % cfg.log_every == 0:
            _log(
                f"up={finished}/{end_update} "
                f"steps/env={batch.mean_steps:.0f} total_steps={batch.total_steps} "
                f"return={batch.mean_return:.2f} "
                f"completed={batch.mean_completed:.1f} "
                f"completion={batch.mean_completion:.1%} "
                f"mean_wait={batch.mean_wait:.1f}s "
                f"pg={stats['pg_loss']:.3f} v={stats['v_loss']:.3f} "
                f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
                f"rollout={batch.rollout_sec:.1f}s"
            )

        if cfg.save_every > 0 and finished % cfg.save_every == 0:
            path = save_dir / f"ppo_up{finished}.pt"
            torch.save(
                _checkpoint_payload(
                    agent,
                    optimizer,
                    update=finished,
                    global_steps=global_steps,
                    cfg=cfg,
                ),
                path,
            )
            _log(f"saved {path}")

    final = save_dir / "ppo_final.pt"
    torch.save(
        _checkpoint_payload(
            agent,
            optimizer,
            update=end_update,
            global_steps=global_steps,
            cfg=cfg,
        ),
        final,
    )
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
    p.add_argument(
        "--updates",
        type=int,
        default=20,
        help="Number of PPO updates to run (additional when resuming).",
    )
    p.add_argument("--num-envs", type=int, default=config.PPO_NUM_ENVS)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--num-vehicles",
        type=int,
        default=None,
        help="Fix vehicle count for every episode (disables vehicle randomization).",
    )
    p.add_argument("--vehicles-min", type=int, default=config.PPO_VEHICLES_MIN)
    p.add_argument("--vehicles-max", type=int, default=config.PPO_VEHICLES_MAX)
    p.add_argument("--num-requests", type=int, default=config.PPO_NUM_REQUESTS)
    p.add_argument(
        "--num-intersections",
        type=int,
        default=None,
        help="Fix intersection count for every episode (disables graph-size randomization).",
    )
    p.add_argument(
        "--intersections-min", type=int, default=config.PPO_INTERSECTIONS_MIN
    )
    p.add_argument(
        "--intersections-max", type=int, default=config.PPO_INTERSECTIONS_MAX
    )
    p.add_argument(
        "--avg-streets",
        type=int,
        default=None,
        help="Fix avg streets/intersection (disables street-density randomization).",
    )
    p.add_argument("--avg-streets-min", type=int, default=config.PPO_AVG_STREETS_MIN)
    p.add_argument("--avg-streets-max", type=int, default=config.PPO_AVG_STREETS_MAX)
    p.add_argument("--horizon-sec", type=int, default=config.PPO_HORIZON_SEC)
    p.add_argument("--lr", type=float, default=config.PPO_LEARNING_RATE)
    p.add_argument("--save-dir", type=str, default=config.PPO_SAVE_DIR)
    p.add_argument(
        "--checkpoint",
        "--resume",
        type=str,
        default=None,
        dest="checkpoint",
        help="Load a prior checkpoint and continue training from its update count.",
    )
    args = p.parse_args()
    updates = args.episodes if args.episodes is not None else args.updates
    if args.num_vehicles is not None:
        vehicles_range = (args.num_vehicles, args.num_vehicles)
    else:
        vehicles_range = (args.vehicles_min, args.vehicles_max)
    if args.num_intersections is not None:
        intersections_range = (args.num_intersections, args.num_intersections)
    else:
        intersections_range = (args.intersections_min, args.intersections_max)
    if args.avg_streets is not None:
        avg_streets_range = (args.avg_streets, args.avg_streets)
    else:
        avg_streets_range = (args.avg_streets_min, args.avg_streets_max)
    return PPOConfig(
        seed=args.seed,
        total_updates=updates,
        num_envs=args.num_envs,
        device=args.device,
        learning_rate=args.lr,
        num_requests=args.num_requests,
        horizon_sec=args.horizon_sec,
        vehicles_range=_clamp_range(*vehicles_range),
        intersections_range=_clamp_range(*intersections_range),
        avg_streets_range=_clamp_range(*avg_streets_range, minimum=2),
        save_dir=args.save_dir,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    train(parse_args())
