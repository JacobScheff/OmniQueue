#!/usr/bin/env python3
"""Phase 3: PPO fine-tuning for a personal next-ride planner (N focals / day)."""

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
import torch.nn as nn
import torch.optim as optim

import _park_sim
import Park.config as config
from Park.model import forward_with_mask, obs_flat_to_tensors
from Park.training.checkpoint import default_model, load_checkpoint, save_checkpoint
from Park.training.features import (
    FLAT_OBS_DIM,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_personal_api() -> None:
    if hasattr(_park_sim.ParkEnv, "reset_personal") and hasattr(
        _park_sim.ParkEnv, "exchange_batch"
    ):
        return
    raise RuntimeError(
        "This PPO script requires a rebuilt native extension with "
        "ParkEnv.reset_personal / exchange_batch.\n"
        "From the Park/ directory, rebuild: pip install -e ."
    )


def _format_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    return f"cuda ({name}, sm_{cap[0]}{cap[1]})"


def _park_time_label(obs: np.ndarray) -> str:
    env_offset = GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
    frac = float(obs[env_offset])
    hour = 8.0 + frac * 15.0
    hours = int(hour)
    minutes = int(round((hour - hours) * 60.0))
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}"


@dataclass
class PPOConfig:
    seed: int = 42
    total_days: int = 20
    num_envs: int = 1
    device: str = "cpu"
    init_checkpoint: str | None = None
    anneal_lr: bool = config.PPO_ANNEAL_LR
    learning_rate: float = config.PPO_LEARNING_RATE
    gamma: float = config.PPO_GAMMA
    gae_lambda: float = config.PPO_GAE_LAMBDA
    num_minibatches: int = config.PPO_NUM_MINIBATCHES
    update_epochs: int = config.PPO_UPDATE_EPOCHS
    clip_coef: float = config.PPO_CLIP_COEF
    ent_coef: float = config.PPO_ENT_COEF
    vf_coef: float = config.PPO_VF_COEF
    max_grad_norm: float = config.PPO_MAX_GRAD_NORM
    subsample_size: int = config.PPO_SUBSAMPLE_SIZE
    max_routing_steps: int = config.PPO_MAX_ROUTING_STEPS
    inference_batch_size: int = config.PPO_INFERENCE_BATCH_SIZE
    num_focals: int = config.PPO_NUM_FOCALS
    update_mb_size: int = getattr(config, "PPO_UPDATE_MB_SIZE", 256)
    update_yield_sec: float = getattr(config, "PPO_UPDATE_YIELD_SEC", 0.05)
    save_dir: str = config.PPO_SAVE_DIR
    save_every: int = config.PPO_SAVE_EVERY
    log_every: int = config.PPO_LOG_EVERY


@dataclass
class EpisodeBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    routing_steps: int
    episode_return: float
    avg_wait_variance: float
    rides_completed: int
    rides_per_party: float
    must_do_completion_rate: float
    avg_preference_score_per_guest: float
    avg_must_do_latency_sec: float
    rollout_sec: float
    n_focals: int


@dataclass
class PPOStats:
    pg_loss: float
    v_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    update_sec: float


class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = default_model("cpu")

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        guest, ride, env = obs_flat_to_tensors(obs)
        _, value, _ = forward_with_mask(self.model, guest, ride, env)
        return value.flatten()

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        guest, ride, env = obs_flat_to_tensors(obs)
        logits, value, _ = forward_with_mask(self.model, guest, ride, env)
        value = value.flatten()
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def _collect_episode(
    agent: Agent,
    cfg: PPOConfig,
    device: torch.device,
    seed: int,
    *,
    day_label: str = "",
    env_label: str = "",
) -> EpisodeBatch:
    """Run one personal-planner day: N focals + heuristic crowd."""
    env = _park_sim.ParkEnv(seed)
    env.reset_personal(seed, cfg.num_focals)

    obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    logprob_buf: list[float] = []
    reward_buf: list[float] = []
    value_buf: list[float] = []
    done_buf: list[float] = []
    party_buf: list[int] = []

    episode_return = 0.0
    routing_steps = 0
    rollout_t0 = time.perf_counter()
    last_log_step = 0
    prefix = " ".join(part for part in (day_label, env_label) if part)

    pending_actions: list[int] = []
    staged_obs: np.ndarray | None = None
    staged_actions: np.ndarray | None = None
    staged_logprobs: np.ndarray | None = None
    staged_values: np.ndarray | None = None
    staged_parties: np.ndarray | None = None
    result = None

    def _record_rewards(rewards_arr: np.ndarray, terminal: bool) -> None:
        nonlocal routing_steps, episode_return, last_log_step
        if (
            staged_obs is None
            or staged_actions is None
            or staged_logprobs is None
            or staged_values is None
            or staged_parties is None
        ):
            raise RuntimeError("Rollout batch state missing staged transitions.")

        for i in range(len(rewards_arr)):
            obs_row = staged_obs[i]
            party_id = int(staged_parties[i])
            obs_buf.append(obs_row.copy())
            action_buf.append(int(staged_actions[i]))
            logprob_buf.append(float(staged_logprobs[i]))
            value_buf.append(float(staged_values[i]))
            reward_buf.append(float(rewards_arr[i]))
            party_buf.append(party_id)
            episode_return += float(rewards_arr[i])
            routing_steps += 1
            # End of day, or next transition belongs to a different focal.
            is_last = i == len(rewards_arr) - 1
            next_party = (
                int(staged_parties[i + 1])
                if (not is_last)
                else (-1 if terminal else party_id)
            )
            done = 1.0 if terminal and is_last else (0.0 if next_party == party_id else 1.0)
            done_buf.append(done)

            if cfg.log_every > 0 and routing_steps - last_log_step >= cfg.log_every:
                elapsed = time.perf_counter() - rollout_t0
                steps_per_sec = routing_steps / max(elapsed, 1e-6)
                remaining = max(cfg.max_routing_steps - routing_steps, 0)
                eta_sec = remaining / max(steps_per_sec, 1e-6)
                env_offset = GUEST_FEAT_DIM + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
                mean_wait_min = float(obs_row[env_offset + 1]) * 60.0
                _log(
                    f"{prefix} rollout progress: steps={routing_steps:,} "
                    f"park_time={_park_time_label(obs_row)} return={episode_return:.2f} "
                    f"mean_wait={mean_wait_min:.0f}m "
                    f"speed={steps_per_sec:,.0f} steps/s eta={eta_sec / 60.0:.1f}m"
                )
                last_log_step = routing_steps

    episode_done = False
    while routing_steps < cfg.max_routing_steps and not episode_done:
        remaining = cfg.max_routing_steps - routing_steps
        if remaining <= 0:
            break

        result = env.exchange_batch(pending_actions, cfg.inference_batch_size)
        pending_actions = []

        if result.n_rewards > 0:
            rewards_arr = np.asarray(result.rewards, dtype=np.float32)
            _record_rewards(rewards_arr, result.episode_done)

        if result.episode_done:
            episode_done = True
            break

        if result.n_obs <= 0:
            break

        obs_np = np.asarray(result.obs, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, FLAT_OBS_DIM)
        parties_np = np.asarray(result.party_ids, dtype=np.int64)
        if obs_np.shape[0] > remaining:
            obs_np = obs_np[:remaining]
            parties_np = parties_np[:remaining]

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, logprobs, _, values = agent.get_action_and_value(obs_t)

        staged_obs = obs_np
        staged_actions = actions.detach().cpu().numpy()
        staged_logprobs = logprobs.detach().cpu().numpy()
        staged_values = values.detach().cpu().numpy()
        staged_parties = parties_np
        pending_actions = [int(a) for a in staged_actions.tolist()]

    if not obs_buf:
        raise RuntimeError("Episode collected zero routing steps.")

    rollout_sec = time.perf_counter() - rollout_t0

    bootstrap_value = 0.0
    if not episode_done and staged_values is not None and len(staged_values) > 0:
        bootstrap_value = float(staged_values[0])

    # Mark trajectory breaks when focal party_id changes across recorded rows.
    for i in range(len(done_buf) - 1):
        if party_buf[i] != party_buf[i + 1]:
            done_buf[i] = 1.0

    rewards_t = torch.tensor(reward_buf, dtype=torch.float32, device=device)
    values_t = torch.tensor(value_buf, dtype=torch.float32, device=device)
    dones_t = torch.tensor(done_buf, dtype=torch.float32, device=device)
    advantages_t, returns_t = _compute_gae(
        rewards_t,
        values_t,
        dones_t,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        bootstrap_value=bootstrap_value,
    )

    obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
    actions_t = torch.tensor(action_buf, dtype=torch.long, device=device)
    logprobs_t = torch.tensor(logprob_buf, dtype=torch.float32, device=device)

    n = obs_t.shape[0]
    if cfg.subsample_size > 0 and n > cfg.subsample_size:
        idx = torch.randperm(n, device=device)[: cfg.subsample_size]
        obs_t = obs_t[idx]
        actions_t = actions_t[idx]
        logprobs_t = logprobs_t[idx]
        advantages_t = advantages_t[idx]
        returns_t = returns_t[idx]

    personal = env.personal_stats()
    metrics = getattr(result, "metrics", None) if (episode_done and result is not None) else None
    avg_wait = float(metrics.avg_wait_variance()) if metrics is not None else 0.0
    n_focals = int(personal.n_focals)
    rides = int(personal.rides_completed)
    return EpisodeBatch(
        obs=obs_t,
        actions=actions_t,
        logprobs=logprobs_t,
        advantages=advantages_t,
        returns=returns_t,
        routing_steps=routing_steps,
        episode_return=episode_return,
        avg_wait_variance=avg_wait,
        rides_completed=rides,
        rides_per_party=(rides / max(n_focals, 1)),
        must_do_completion_rate=float(personal.must_do_completion_rate),
        avg_preference_score_per_guest=float(personal.avg_preference_score_per_guest),
        avg_must_do_latency_sec=0.0,
        rollout_sec=rollout_sec,
        n_focals=n_focals,
    )


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - float(dones[t].item())
            next_value = 0.0 if next_nonterminal == 0.0 else bootstrap_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _ppo_update(
    agent: Agent,
    optimizer: optim.Optimizer,
    batch: EpisodeBatch,
    cfg: PPOConfig,
) -> PPOStats:
    n = batch.obs.shape[0]
    mb_size = max(1, int(cfg.update_mb_size))
    yield_sec = float(getattr(cfg, "update_yield_sec", 0.0) or 0.0)
    mb_adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    last_pg_loss = 0.0
    last_v_loss = 0.0
    last_entropy = 0.0
    last_approx_kl = 0.0
    last_clipfrac = 0.0
    update_t0 = time.perf_counter()
    steps_per_epoch = max(1, (n + mb_size - 1) // mb_size)
    total_mb = cfg.update_epochs * steps_per_epoch
    mb_done = 0
    last_log_t = update_t0
    device = batch.obs.device

    for epoch in range(cfg.update_epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, mb_size):
            idx = order[start : start + mb_size]
            obs = batch.obs.index_select(0, idx)
            actions = batch.actions.index_select(0, idx)
            old_logprob = batch.logprobs.index_select(0, idx)
            adv = mb_adv.index_select(0, idx)
            ret = batch.returns.index_select(0, idx)

            _, new_logprob, entropy, new_value = agent.get_action_and_value(obs, actions)
            logratio = new_logprob - old_logprob
            ratio = logratio.exp()

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((new_value - ret) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

            last_pg_loss = float(pg_loss.item())
            last_v_loss = float(v_loss.item())
            last_entropy = float(entropy_loss.item())
            last_approx_kl = float(approx_kl.item())
            last_clipfrac = float(clipfrac.item())

            mb_done += 1
            now = time.perf_counter()
            if now - last_log_t >= 15.0 or mb_done == total_mb:
                _log(
                    f"  PPO update progress: epoch={epoch + 1}/{cfg.update_epochs} "
                    f"mb={mb_done}/{total_mb} "
                    f"({100.0 * mb_done / max(total_mb, 1):.0f}%) "
                    f"elapsed={now - update_t0:.1f}s"
                )
                last_log_t = now

            if yield_sec > 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                time.sleep(yield_sec)

    return PPOStats(
        pg_loss=last_pg_loss,
        v_loss=last_v_loss,
        entropy=last_entropy,
        approx_kl=last_approx_kl,
        clipfrac=last_clipfrac,
        update_sec=time.perf_counter() - update_t0,
    )


def train(cfg: PPOConfig) -> None:
    _require_personal_api()
    device = torch.device(cfg.device)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    num_params = sum(p.numel() for p in agent.parameters())

    global_step = 0
    init_extra: dict = {}
    if cfg.init_checkpoint:
        loaded_model, global_step, init_extra = load_checkpoint(cfg.init_checkpoint, device)
        agent.model.load_state_dict(loaded_model.state_dict())
        _log(f"Loaded init checkpoint: {cfg.init_checkpoint} (prior step={global_step})")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _log(f"PyTorch {torch.__version__} on {_format_device(device)}")
    _log(f"Model parameters: {num_params:,}")
    _log(
        f"PPO personal mode: focals={cfg.num_focals}, target_days={cfg.total_days}, "
        f"num_envs={cfg.num_envs}, subsample={cfg.subsample_size}, "
        f"inference_batch={cfg.inference_batch_size}, update_mb={cfg.update_mb_size}"
    )
    _log(
        "Primary metrics (focals only): must_do rate, pref/guest; "
        "wait_var is diagnostic only"
    )
    if init_extra:
        _log(f"Init checkpoint metadata: {init_extra}")

    days_done = 0
    update = 0
    last_save_step = 0
    best_must_do_rate = -1.0
    day_durations: list[float] = []

    while days_done < cfg.total_days:
        update += 1
        t0 = time.perf_counter()
        envs_this_update = min(cfg.num_envs, cfg.total_days - days_done)
        day_num = days_done + 1
        day_label = f"[day {day_num}/{cfg.total_days}]"

        _log(
            f"{day_label} starting personal rollout "
            f"(seed={cfg.seed + days_done}, focals={cfg.num_focals})..."
        )

        episodes: list[EpisodeBatch] = []
        for i in range(envs_this_update):
            env_label = f"[env {i + 1}/{envs_this_update}]" if envs_this_update > 1 else ""
            episodes.append(
                _collect_episode(
                    agent,
                    cfg,
                    device,
                    cfg.seed + days_done + i,
                    day_label=day_label,
                    env_label=env_label,
                )
            )

        rollout_sec = sum(ep.rollout_sec for ep in episodes)
        raw_steps = sum(ep.routing_steps for ep in episodes)
        for i, ep in enumerate(episodes):
            env_label = f" env={i + 1}" if envs_this_update > 1 else ""
            _log(
                f"{day_label}{env_label} rollout done: steps={ep.routing_steps:,} "
                f"focals={ep.n_focals} return={ep.episode_return:.2f} "
                f"must_do={ep.must_do_completion_rate:.3f} "
                f"pref/guest={ep.avg_preference_score_per_guest:.4f} "
                f"rides={ep.rides_completed:,} rides/focal={ep.rides_per_party:.2f} "
                f"time={ep.rollout_sec:.1f}s "
                f"({ep.routing_steps / max(ep.rollout_sec, 1e-6):,.0f} steps/s)"
            )

        batch = EpisodeBatch(
            obs=torch.cat([ep.obs for ep in episodes], dim=0),
            actions=torch.cat([ep.actions for ep in episodes], dim=0),
            logprobs=torch.cat([ep.logprobs for ep in episodes], dim=0),
            advantages=torch.cat([ep.advantages for ep in episodes], dim=0),
            returns=torch.cat([ep.returns for ep in episodes], dim=0),
            routing_steps=raw_steps,
            episode_return=float(np.mean([ep.episode_return for ep in episodes])),
            avg_wait_variance=float(np.mean([ep.avg_wait_variance for ep in episodes])),
            rides_completed=int(np.mean([ep.rides_completed for ep in episodes])),
            rides_per_party=float(np.mean([ep.rides_per_party for ep in episodes])),
            must_do_completion_rate=float(
                np.mean([ep.must_do_completion_rate for ep in episodes])
            ),
            avg_preference_score_per_guest=float(
                np.mean([ep.avg_preference_score_per_guest for ep in episodes])
            ),
            avg_must_do_latency_sec=float(
                np.mean([ep.avg_must_do_latency_sec for ep in episodes])
            ),
            rollout_sec=rollout_sec,
            n_focals=int(np.mean([ep.n_focals for ep in episodes])),
        )

        global_step += batch.routing_steps
        days_done += envs_this_update
        day_durations.append(rollout_sec)

        if cfg.anneal_lr:
            frac = 1.0 - (days_done / max(1, cfg.total_days))
            optimizer.param_groups[0]["lr"] = max(frac, 0.05) * cfg.learning_rate

        lr = optimizer.param_groups[0]["lr"]
        _log(
            f"{day_label} PPO update on {batch.obs.shape[0]:,} samples "
            f"(from {raw_steps:,} focal decisions) lr={lr:.2e}..."
        )
        stats = _ppo_update(agent, optimizer, batch, cfg)
        elapsed = time.perf_counter() - t0

        if batch.must_do_completion_rate > best_must_do_rate:
            best_must_do_rate = batch.must_do_completion_rate
            best_marker = " new_best"
        else:
            best_marker = ""

        avg_day_sec = float(np.mean(day_durations))
        remaining_days = max(cfg.total_days - days_done, 0)
        eta_sec = avg_day_sec * remaining_days

        _log(
            f"{day_label} update={update} global_steps={global_step:,} "
            f"train_samples={batch.obs.shape[0]:,} "
            f"return={batch.episode_return:.2f} must_do={batch.must_do_completion_rate:.3f}"
            f"{best_marker} pref/guest={batch.avg_preference_score_per_guest:.4f} "
            f"rides={batch.rides_completed:,} rides/focal={batch.rides_per_party:.2f} "
            f"pg_loss={stats.pg_loss:.4f} v_loss={stats.v_loss:.4f} "
            f"entropy={stats.entropy:.3f} kl={stats.approx_kl:.4f} clipfrac={stats.clipfrac:.3f} "
            f"rollout={rollout_sec:.1f}s update={stats.update_sec:.1f}s total={elapsed:.1f}s "
            f"eta={eta_sec / 60.0:.1f}m for {remaining_days} day(s)"
        )

        if global_step - last_save_step >= cfg.save_every:
            ckpt = save_dir / f"ppo_step_{global_step}.pt"
            save_checkpoint(
                ckpt,
                agent.model,
                optimizer,
                global_step,
                {
                    "phase": "ppo",
                    "update": update,
                    "days_done": days_done,
                    "num_focals": cfg.num_focals,
                    "must_do_completion_rate": batch.must_do_completion_rate,
                    "avg_preference_score_per_guest": batch.avg_preference_score_per_guest,
                    "avg_wait_variance": batch.avg_wait_variance,
                    "rides_per_party": batch.rides_per_party,
                },
            )
            _log(f"Saved checkpoint: {ckpt}")
            last_save_step = global_step

    final_path = save_dir / "ppo_final.pt"
    save_checkpoint(
        final_path,
        agent.model,
        optimizer,
        global_step,
        {
            "phase": "ppo",
            "days_done": days_done,
            "num_focals": cfg.num_focals,
            "best_must_do_completion_rate": best_must_do_rate,
        },
    )
    _log(
        f"PPO complete after {days_done} day(s) and {global_step:,} routing steps. "
        f"Best must_do rate={best_must_do_rate:.3f}. Final checkpoint: {final_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PPO personal planner training (N focals + heuristic crowd). "
            "Hyperparameters are configured in config.py."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-days",
        type=int,
        default=20,
        help="Number of complete park days to simulate",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel full days per update")
    parser.add_argument(
        "--num-focals",
        type=int,
        default=config.PPO_NUM_FOCALS,
        help="Focal parties trained per day against the heuristic crowd",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional BC checkpoint for warm-start (e.g. checkpoints/bc/bc_final.pt)",
    )
    parser.add_argument(
        "--anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=config.PPO_ANNEAL_LR,
        help="Linearly decay LR over total_days (default from config.PPO_ANNEAL_LR)",
    )
    args = parser.parse_args()

    cfg = PPOConfig(
        seed=args.seed,
        total_days=args.total_days,
        num_envs=args.num_envs,
        device=args.device,
        init_checkpoint=args.init_checkpoint or None,
        anneal_lr=args.anneal_lr,
        num_focals=args.num_focals,
    )
    train(cfg)


if __name__ == "__main__":
    main()
