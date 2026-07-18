#!/usr/bin/env python3
"""Phase 3: PPO fine-tuning on full park-day episodes."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Parent of the Park/ package dir must be on sys.path for `import Park.*`.
_PARENT = ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import _park_sim
import Park.config as config
from Park.model import forward_with_mask, obs_flat_to_tensors, obs_group_to_tensors
from Park.training.checkpoint import default_model, load_checkpoint, save_checkpoint
from Park.training.features import (
    ENV_DYNAMIC_FEAT_DIM,
    FLAT_OBS_DIM,
    GUEST_FEAT_DIM,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
)


def _coordinator_chunk_size() -> int:
    return int(getattr(config, "MAX_COORDINATOR_GUESTS", 32))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_batched_rollout_api() -> None:
    """Fail fast when the installed C++ extension predates exchange_batch."""
    if hasattr(_park_sim.ParkEnv, "exchange_batch"):
        return
    raise RuntimeError(
        "This PPO script requires a rebuilt native extension with ParkEnv.exchange_batch.\n"
        "Your _park_sim module is out of date. From the repo root, rebuild:\n"
        "  pip install -e .\n"
        "Verify with: python -c \"import _park_sim; print(hasattr(_park_sim.ParkEnv,'exchange_batch'))\""
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
    """PPO run configuration.

    Runtime parameters are provided via CLI. All hyperparameters default to
    the values in ``config.py``; edit that file to change training behaviour
    without touching the script.
    """
    # Runtime parameters (set via CLI)
    seed: int = 42
    total_days: int = 20
    num_envs: int = 1
    device: str = "cpu"
    init_checkpoint: str | None = None
    anneal_lr: bool = config.PPO_ANNEAL_LR
    # Hyperparameters (from config.py)
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
    update_wave_batch: int = config.PPO_UPDATE_WAVE_BATCH
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
    wave_ids: torch.Tensor
    routing_steps: int
    episode_return: float
    avg_wait_variance: float
    rides_completed: int
    rides_per_party: float
    must_do_completion_rate: float
    avg_preference_score_per_guest: float
    avg_must_do_latency_sec: float
    rollout_sec: float


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
        *,
        joint_group: bool = False,
    ):
        """Policy forward with legal-action masking.

        joint_group=True treats ``obs`` as one co-timed wave (G, flat) so the
        coordinator can attend across parties. Oversized waves are automatically
        split into chunks of ``MAX_COORDINATOR_GUESTS``.
        """
        if joint_group:
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            max_g = _coordinator_chunk_size()
            if obs.shape[0] > max_g:
                actions_out: list[torch.Tensor] = []
                logprobs_out: list[torch.Tensor] = []
                entropy_out: list[torch.Tensor] = []
                values_out: list[torch.Tensor] = []
                offset = 0
                while offset < obs.shape[0]:
                    chunk = obs[offset : offset + max_g]
                    chunk_action = None
                    if action is not None:
                        chunk_action = action[offset : offset + max_g]
                    a, lp, ent, val = self.get_action_and_value(
                        chunk, chunk_action, joint_group=True
                    )
                    actions_out.append(a)
                    logprobs_out.append(lp)
                    entropy_out.append(ent)
                    values_out.append(val)
                    offset += max_g
                return (
                    torch.cat(actions_out, dim=0),
                    torch.cat(logprobs_out, dim=0),
                    torch.cat(entropy_out, dim=0),
                    torch.cat(values_out, dim=0),
                )

            guest, ride, env = obs_group_to_tensors(obs)
            logits, value, _ = forward_with_mask(self.model, guest, ride, env)
            logits = logits[0]  # (G, A)
            value = value[0, :, 0]
        else:
            guest, ride, env = obs_flat_to_tensors(obs)
            logits, value, _ = forward_with_mask(self.model, guest, ride, env)
            logits = logits[:, 0, :]
            value = value.flatten()

        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_action_and_value_by_waves(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        wave_ids: torch.Tensor,
        *,
        wave_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute logprobs/values with the same joint waves used at rollout.

        Sorts by ``wave_id`` once so members are contiguous, then packs many
        waves into padded ``(B, G, …)`` forwards (no per-wave full-tensor scans).
        """
        n = obs.shape[0]
        device = obs.device
        logprobs = torch.empty(n, dtype=torch.float32, device=device)
        entropy = torch.empty(n, dtype=torch.float32, device=device)
        values = torch.empty(n, dtype=torch.float32, device=device)
        if n == 0:
            return logprobs, entropy, values

        pack = max(1, int(wave_batch_size or getattr(config, "PPO_UPDATE_WAVE_BATCH", 256)))
        table = _WaveTable.build(obs, actions, wave_ids)
        self._eval_wave_table(
            table,
            torch.arange(table.n_waves, device=device),
            logprobs,
            entropy,
            values,
            pack=pack,
        )
        return logprobs, entropy, values

    def _eval_wave_table(
        self,
        table: "_WaveTable",
        wave_rows: torch.Tensor,
        logprobs_out: torch.Tensor,
        entropy_out: torch.Tensor,
        values_out: torch.Tensor,
        *,
        pack: int,
    ) -> None:
        """Run joint policy on selected waves; write outputs at original row indices."""
        rows = wave_rows.detach().cpu().tolist()
        for start in range(0, len(rows), pack):
            chunk = rows[start : start + pack]
            flat_lp, flat_ent, flat_val, flat_orig = self._forward_wave_rows(table, chunk)
            logprobs_out.index_copy_(0, flat_orig, flat_lp)
            entropy_out.index_copy_(0, flat_orig, flat_ent)
            values_out.index_copy_(0, flat_orig, flat_val)

    def _forward_wave_rows(
        self,
        table: "_WaveTable",
        rows: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single joint forward over ``rows`` (one pack). Returns flat lp/ent/val/orig_idx."""
        if not rows:
            device = table.obs.device
            empty_f = table.obs.new_zeros((0,))
            empty_i = torch.zeros(0, dtype=torch.long, device=device)
            return empty_f, empty_f, empty_f, empty_i

        guest_end = GUEST_FEAT_DIM
        ride_end = guest_end + NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
        device = table.obs.device
        lengths = [int(table.ends[r] - table.starts[r]) for r in rows]
        max_g = max(lengths)
        bsz = len(rows)

        guest = table.obs.new_zeros((bsz, max_g, GUEST_FEAT_DIM))
        ride = table.obs.new_zeros((bsz, max_g, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM))
        env = table.obs.new_zeros((bsz, ENV_DYNAMIC_FEAT_DIM))
        act = torch.zeros(bsz, max_g, dtype=torch.long, device=device)
        padding = torch.zeros(bsz, max_g, dtype=torch.bool, device=device)
        orig_idx = torch.empty(bsz, max_g, dtype=torch.long, device=device)

        for row, w in enumerate(rows):
            s = int(table.starts[w])
            e = int(table.ends[w])
            g = e - s
            flat = table.obs[s:e]
            guest[row, :g] = flat[:, :guest_end]
            ride[row, :g] = flat[:, guest_end:ride_end].view(g, NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
            env[row] = flat[0, ride_end:]
            act[row, :g] = table.actions[s:e]
            padding[row, :g] = True
            orig_idx[row, :g] = table.order[s:e]

        logits, value, _ = forward_with_mask(
            self.model, guest, ride, env, guest_padding_mask=padding
        )
        flat_logits = logits[padding]
        flat_actions = act[padding]
        flat_values = value.squeeze(-1)[padding]
        flat_orig = orig_idx[padding]
        dist = torch.distributions.Categorical(logits=flat_logits)
        return dist.log_prob(flat_actions), dist.entropy(), flat_values, flat_orig

@dataclass
class _WaveTable:
    """Wave members laid out contiguously after sorting by wave_id."""

    obs: torch.Tensor
    actions: torch.Tensor
    order: torch.Tensor  # sorted -> original row
    starts: list[int]
    ends: list[int]
    n_waves: int

    @staticmethod
    def build(obs: torch.Tensor, actions: torch.Tensor, wave_ids: torch.Tensor) -> "_WaveTable":
        order = torch.argsort(wave_ids, stable=True)
        sorted_ids = wave_ids.index_select(0, order)
        sorted_obs = obs.index_select(0, order)
        sorted_actions = actions.index_select(0, order)

        ids_np = sorted_ids.detach().cpu().numpy()
        if ids_np.size == 0:
            starts_arr = np.array([], dtype=np.int64)
            ends_arr = np.array([], dtype=np.int64)
        else:
            changes = np.flatnonzero(ids_np[1:] != ids_np[:-1]) + 1
            starts_arr = np.concatenate((np.array([0], dtype=np.int64), changes.astype(np.int64)))
            ends_arr = np.concatenate((starts_arr[1:], np.array([ids_np.size], dtype=np.int64)))
        return _WaveTable(
            obs=sorted_obs,
            actions=sorted_actions,
            order=order,
            starts=starts_arr.tolist(),
            ends=ends_arr.tolist(),
            n_waves=int(starts_arr.size),
        )


def _collect_episode(
    agent: Agent,
    cfg: PPOConfig,
    device: torch.device,
    seed: int,
    *,
    day_label: str = "",
    env_label: str = "",
) -> EpisodeBatch:
    """Run one full park day via native C++ sim + batched policy inference."""
    env = _park_sim.ParkEnv(seed)
    env.reset(seed)

    obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    logprob_buf: list[float] = []
    reward_buf: list[float] = []
    value_buf: list[float] = []
    done_buf: list[float] = []
    wave_id_buf: list[int] = []

    episode_return = 0.0
    terminal_info: dict = {}
    routing_steps = 0
    rollout_t0 = time.perf_counter()
    last_log_step = 0
    prefix = " ".join(part for part in (day_label, env_label) if part)
    wave_counter = 0

    pending_actions: list[int] = []
    staged_obs: np.ndarray | None = None
    staged_actions: np.ndarray | None = None
    staged_logprobs: np.ndarray | None = None
    staged_values: np.ndarray | None = None
    staged_wave_ids: np.ndarray | None = None

    def _record_rewards(rewards_arr: np.ndarray, terminal: bool) -> None:
        nonlocal routing_steps, episode_return, last_log_step
        if staged_obs is None or staged_actions is None:
            raise RuntimeError("Rollout batch state missing staged transitions.")
        if staged_logprobs is None or staged_values is None:
            raise RuntimeError("Rollout batch state missing staged policy outputs.")
        if staged_wave_ids is None:
            raise RuntimeError("Rollout batch state missing staged wave ids.")

        for i in range(len(rewards_arr)):
            obs_row = staged_obs[i]
            obs_buf.append(obs_row.copy())
            action_buf.append(int(staged_actions[i]))
            logprob_buf.append(float(staged_logprobs[i]))
            value_buf.append(float(staged_values[i]))
            reward_buf.append(float(rewards_arr[i]))
            wave_id_buf.append(int(staged_wave_ids[i]))
            episode_return += float(rewards_arr[i])
            routing_steps += 1
            done_buf.append(1.0 if terminal and i == len(rewards_arr) - 1 else 0.0)

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
            terminal_info = {
                "avg_wait_variance": result.metrics.avg_wait_variance(),
                "rides_completed": result.metrics.rides_completed,
                "rides_per_party": result.metrics.rides_per_party(),
                "must_do_completion_rate": result.metrics.must_do_completion_rate(),
                "avg_preference_score_per_guest": result.metrics.avg_preference_score_per_guest(),
                "avg_must_do_latency_sec": result.metrics.avg_must_do_latency_sec(),
            }
            break

        if result.n_obs <= 0:
            break

        obs_np = np.asarray(result.obs, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, FLAT_OBS_DIM)
        if obs_np.shape[0] > remaining:
            obs_np = obs_np[:remaining]

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, logprobs, _, values = agent.get_action_and_value(
                obs_t, joint_group=True
            )

        staged_obs = obs_np
        staged_actions = actions.detach().cpu().numpy()
        staged_logprobs = logprobs.detach().cpu().numpy()
        staged_values = values.detach().cpu().numpy()
        # One wave id per coordinator chunk so PPO updates never rebuild huge G.
        max_g = _coordinator_chunk_size()
        staged_wave_ids = np.empty(obs_np.shape[0], dtype=np.int64)
        for start in range(0, obs_np.shape[0], max_g):
            end = min(start + max_g, obs_np.shape[0])
            staged_wave_ids[start:end] = wave_counter
            wave_counter += 1
        pending_actions = [int(a) for a in staged_actions.tolist()]

    if not obs_buf:
        raise RuntimeError("Episode collected zero routing steps.")

    rollout_sec = time.perf_counter() - rollout_t0

    # If we hit the routing-step cap mid-day, bootstrap GAE from V(s') of the
    # next unconsumed observation instead of treating the trajectory as terminal.
    bootstrap_value = 0.0
    if (
        not episode_done
        and staged_values is not None
        and len(staged_values) > 0
    ):
        bootstrap_value = float(staged_values[0])

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
    wave_ids_t = torch.tensor(wave_id_buf, dtype=torch.long, device=device)

    n = obs_t.shape[0]
    if cfg.subsample_size > 0 and n > cfg.subsample_size:
        # Subsample whole waves so joint coordinator context stays intact.
        unique_waves = torch.unique(wave_ids_t).tolist()
        random.shuffle(unique_waves)
        chosen: list[int] = []
        count = 0
        for wid in unique_waves:
            members = (wave_ids_t == wid).nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(members, int):
                members = [members]
            if count + len(members) > cfg.subsample_size and count > 0:
                break
            chosen.extend(members)
            count += len(members)
            if count >= cfg.subsample_size:
                break
        idx = torch.tensor(sorted(chosen), dtype=torch.long, device=device)
        obs_t = obs_t[idx]
        actions_t = actions_t[idx]
        logprobs_t = logprobs_t[idx]
        advantages_t = advantages_t[idx]
        returns_t = returns_t[idx]
        wave_ids_t = wave_ids_t[idx]

    return EpisodeBatch(
        obs=obs_t,
        actions=actions_t,
        logprobs=logprobs_t,
        advantages=advantages_t,
        returns=returns_t,
        wave_ids=wave_ids_t,
        routing_steps=routing_steps,
        episode_return=episode_return,
        avg_wait_variance=float(terminal_info.get("avg_wait_variance", 0.0)),
        rides_completed=int(terminal_info.get("rides_completed", 0)),
        rides_per_party=float(terminal_info.get("rides_per_party", 0.0)),
        must_do_completion_rate=float(terminal_info.get("must_do_completion_rate", 0.0)),
        avg_preference_score_per_guest=float(
            terminal_info.get("avg_preference_score_per_guest", 0.0)
        ),
        avg_must_do_latency_sec=float(terminal_info.get("avg_must_do_latency_sec", 0.0)),
        rollout_sec=rollout_sec,
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
            # Terminal steps have dones[t]=1 → value bootstrap 0.
            # Truncated rollouts keep dones[t]=0 and use bootstrap_value = V(s').
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
    """PPO update with small per-step graphs (laptop-friendly).

    Each optimizer step forwards at most ``update_wave_batch`` waves, then
    ``backward``/``step`` immediately so autograd does not retain a ~30k-sample
    graph. A short yield after each step reduces display freezes on laptop dGPUs.
    """
    table = _WaveTable.build(batch.obs, batch.actions, batch.wave_ids)
    n_waves = table.n_waves
    pack = max(1, int(cfg.update_wave_batch))
    wave_order = list(range(n_waves))
    yield_sec = float(getattr(cfg, "update_yield_sec", 0.0) or 0.0)

    last_pg_loss = 0.0
    last_v_loss = 0.0
    last_entropy = 0.0
    last_approx_kl = 0.0
    last_clipfrac = 0.0
    mb_adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    update_t0 = time.perf_counter()
    steps_per_epoch = max(1, (n_waves + pack - 1) // pack)
    total_mb = cfg.update_epochs * steps_per_epoch
    mb_done = 0
    last_log_t = update_t0
    device = batch.obs.device

    for epoch in range(cfg.update_epochs):
        random.shuffle(wave_order)
        for start in range(0, n_waves, pack):
            rows = wave_order[start : start + pack]
            new_logprob, entropy, new_value, orig = agent._forward_wave_rows(table, rows)
            if orig.numel() == 0:
                continue

            logratio = new_logprob - batch.logprobs.index_select(0, orig)
            ratio = logratio.exp()
            adv = mb_adv.index_select(0, orig)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((new_value - batch.returns.index_select(0, orig)) ** 2).mean()
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
    _require_batched_rollout_api()
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
        # Load weights into the existing Agent submodule so the Adam optimizer
        # (created on agent.parameters()) keeps tracking the live tensors.
        # Do not restore the BC optimizer state — PPO uses a fresh Adam.
        loaded_model, global_step, init_extra = load_checkpoint(cfg.init_checkpoint, device)
        agent.model.load_state_dict(loaded_model.state_dict())
        _log(f"Loaded init checkpoint: {cfg.init_checkpoint} (prior step={global_step})")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _log(f"PyTorch {torch.__version__} on {_format_device(device)}")
    _log(f"Model parameters: {num_params:,}")
    _log(
        f"PPO full-day mode: target_days={cfg.total_days}, num_envs={cfg.num_envs}, "
        f"subsample={cfg.subsample_size}, inference_batch={cfg.inference_batch_size}, "
        f"update_wave_batch={cfg.update_wave_batch}, "
        f"update_yield_sec={cfg.update_yield_sec}, "
        f"save_every={cfg.save_every} routing steps, log_every={cfg.log_every} rollout steps"
    )
    _log(
        "Primary metrics: must_do rate (higher better), pref/guest (higher better), "
        "must-do latency (lower better); wait_var is diagnostic only"
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
        # Clamp so we never simulate more days than requested
        envs_this_update = min(cfg.num_envs, cfg.total_days - days_done)
        day_num = days_done + 1
        day_label = f"[day {day_num}/{cfg.total_days}]"

        _log(f"{day_label} starting rollout (seed={cfg.seed + days_done})...")

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
                f"return={ep.episode_return:.2f} must_do={ep.must_do_completion_rate:.3f} "
                f"pref/guest={ep.avg_preference_score_per_guest:.4f} "
                f"must_do_lat={ep.avg_must_do_latency_sec / 60.0:.1f}m "
                f"rides={ep.rides_completed:,} rides/party={ep.rides_per_party:.2f} "
                f"time={ep.rollout_sec:.1f}s ({ep.routing_steps / max(ep.rollout_sec, 1e-6):,.0f} steps/s)"
            )

        # Offset wave ids across parallel envs so cohorts stay unique.
        wave_parts = []
        wave_offset = 0
        for ep in episodes:
            wave_parts.append(ep.wave_ids + wave_offset)
            wave_offset = int(wave_parts[-1].max().item()) + 1
        combined_obs = torch.cat([ep.obs for ep in episodes], dim=0)
        combined_actions = torch.cat([ep.actions for ep in episodes], dim=0)
        combined_logprobs = torch.cat([ep.logprobs for ep in episodes], dim=0)
        combined_advantages = torch.cat([ep.advantages for ep in episodes], dim=0)
        combined_returns = torch.cat([ep.returns for ep in episodes], dim=0)
        combined_wave_ids = torch.cat(wave_parts, dim=0)

        batch = EpisodeBatch(
            obs=combined_obs,
            actions=combined_actions,
            logprobs=combined_logprobs,
            advantages=combined_advantages,
            returns=combined_returns,
            wave_ids=combined_wave_ids,
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
            f"(subsampled from {raw_steps:,}) lr={lr:.2e}..."
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
            f"must_do_lat={batch.avg_must_do_latency_sec / 60.0:.1f}m "
            f"rides={batch.rides_completed:,} rides/party={batch.rides_per_party:.2f} "
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
                    "must_do_completion_rate": batch.must_do_completion_rate,
                    "avg_preference_score_per_guest": batch.avg_preference_score_per_guest,
                    "avg_must_do_latency_sec": batch.avg_must_do_latency_sec,
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
            "PPO training on full park-day episodes. "
            "Hyperparameters (learning rate, gamma, GAE, PPO clip, etc.) "
            "are configured in config.py — edit that file to tune them."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-days",
        type=int,
        default=20,
        help="Number of complete park days to simulate (each day ~500k routing decisions)",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel full days per update")
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
    )
    train(cfg)


if __name__ == "__main__":
    main()
