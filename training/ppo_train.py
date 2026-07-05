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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import obs_flat_to_tensors
from training.checkpoint import default_model, load_checkpoint, save_checkpoint
from training.env import ParkRoutingEnv
from training.features import FLAT_OBS_DIM


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


class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = default_model("cpu")

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        guest, ride, env = obs_flat_to_tensors(obs)
        _, value = self.model(guest, ride, env)
        return value.flatten()

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        guest, ride, env = obs_flat_to_tensors(obs)
        logits, value = self.model(guest, ride, env)
        logits = logits[:, 0, :]
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.flatten()


def _collect_episode(
    agent: Agent,
    device: torch.device,
    seed: int,
    max_routing_steps: int,
    subsample_size: int,
    *,
    gamma: float,
    gae_lambda: float,
) -> EpisodeBatch:
    """Run one full park day (until terminated) and optionally subsample transitions."""
    env = ParkRoutingEnv(seed=seed)
    obs, _ = env.reset(seed=seed)

    obs_buf: list[np.ndarray] = []
    action_buf: list[int] = []
    logprob_buf: list[float] = []
    reward_buf: list[float] = []
    value_buf: list[float] = []
    done_buf: list[float] = []

    episode_return = 0.0
    terminal_info: dict = {}
    routing_steps = 0

    while routing_steps < max_routing_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs_t)

        obs_buf.append(obs.copy())
        action_buf.append(int(action.item()))
        logprob_buf.append(float(logprob.item()))
        value_buf.append(float(value.item()))

        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        reward_buf.append(float(reward))
        episode_return += float(reward)
        routing_steps += 1
        done_buf.append(1.0 if (terminated or truncated) else 0.0)

        if terminated or truncated:
            terminal_info = info.get("metrics", {})
            break

    if not obs_buf:
        raise RuntimeError("Episode collected zero routing steps.")

    rewards_t = torch.tensor(reward_buf, dtype=torch.float32, device=device)
    values_t = torch.tensor(value_buf, dtype=torch.float32, device=device)
    dones_t = torch.tensor(done_buf, dtype=torch.float32, device=device)
    advantages_t, returns_t = _compute_gae(
        rewards_t, values_t, dones_t, gamma=gamma, gae_lambda=gae_lambda
    )

    obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
    actions_t = torch.tensor(action_buf, dtype=torch.long, device=device)
    logprobs_t = torch.tensor(logprob_buf, dtype=torch.float32, device=device)

    n = obs_t.shape[0]
    if subsample_size > 0 and n > subsample_size:
        idx = torch.tensor(sorted(random.sample(range(n), subsample_size)), device=device)
        obs_t = obs_t[idx]
        actions_t = actions_t[idx]
        logprobs_t = logprobs_t[idx]
        advantages_t = advantages_t[idx]
        returns_t = returns_t[idx]

    return EpisodeBatch(
        obs=obs_t,
        actions=actions_t,
        logprobs=logprobs_t,
        advantages=advantages_t,
        returns=returns_t,
        routing_steps=routing_steps,
        episode_return=episode_return,
        avg_wait_variance=float(terminal_info.get("avg_wait_variance", 0.0)),
        rides_completed=int(terminal_info.get("rides_completed", 0)),
    )


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_nonterminal = 0.0
            next_value = 0.0
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
    args: argparse.Namespace,
) -> tuple[float, float]:
    batch_size = batch.obs.shape[0]
    minibatch_size = max(1, batch_size // args.num_minibatches)
    indices = np.arange(batch_size)

    last_pg_loss = 0.0
    last_v_loss = 0.0
    mb_adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    for _ in range(args.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            mb = torch.as_tensor(indices[start:end], device=batch.obs.device, dtype=torch.long)

            _, new_logprob, entropy, new_value = agent.get_action_and_value(
                batch.obs[mb], batch.actions[mb]
            )
            logratio = new_logprob - batch.logprobs[mb]
            ratio = logratio.exp()

            adv = mb_adv[mb]
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((new_value - batch.returns[mb]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            last_pg_loss = float(pg_loss.item())
            last_v_loss = float(v_loss.item())

    return last_pg_loss, last_v_loss


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    global_step = 0
    if args.init_checkpoint:
        agent.model, global_step, _ = load_checkpoint(args.init_checkpoint, device, optimizer)
        print(f"Loaded init checkpoint: {args.init_checkpoint}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"PPO full-day mode: target_days={args.total_days}, num_envs={args.num_envs}, "
        f"subsample={args.subsample_size}, save_every={args.save_every} routing steps",
        flush=True,
    )

    days_done = 0
    update = 0
    last_save_step = 0

    while days_done < args.total_days:
        update += 1
        t0 = time.perf_counter()

        episodes: list[EpisodeBatch] = []
        for i in range(args.num_envs):
            episodes.append(
                _collect_episode(
                    agent,
                    device,
                    args.seed + days_done + i,
                    args.max_routing_steps,
                    args.subsample_size,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
            )

        combined_obs = torch.cat([ep.obs for ep in episodes], dim=0)
        combined_actions = torch.cat([ep.actions for ep in episodes], dim=0)
        combined_logprobs = torch.cat([ep.logprobs for ep in episodes], dim=0)
        combined_advantages = torch.cat([ep.advantages for ep in episodes], dim=0)
        combined_returns = torch.cat([ep.returns for ep in episodes], dim=0)

        batch = EpisodeBatch(
            obs=combined_obs,
            actions=combined_actions,
            logprobs=combined_logprobs,
            advantages=combined_advantages,
            returns=combined_returns,
            routing_steps=sum(ep.routing_steps for ep in episodes),
            episode_return=float(np.mean([ep.episode_return for ep in episodes])),
            avg_wait_variance=float(np.mean([ep.avg_wait_variance for ep in episodes])),
            rides_completed=int(np.mean([ep.rides_completed for ep in episodes])),
        )

        global_step += batch.routing_steps
        days_done += args.num_envs

        if args.anneal_lr:
            frac = 1.0 - (days_done / max(1, args.total_days))
            optimizer.param_groups[0]["lr"] = max(frac, 0.05) * args.learning_rate

        pg_loss, v_loss = _ppo_update(agent, optimizer, batch, args)
        elapsed = time.perf_counter() - t0

        print(
            f"update={update} days={days_done}/{args.total_days} "
            f"routing_steps={batch.routing_steps} train_samples={batch.obs.shape[0]} "
            f"day_return={batch.episode_return:.2f} wait_var={batch.avg_wait_variance:.0f} "
            f"rides={batch.rides_completed} pg_loss={pg_loss:.4f} v_loss={v_loss:.4f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

        if global_step - last_save_step >= args.save_every:
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
                    "avg_wait_variance": batch.avg_wait_variance,
                },
            )
            print(f"Saved checkpoint: {ckpt}", flush=True)
            last_save_step = global_step

    final_path = save_dir / "ppo_final.pt"
    save_checkpoint(
        final_path,
        agent.model,
        optimizer,
        global_step,
        {"phase": "ppo", "days_done": days_done},
    )
    print(f"PPO complete. Final checkpoint: {final_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training on full park-day episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-days",
        type=int,
        default=20,
        help="Number of complete park days to simulate (each day ~500k routing decisions)",
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel full days per update")
    parser.add_argument(
        "--subsample-size",
        type=int,
        default=8192,
        help="Random transitions per day used for PPO update (full day still simulated)",
    )
    parser.add_argument(
        "--max-routing-steps",
        type=int,
        default=600_000,
        help="Safety cap on routing decisions per day",
    )
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints/ppo")
    parser.add_argument("--save-every", type=int, default=500_000, help="Save every N routing steps")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional BC checkpoint (e.g. checkpoints/bc/bc_final.pt)",
    )
    # Legacy alias — maps to total-days estimate when users pass old flag
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--num-steps", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.init_checkpoint == "":
        args.init_checkpoint = None
    if args.total_timesteps is not None:
        # ~500k routing decisions per full day
        args.total_days = max(1, args.total_timesteps // 500_000)
        print(
            f"Note: --total-timesteps is deprecated; treating {args.total_timesteps} as "
            f"~{args.total_days} full day(s). Use --total-days directly.",
            flush=True,
        )
    train(args)


if __name__ == "__main__":
    main()
