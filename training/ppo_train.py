#!/usr/bin/env python3
"""Phase 3: PPO fine-tuning with CleanRL-style loop and automatic checkpoints."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ParkRouterModel, obs_flat_to_tensors
from training.checkpoint import default_model, load_checkpoint, save_checkpoint
from training.env import ParkRoutingEnv
from training.features import FLAT_OBS_DIM, NUM_ACTIONS


class Agent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        self.model = default_model("cpu")
        self.num_actions = envs.single_action_space.n

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


def make_env(seed: int, idx: int):
    def thunk():
        env = ParkRoutingEnv(seed=seed + idx)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv([make_env(args.seed, i) for i in range(args.num_envs)])
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    global_step = 0
    start_step = 0
    if args.init_checkpoint:
        agent.model, start_step, _ = load_checkpoint(args.init_checkpoint, device, optimizer)

    obs = torch.zeros((args.num_steps, args.num_envs, FLAT_OBS_DIM), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    print(
        f"PPO: {num_updates} updates, {args.num_envs} envs, save every {args.save_every} steps",
        flush=True,
    )

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done, device=device, dtype=torch.float32)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            last_gae = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                advantages[t] = last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            returns = advantages + values

        b_obs = obs.reshape(-1, FLAT_OBS_DIM)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        batch_size = args.num_envs * args.num_steps
        minibatch_size = batch_size // args.num_minibatches
        indices = np.arange(batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb = torch.as_tensor(indices[start:end], device=device, dtype=torch.long)

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb], b_actions.long()[mb]
                )
                logratio = new_logprob - b_logprobs[mb]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        avg_return = rewards.sum(dim=0).mean().item()
        print(
            f"update={update}/{num_updates} step={global_step} return={avg_return:.3f} "
            f"pg_loss={pg_loss.item():.4f} v_loss={v_loss.item():.4f}",
            flush=True,
        )

        if global_step // args.save_every > (global_step - args.num_envs * args.num_steps) // args.save_every or update == num_updates:
            ckpt = save_dir / f"ppo_step_{global_step}.pt"
            save_checkpoint(
                ckpt,
                agent.model,
                optimizer,
                global_step,
                {"phase": "ppo", "update": update, "avg_return": avg_return},
            )
            print(f"Saved checkpoint: {ckpt}", flush=True)

    final_path = save_dir / "ppo_final.pt"
    save_checkpoint(final_path, agent.model, optimizer, global_step, {"phase": "ppo"})
    print(f"PPO complete. Final checkpoint: {final_path}", flush=True)
    envs.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training for ParkRouterModel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
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
    parser.add_argument("--save-every", type=int, default=10_000, help="Save every N env steps")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional BC checkpoint to warm-start PPO (e.g. checkpoints/bc/bc_final.pt)",
    )
    args = parser.parse_args()
    if args.init_checkpoint == "":
        args.init_checkpoint = None
    train(args)


if __name__ == "__main__":
    main()
