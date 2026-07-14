# Interactive Play (Human vs AI)

**Modules:** `play.py`, `play/`, `_park_sim` play APIs (`ParkEnv.reset_play`, `run_play_day`)

## Overview

A **new** interactive program (separate from Phase 4 `visualize.py` replay) where you play as **one size-1 party** on a **full-population** park day. Other guests are routed by the **heuristic** or a **PPO** checkpoint. Session runs (human play, 4-cell AI compare, multi-day benchmark) are stored **in memory only** for the lifetime of the process.

Walks use the same near-shortest path sampler as the AI (including idle **Wander**).

## Run

```bash
pip install -e ".[viz]"
python play.py --seed 42 --crowd heuristic
python play.py --seed 42 --crowd ppo --checkpoint checkpoints/ppo/ppo_final.pt
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--seed` | `42` | Shared day seed (human + AI compare) |
| `--crowd` | `heuristic` | Router for all non-you parties while playing |
| `--checkpoint` | — | Required for PPO crowd, AI compare, and benchmark |
| `--speed` | `120` | Simulated seconds per real second during segment animation |
| `--sample-interval` | `60` | Ride wait snapshot interval in the live recording |
| `--device` | `cpu` | Torch device for PPO |

Missing / invalid checkpoint when PPO is needed → hard error.

## Setup UI

- Enter time / soft leave time (`[` / `]` and `;` / `'` ±30 min)
- Full manual preference weights for all 34 rides (`+` / `-`, **Sort by preference**)
- Must-do checkbox per ride (default off; click checkbox or press `M`)
- Toggle crowd router Heuristic ↔ PPO (`C` or button)
- **Play day**, **AI compare (4)**, **Benchmark 3d**

## Live play

1. Sim advances with hybrid routing until **your** party needs a decision.
2. Recorded walks animate at `--speed` up to that decision (full crowd).
3. Modal: pick a ride (sorted by your prefs), **Wander** (AI idle action), or **Exit**.
4. Soft leave: shown as a target; you are not hard-forced out until park close. AI shadow runs use the same `leave_sec` in the normal router/obs sense.

Focal party is always **party id 0**, size 1, with your prefs/must-dos/spawn/leave injected after seed spawn.

## AI compare (4 cells)

Same seed + profile, anytime (even without playing):

| | Focal: Heuristic | Focal: PPO |
|---|---|---|
| **Crowd: Heuristic** | H / H | H / P |
| **Crowd: PPO** | P / H | P / P |

Each cell stores park KPIs (rides/party, mean wait, wait variance) and focal preference KPIs (preference score, must-do completion, top-3 hits).

## Benchmark

Multi-day **all-heuristic vs all-PPO** (no human), reporting average rides/party, mean wait, and wait variance. Preference-sensitive comparison is covered by the 4-cell AI compare with your UI prefs.

## Native APIs

```python
import _park_sim
cfg = _park_sim.FocalPartyConfig()
# set spawn_sec, leave_sec, preference_weights[34], must_dos[34]
result = _park_sim.run_play_day(seed, cfg)  # heuristic crowd + heuristic focal

env = _park_sim.ParkEnv(seed)
env.reset_play(seed, cfg, crowd_auto_heuristic=True, focal_policy=0, ...)
# focal_policy: 0=human, 1=heuristic, 2=ppo
step = env.play_advance()  # needs_human | needs_ppo_batch | done
```

Python helpers: `play.driver.HybridDriver`, `play.benchmark.run_ai_compare`, `play.benchmark.run_park_benchmark`.

## Notes

- Does **not** replace `visualize.py`.
- Runs are **not** saved between process launches.
- Full park population; hybrid days can take several seconds of wall time between human decisions.
