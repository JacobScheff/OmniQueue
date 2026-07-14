# Interactive Play (Human vs AI)

**Modules:** `play.py`, `play/`, `_park_sim` play APIs (`ParkEnv.reset_play`, `run_play_day`)

## Overview

A **new** interactive program (separate from Phase 4 `visualize.py` replay) where you play as **one size-1 party** on a **full-population** park day. Other guests are routed by the **heuristic** or a **PPO** model. Session runs (human play, 4-cell AI compare, multi-day benchmark) are stored **in memory only** for the lifetime of the process.

Walks use the same near-shortest path sampler as the AI (including idle **Wander**).

## Run

```bash
pip install -e ".[viz]"
python play.py --seed 42
python play.py --seed 42 --model checkpoints/ppo/ppo_final.pt
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--seed` | `42` | Shared day seed (human + AI compare) |
| `--model` | — | Optional PPO model/checkpoint path (also editable in the Setup UI). Alias: `--checkpoint` |
| `--speed` | `120` | Simulated seconds per real second during segment animation |
| `--sample-interval` | `60` | Ride wait snapshot interval in the live recording |
| `--device` | `cpu` | Torch device for PPO |

Crowd router (**Heuristic** vs **PPO**) is toggled in the Setup UI, not via CLI. Missing / invalid model when PPO is needed → hard error.

## Setup UI

- **Enter** / **Leave** time pickers (`−` / `+` buttons, ±30 min). AI compare and shadow runs use these **exact** times for the focal guest.
- Preference list rows: **ride name | preference slider | must-do checkbox** (large readable names)
- **Sort by preference** reorders rows by current slider values
- **Crowd AI** toggle: Heuristic ↔ PPO
- **PPO model** text field: click and type a path (pre-filled from `--model` when provided)
- **Play day**, **AI compare (4)**, **Benchmark 3d**

## Live play

1. Sim advances with hybrid routing until **your** party needs a decision.
2. Recorded walks animate at `--speed` up to that decision (full crowd).
3. Map shows ride wait minutes and short names (same style as `visualize.py`), plus a pulsing **YOU** marker for your guest.
4. Choose a ride from the wider sidebar list **or by clicking a ride circle on the map**; **Wander** (AI idle) or **Exit**.
5. Soft leave: shown as a target; you are not hard-forced out until park close. AI shadow runs use the same `leave_sec` in the normal router/obs sense.

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
