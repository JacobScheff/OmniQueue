# Watch (PPO-focal day viewer)

**Modules:** `watch.py`, `watch/`, `_park_sim` play APIs (`play_update_focal_preferences`, `play_focal_state`)

## Overview

A **new** interactive program (separate from `play.py` and `visualize.py`) where you **watch** one **size-1 focal guest routed always by PPO** through a full-population park day. Background guests use the **heuristic** or the **same PPO checkpoint** (UI toggle). The day always runs **park open → close** (no enter/leave pickers).

Sessions store runs **in memory only** for the process lifetime.

## Run

```bash
pip install -e ".[viz]"
python watch.py --model checkpoints/ppo/ppo_final.pt
python watch.py --seed 42 --model checkpoints/ppo/ppo_final.pt --crowd heuristic
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` / `--checkpoint` | *(required)* | Shared PPO checkpoint for focal (+ crowd when PPO) |
| `--seed` | `42` | Day seed |
| `--crowd` | `heuristic` | Initial background router (`heuristic` \| `ppo`); also toggleable in Setup |
| `--speed` | `120` | Simulated seconds per real second while playing the recording |
| `--sample-interval` | `60` | Ride wait snapshot interval in the live recording |
| `--device` | `cpu` | Torch device |

Missing / invalid model → hard error.

**Build note:** watch needs a rebuilt `_park_sim` (adds `play_focal_state` / `play_update_focal_preferences`). After pulling this branch:

```bash
pip install -e ".[viz]"
```

If you see `ParkEnv has no attribute 'play_focal_state'`, the extension is stale — rebuild as above.

## Setup UI

- Focal **preference sliders** + **must-do** flags (crowd parties keep normal randomized spawn prefs)
- **Crowd** toggle: Heuristic ↔ PPO (same model path)
- **Model** path field (pre-filled from `--model`)
- **Start day** — always `0` → `DAY_SECONDS`
- No enter/leave time controls

## Live watch

1. Sim advances at the **recording frontier** while you play the timeline.
2. Map shows ride waits and pathway network; the focal guest is a pulsing **golden/orange** marker labeled `FOCAL` (offset beside the ride while queued / on-ride).
3. Right sidebar lists rides sorted by preference (tall scrollable list) with:
   - **×N** focal completion counts at the playhead
   - **Green** name once completed ≥ 1
   - **Amber** name for outstanding must-dos (until completed)
4. **Timeline:** Play/Pause, scrub **backward** freely, forward by playing. Optional **Skip** jumps to the next focal PPO decision (or queue entry).
5. **Decision marks** on the timeline (toggle **Focal** vs **All** = focal + crowd PPO). Click a mark to inspect the masked softmax distribution in the sidebar.
6. **Preference edits** are allowed only when **paused at the recorded frontier** (cannot rewrite the past). Click **Apply prefs** to push weights/must-dos into the live sim for **future** decisions.
7. When the focal guest **enters a queue**, playback **auto-pauses** at the frontier so you can adjust prefs before resuming.

## Mid-day preference API

```python
env.play_update_focal_preferences(focal_cfg)  # keeps location, history, clock
env.play_focal_state()  # Walking=1, InQueue=2, OnRide=4, ...
env.play_focal_ride_history()  # int16[NUM_RIDES]
```

Must-do remaining flags are recomputed as: UI must-do **and** zero completions so far for that ride.

## Notes

- Does **not** replace `play.py` (human focal) or `visualize.py` (heuristic-only replay).
- Focal is **PPO forever** (no human takeover).
- One shared checkpoint for crowd and focal; only preference vectors differ.
- Runs are **not** saved between process launches.
