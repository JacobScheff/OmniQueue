# Rank-Then-Route Personal Planner (`rank_route_v1`)

**Status:** implemented (model, obs, reward/config, PPO/BC hooks, companion inference)  
**Goal:** Personal next-ride planner that ranks rides with a preference/wait-sensitive Stage A, then orders a short route with Stage B. No legacy checkpoint / ONNX compatibility with `route_v1`.

Supersedes the sticky-route failure modes of [`route-plan-output-plan.md`](route-plan-output-plan.md) (consistency shaping + greedy argmax + wait-blind openers). Primary reward remains preference / must-do latency ([`pref-mustdo-reward-plan.md`](pref-mustdo-reward-plan.md)).

---

## Architecture

```
obs → encoders → guest↔ride cross-attn (8 heads)
                → Stage A MLP scorer → π_A (rides + exit/idle)
                → top-M candidates (must-dos forced)
                → Stage B GRU + single-head pointer → route[K]
                → commit route[0]
```

| Piece | Detail |
|-------|--------|
| `D_MODEL` | 512 |
| Cross-attn | Guest queries rides (8 heads); rides get guest-broadcast residual |
| Stage A | Per-ride MLP on `[ride_ctx; guest_ctx; wait; walk; pref; must_do; eta]` |
| Candidates | `PPO_CANDIDATE_M=8` |
| Stage B | AR length `PPO_ROUTE_K=5` over candidates |
| Critic | 4-layer MLP on `guest_ctx ‖ mean(ride_ctx) ‖ env` |

No ride↔ride self-attention. No transformer decoder.

---

## Observation layout

Flat dim = `43 + 34×11 + 3 = 420`.

**Guest (43):** prefs `0..33`, remaining pref mass `34`, speed `35`, time left `36`, loc `37`, rides completed `38`, must-do count `39`, at-ride `40`, state `41`, elapsed `42`.  
Removed vs older layouts: party size, mean balk, walk-target flag.

**Ride (11):** wait, incoming, open, duration, capacity, walk, history, must_do, unfinished pref, **ETA**, **wait_vs_mean**.

**Env (3):** time of day, mean wait, broken fraction (wait-variance slot removed).

Mirrored in `native/src/park_sim.cpp` `build_observation`, `training/features.py`, `companion/server/obs.py`.

---

## Training signals

| Signal | Role |
|--------|------|
| PPO on committed `route[0]` | Pref/must-do completions + walk shaping |
| Stage A entropy | Anti-collapse / branching |
| Pref CF (hinge JS) | Swap prefs → `π_A` must move |
| Wait CF (hinge JS) | Perturb waits → `π_A` must move |
| Pref-rank soft CE | Stage A mass on high-pref / must-dos |
| Planned + realized walk | Keep routes compact |

**Removed:** route-consistency bonus, party-size reward weighting, legacy ride-feat widen bridges.

BC clones heuristic next-ride into **Stage A only**.

---

## Inference diversity

Config: `INFER_TEMP`, `INFER_TOP_P`, `INFER_CLOSE_MARGIN`.

When top-1 vs top-2 Stage A probability gap `< INFER_CLOSE_MARGIN`, sample with temperature / top-p (companion torch path inside `forward_route`; ONNX path re-forces sampled slot 0). Otherwise argmax. Full Stage A distribution is always returned for UI alternatives.

---

## Retrain / export

Old `route_v1` checkpoints will not load usefully (dims + modules differ). Fresh pipeline:

```bash
# from Park/
pip install -e .   # rebuild _park_sim after obs/reward C++ changes
python training/bc_train.py ...
python training/ppo_train.py --init-checkpoint checkpoints/bc/...
python tools/export_companion_onnx.py
```

`MODEL_ARCH_VERSION = "rank_route_v1"`. Companion ONNX meta uses the same arch string.
