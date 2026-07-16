# Plan: Preference / Must-Do PPO Objective

**Status:** implemented (reward, obs, metrics, eval/logging)  
**Goal:** Retarget PPO so the policy maximizes **getting each party’s preferred and must-do rides done quickly**, and **drop wait-variance from the training objective**.

Related code today: `native/src/park_sim.cpp` (`env_reward_delta`, `handle_ride_complete`, `build_observation`), `config.py` (`PPO_*` knobs), `model.py`, `training/ppo_train.py`, `training/eval_policy.py`, `docs/training.md`, `docs/metrics.md`. Interactive play already tracks per-party preference KPIs via `FocalPartyStats` / `play/scoring.py` — reuse those ideas for park-wide eval.

---

## 1. Current behavior (baseline)

| Signal | When | Role today |
|--------|------|------------|
| Dense wait variance | Every routing step | **Primary** — `-PPO_WAIT_VAR_STEP_COEF × var / 1e6` |
| Preference + must-do | On `RideComplete`, flushed on that party’s next route | **Secondary** — `scale × preference[ride]` (+ small must-do bonus) |
| Terminal wait variance | Last routing step | `-avg_wait_variance / 1000` |
| Terminal unfulfilled must-dos | Last routing step | `-PPO_UNFULFILLED_MUST_DO_PENALTY × remaining` (park-wide count) |

Observations already expose what a preference-first policy needs:

- Guest feats `0..33`: normalized preference masses  
- Guest `40`: remaining must-do count  
- Ride feat `7`: per-ride must-do remaining flag  
- Ride feat `6`: completion history  

Env feats still include park mean wait and wait variance (`env[1]`, `env[2]`). The critic is **per-party**; rewards are attached to the party being routed when pending bonuses flush — that credit path is a good fit for individual satisfaction.

**Unit of “guest”:** the DES routes **parties**, not individuals. Preferences and must-dos are party-level. Treat “per guest” as **party outcomes, optionally weighted by `party_size`**, not a new per-person state vector.

---

## 2. Target objective

Optimize, over a full park day:

1. **Must-do fulfillment** — complete assigned must-dos before leave/close.  
2. **Preference mass realized** — completions weighted by `preference[ride]` (and/or top-K hits).  
3. **Latency** — earlier completions of must-dos and high-preference rides are worth more than late ones.

Explicitly **do not** reward or penalize wait variance during PPO. Keep wait-variance samples in `DayMetrics` for logging only.

Instrumental congestion still exists: stampeding one popular ride slows everyone’s preferred completions. The policy may learn mild load-spreading as a means, without an explicit variance term.

---

## 3. Reward redesign (C++ `ParkEnv`)

All knobs stay mirrored in `config.py` ↔ `native/include/park_sim.hpp`.

### 3.1 Remove

- Dense wait-variance term in `env_reward_delta()` (`kWaitVarStepCoef` / `PPO_WAIT_VAR_STEP_COEF` → `0` or delete path).  
- Terminal `-avg_wait_variance / 1000`.  
- Fallback `-kRoutingStepPenalty` when no valid waits (that penalty only existed to densify variance).

### 3.2 Primary: time-weighted completion bonus (sparse)

On real `RideComplete` (not breakdown evacuation), accumulate into `pending_pref_reward_[party]`:

```
time_factor = max(0, 1 - PPO_TIME_DECAY * (now_sec - spawn_sec) / DAY_SECONDS)
# or: exp(-PPO_TIME_DECAY * (now_sec - spawn_sec) / DAY_SECONDS)

bonus  = PPO_PREF_REWARD_SCALE * preference[ride] * time_factor
bonus += PPO_MUST_DO_COMPLETION_BONUS * time_factor   # if was_must_do
bonus *= party_size                                 # optional guest-weighting
pending[party] += bonus
```

Flush unchanged: add pending into the reward on that party’s **next** routing step (and flush leftovers in `terminal_reward_bonus`).

**Must-do scale should dominate preference scale** (today must-do is smaller than pref — invert that). Suggested starting ratios: must-do bonus ≫ per-ride pref mass (e.g. must-do `0.05–0.2`, pref scale `0.02–0.1`), tuned so a day of heuristic routing has a stable return magnitude for the critic.

### 3.3 Dense urgency (recommended)

Sparse completions alone are weak for GAE across ~300k–500k routing steps. On **each** routing step for party `p`, add a small party-local cost:

```
urgency = PPO_MUST_DO_URGENCY_COEF * remaining_must_dos[p]
        + PPO_PREF_URGENCY_COEF * remaining_pref_mass[p]
reward = -urgency  (+ flushed completion bonuses)
```

Define `remaining_pref_mass` as sum of `preference[r]` over rides with `ride_history[r] == 0` (or over unfinished must-dos only, if you want urgency strictly on itinerary items). This creates pressure to clear high-value unfinished rides **quickly** without referencing wait variance.

Keep coefficients small enough that one completion bonus outweighs many steps of urgency (so the agent does not exit early to stop the clock unless leave time forces it).

### 3.4 Terminal

```
terminal = -PPO_UNFULFILLED_MUST_DO_PENALTY * remaining_must_dos_park
         + flush_all_pending_pref_reward()
         - PPO_UNREALIZED_PREF_PENALTY * mean_unrealized_pref_mass   # optional
```

Raise `PPO_UNFULFILLED_MUST_DO_PENALTY` substantially vs today’s `0.002` so unfinished must-dos hurt at day end.

### 3.5 What not to add (v1)

- Park-wide wait / variance penalties.  
- Rewards for arbitrary low-preference fillers (unless you keep a tiny floor so the policy does not idle forever). Idle wander already burns time via urgency + lost completion opportunity.

---

## 4. Model / observation changes

### 4.1 Likely sufficient with small feature tweaks

`ParkRouterModel` (guest transformer + pointer over rides + per-party critic) already matches a **per-party** return. No architecture change required for v1.

Recommended observation updates in `build_observation` / `training/features.py`:

| Change | Why |
|--------|-----|
| Guest: `elapsed_since_spawn = (now - spawn) / DAY` | Makes “quickly” visible to policy/critic |
| Guest: `remaining_pref_mass` | Aligns with urgency reward |
| Ride: keep must-do flag + history; optionally add `preference[r]` duplicated on ride axis if helpful | Pointer already sees prefs on guest vector |
| Env: stop feeding wait variance as a privileged training signal — zero `env[2]`, or replace with neutral park load (e.g. mean wait only) if you still want congestion context | Prevents critic from overfitting to the old KPI |

Update `GUEST_FEAT_DIM` / `FLAT_OBS_DIM` and any hard-coded index docs if guest dim grows. Rebuild `_park_sim` after C++ feature changes.

### 4.2 Optional later (not required for objective flip)

- Separate must-do vs preference value heads.  
- Action prior / mask bias toward unfinished must-dos (usually unnecessary if reward + features are clear).  
- Shrink coordinator depth if multi-party attention proves less useful without park-wide variance sharing.

---

## 5. Training pipeline changes

| Area | Change |
|------|--------|
| `config.py` / `park_sim.hpp` | New knobs: `PPO_TIME_DECAY`, `PPO_MUST_DO_URGENCY_COEF`, `PPO_PREF_URGENCY_COEF`, guest-weight flag; zero/remove wait-var coefs; retune pref/must-do/terminal scales |
| BC warm-start | **Keep.** Heuristic already sorts must-dos first and preference-orders candidates (`docs/heuristic-router.md`). Re-BC only if spawn/heuristic semantics change. Expect PPO to unlearn opportunistic short-wait herding when it conflicts with personal prefs |
| `ppo_train.py` | Log new return components if exposed; no GAE math change. Revisit `PPO_GAMMA` only if dense urgency makes returns much shorter-horizon |
| `eval_policy.py` | Report must-do rate, mean preference score, mean must-do latency, top-3 hit rate — not wait variance as the headline |
| Metrics (`DayMetrics` / Python) | Add park-averaged: must-do completion rate, preference score per party (or per guest), mean time-from-spawn to each must-do completion. Keep wait variance as a **diagnostic** only |
| Docs | Update `docs/training.md`, `docs/metrics.md`, and AGENTS.md reward bullet to match |
| Tests | Unit-test reward formulas on a tiny `ParkEnv` fixture: completion bonus scales with pref/must-do/time; no variance term; urgency decreases when a must-do clears; terminal penalty counts remaining |

### Evaluation protocol

1. Heuristic baseline focal + park preference scores (`play/scoring.py` style).  
2. BC checkpoint under the **new** reward (for apples-to-apples returns).  
3. PPO after N days: must-do fulfillment ↑, mean must-do latency ↓, preference score ↑.  
4. Watch wait variance as a side effect (may worsen); only reintroduce a soft congestion term if latency regresses due to stampeding.

---

## 6. Implementation order

1. **Reward only** in C++ + config mirrors; leave obs/model dims unchanged.  
2. Wire eval/metrics for preference KPIs; update docs/tests.  
3. Smoke PPO for a few days from existing BC checkpoint; check return scale and must-do rate.  
4. Add urgency + time decay if completions are too sparse or “quickly” is weak.  
5. Add guest elapsed / remaining-pref features + rebuild if the policy under-uses must-do flags.  
6. Only then consider architecture tweaks.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Herding onto top popularity rides → long waits → slow must-dos | Urgency + time decay already punish that; optional later mild mean-wait penalty, still not variance |
| Reward scale shock vs BC init | Log component magnitudes; normalize so heuristic day return is O(1)–O(10) |
| Early exit to stop urgency | Cap urgency; require exit only via soft-close / leave; make completion bonuses large vs residual urgency |
| Party vs guest fairness | Weight bonuses by `party_size` in reward and report per-guest KPIs |
| Credit delay until next route | Existing pending buffer is fine; flush at terminal covers last completion |

---

## 8. Out of scope for this objective flip

- Pathway congestion (future phase).  
- Changing spawn preference / must-do generation (keep popularity-weighted, non-land).  
- Replacing the pointer actor-critic.  
- Making wait variance the primary KPI again.

---

## 9. Acceptance criteria

- PPO reward path contains **no** wait-variance term (step or terminal).  
- Completing a must-do or high-preference ride yields a clearly larger training signal than filler rides.  
- Earlier completions yield higher bonus than identical late completions (time decay and/or urgency).  
- Eval headlines are preference / must-do / latency; wait variance is diagnostic only.  
- Docs and config mirrors stay in sync; tests lock the new formulas.
