# Plan: Multi-Ride Route Output + Consistency / Walk Shaping

**Status:** implemented (model, PPO shaping + CF KL, companion route API; retrain + re-export ONNX required)  
**Goal:** Change the personal planner from a **single next-ride** action to a **length-K route** (default K=6). Execute only `route[0]`; keep the existing preference / must-do reward as primary; add front-weighted **route consistency**, **planned walk**, and **realized walk** shaping; fight mode collapse (universal opener) with a **counterfactual preference KL** regularizer.

Related code today: `model.py` (`ParkRouterModel`), `training/features.py`, `training/ppo_train.py`, `training/bc_train.py`, `router/ppo.py`, `native/src/park_sim.cpp` (`env_reward_delta`, `handle_ride_complete`, `ParkEnv`), `config.py`, `companion/server/recommend.py`, `docs/training.md`, `docs/pref-mustdo-reward-plan.md` (reward baseline — remains in force).

---

## 1. Current behavior (baseline)

| Piece | Today |
|-------|--------|
| Action | Discrete 36: rides `0..33`, exit `34`, idle `35` |
| Model | Single-shot pointer: one guest query × ride keys + exit/idle head |
| Execution | Chosen action applied immediately in DES; next decision only after walk/queue/ride cycle |
| Reward | Pref/must-do completion (time-decayed) + dense urgency + terminal unfulfilled must-do (`docs/pref-mustdo-reward-plan.md`) |
| Walk | Feature only (`RIDE_FEAT_WALK`); **not** in reward |
| Plan object | None — companion/watch/play re-query per decision; argmax over one distribution |
| Known failure | Policy collapses toward a near-universal opener (historically Pirates, then Peter Pan) despite randomized training prefs |

---

## 2. Target behavior

1. At each focal routing decision, the policy emits an ordered route of up to **K=6** distinct rides (or exit/idle as a length-1 special action).
2. The DES / live companion **commits only `route[0]`** (same physical semantics as today’s single action).
3. Primary objective stays **preferred + must-do rides done quickly** (unchanged formulas).
4. Secondary shaping:
   - **Route consistency** — reward replans that keep early slots of the previous plan (weights decay toward the tail).
   - **Planned walk** — penalize sum of inter-ride walk times along the emitted route.
   - **Realized walk** — penalize actual walk time incurred to reach the committed `route[0]`.
5. Anti-collapse: **counterfactual preference KL** so swapping prefs/must-dos (world fixed) moves `π(a_0)` by at least a margin.
6. Consumers (companion, and later watch/play UI) can show the full route for explanation; inference remains deterministic argmax over the decoded route (no sampling / temperature tricks in this plan).

**Out of scope:** branching itinerary trees; temperature annealing; changing companion to sample when top probs are close.

---

## 3. Action and execution contract

### 3.1 Route tensor

```
route = [a_0, a_1, …, a_{L-1}]
L = 1 if a_0 ∈ {exit, idle} else K   # K=6
```

Constraints when `a_0` is a ride:

- All `a_k` are ride ids in `0..NUM_RIDES-1`.
- No repeats within the route (no-replacement).
- Each `a_k` must be **legal under today’s mask** at emission time *as if chosen alone from the current location* for slot 0; for slots `k≥1`, legality is **open + not already picked + not completed (history)**; do **not** require full time-feasibility from the current node for the whole chain (waits will change). Optional v2: soft time-feasibility using cumulative planned walk + current waits.

When soft-closed / exit-only mask: force `a_0 = exit`, `L = 1`.

### 3.2 Execution

| Layer | Behavior |
|-------|----------|
| C++ `ParkEnv` | Still accepts a **single** action id per exchange (unchanged DES). Python applies `route[0]` only. |
| PPO storage | Transition stores full route (+ old route for consistency), log-prob = sum of per-slot log-probs, entropy = sum of per-slot entropies. |
| Companion / ONNX | Export decoder; HTTP returns `route: [{action_id, label, prob_slot}, …]` plus committed recommendation = `route[0]`. Keep full slot-0 distribution for debugging. |

### 3.3 Persistent plan state (per focal party)

Track in Python env wrapper and/or C++ party state:

| Field | Meaning |
|-------|---------|
| `prev_route[K]` | Last emitted ride route (invalid / empty if last action was exit/idle or first decision) |
| `prev_route_len` | 0 or K |
| `last_commit_ride` | Ride id of last committed `route[0]` (for realized walk pairing) |

On each new decision, after sampling/decoding `route`, compute consistency vs `prev_route`, then set `prev_route ← route` (rides only).

---

## 4. Model architecture

Keep the existing ride/guest encoders. Replace the single pointer head with an **autoregressive pointer decoder**. Critic stays a **single scalar** over state (not per route slot).

### 4.1 Forward (training & inference)

```
g = GuestEncoder(guest_feats, env_feats)           # (B, D)
R = RideEncoder(ride_id_embed, ride_feats)         # (B, R, D)

h_0 = g
picked_mask = empty
log_probs = []
entropies = []
actions = []

for k in 0 .. K-1:
    if k == 0:
        logits = [ pointer(h_k, R) | exit_idle_head(h_k) ]   # (B, R+2)
        mask = build_action_mask(...)                        # today’s mask
    else:
        # After a ride commit path only; if a_0 was exit/idle, stop.
        logits = pointer(h_k, R)                             # (B, R)
        mask = open & ~history_done & ~picked_mask
    logits = apply_mask(logits, mask)
    dist = Categorical(logits)
    a_k ~ dist   # or teacher/argmax
    log_probs.append(dist.log_prob(a_k))
    entropies.append(dist.entropy())
    actions.append(a_k)
    picked_mask |= one_hot(a_k)   # rides only
    h_{k+1} = DecoderStep(h_k, EmbedAction(a_k))   # GRU or 1-layer Transformer decoder block

route_log_prob = sum(log_probs)
route_entropy  = sum(entropies)   # see §6 for slot weights in the bonus
value = Critic(g, mean(R), env)
```

**DecoderStep (v1 recommendation):** single-layer GRU with input = ride embedding of `a_k` (learned exit/idle embeds for slot 0 only). Keep parameter count modest vs a deep Transformer decoder.

**Pointer:** reuse today’s `q_proj(h) · k_proj(R) / √D` pattern so BC warm-start of ride encoders remains meaningful.

### 4.2 Checkpoint break

This is a **new architecture**. Old single-head `ppo_final.pt` / ONNX will not load. Plan:

1. Retrain BC with route labels (§7), or init decoder randomly on top of encoder weights from current BC/PPO.
2. Bump a `model_version` / meta field in checkpoints; companion refuses mismatched graphs.

### 4.3 ONNX export

Export a wrapper that returns either:

- `route_actions: (B, K)` via internal argmax decode, plus `slot0_logits: (B, A)`, or  
- stacked per-slot logits with masks (heavier).

Prefer **greedy decode inside the wrapper** for companion simplicity; training stays in PyTorch with sampling.

Export must trace a mid-day open-park path (non-zero `time_left`) and keep `picked` / decoder GRU updates as always-on tensor ops — data-dependent Python `if` around those updates is dropped by legacy `torch.onnx` tracing and yields K copies of the same ride.

---

## 5. Reward redesign (additive)

Primary pref/must-do terms are **unchanged**. All new knobs mirrored `config.py` ↔ `park_sim.hpp` (or computed in Python if easier for plan state — prefer one place; see §5.5).

### 5.1 Front-weighted route consistency

When `prev_route_len == K` and new `route` is a full ride route, on the routing step reward:

```
weights = [w0, w1, w2, w3, w4]   # default: 1.0, 0.5, 0.25, 0.1, 0.05
# Compare new[i] to old[i+1] for i = 0..K-2  (shift consistency)

consist = 0
for i, w in enumerate(weights):
    if not legal_to_prefer(new[i]):          # closed / already done → skip (no penalty)
        continue
    if new[i] == old[i+1]:
        consist += w

reward += PPO_ROUTE_CONSIST_COEF * consist
```

Defaults:

| Knob | Suggested | Notes |
|------|-----------|--------|
| `PPO_ROUTE_K` | `6` | Route length |
| `PPO_ROUTE_CONSIST_COEF` | `0.02` | Scale so max consist (~1.9) ≪ one must-do completion (`~0.15`) |
| `PPO_ROUTE_CONSIST_WEIGHTS` | `(1.0, 0.5, 0.25, 0.1, 0.05)` | Length `K-1` |

No consistency term on first decision of a party, after exit/idle, or when `prev_route` empty.

### 5.2 Planned walk (at emission)

Using park walk-time matrix between ride nodes (party speed or a fixed reference speed — pick **focal effective speed** for training parity):

```
planned = Σ_{i=0}^{K-2} walk_sec(route[i], route[i+1])
reward -= PPO_ROUTE_PLANNED_WALK_COEF * (planned / (K-1) / WALK_NORM_SEC)
```

| Knob | Suggested |
|------|-----------|
| `PPO_ROUTE_PLANNED_WALK_COEF` | `0.01` |
| `WALK_NORM_SEC` | `600` (10 min) — normalize so typical hops are O(0.1) |

### 5.3 Realized walk (after walk completes)

When the party finishes walking to the committed ride (or balks/re-routes — only count successful arrival at the committed entrance), add on the **next** routing reward flush (or immediate step if the env exposes walk duration):

```
reward -= PPO_ROUTE_REALIZED_WALK_COEF * (walk_sec_actual / WALK_NORM_SEC)
```

| Knob | Suggested |
|------|-----------|
| `PPO_ROUTE_REALIZED_WALK_COEF` | `0.02` |

Slightly larger than planned walk: paper routes that ignore geography should not be free if the committed hop is long.

### 5.4 Magnitude rule (non-negotiable)

```
max|consistency| + max|planned walk| + typical|realized walk|
    ≪ PPO_MUST_DO_COMPLETION_BONUS   (and well below a timely high-pref completion)
```

If consistency approaches must-do scale, the agent will refuse to pivot when waits explode. Log component means during PPO.

### 5.5 Where to implement

| Term | Recommended home |
|------|------------------|
| Pref/must-do / urgency / terminal | Stay in C++ `env_reward_delta` / `handle_ride_complete` |
| Consistency + planned walk | **Python** `ParkRoutingEnv` / PPO rollout (has full route + `prev_route`) added into `reward` after C++ delta |
| Realized walk | C++ can expose `last_walk_sec` on the party; Python applies coef — or C++ applies if walk is already known at route time |

Do not fork a second preference objective in Python.

---

## 6. Exploration / entropy (no temperature annealing)

| Knob | Change |
|------|--------|
| `PPO_ENT_COEF` | Raise from `0.01` → start `0.03` (tune) |
| Slot weighting | `route_entropy_bonus = Σ_k α_k H_k` with `α = (1.0, 0.75, 0.5, 0.25, 0.15, 0.1)` so early slots explore more |
| Target entropy (optional v1.1) | If mean `H_0` over a day falls below `H_target` (e.g. `1.2` nats on legal set), temporarily scale ent coef up |

**Do not** use softmax temperature annealing.

Entropy alone does not fix pref-blind collapse — §7 is required.

---

## 7. Counterfactual preference KL (anti-collapse)

### 7.1 Goal

With park state fixed, different preference/must-do vectors must induce **meaningfully different** distributions over the committed action `a_0`.

### 7.2 Pair construction (each PPO minibatch update)

For a fraction `PPO_CF_FRAC` of transitions in the minibatch (default `0.25`):

1. Start from real obs tensors `(guest, ride, env)`.
2. Clone → `guest_cf`, `ride_cf`.
3. Resample a training-style random preference vector + must-do set (same rules as `reset_personal` spawn: uniform prefs / random must-dos — see `docs/parties.md`).
4. Write prefs into guest feats `0..33`, recompute remaining sharpened pref mass (`Σ pref**PPO_PREF_REWARD_EXP` unfinished), rewrite ride must-do flags (feat 7), update guest remaining must-do count; leave waits, walks, history, time, location unchanged.
5. Reject the pair if top must-do (or top pref ride) equals the original — resample up to N times so the counterfactual actually differs.

### 7.3 Loss

Forward policy on both obs (no grad on value for this term, or detach value). Let `π` and `π_cf` be **masked softmax over slot-0 action logits** (rides+exit+idle).

```
js = 0.5 * KL(π || m) + 0.5 * KL(π_cf || m),   m = 0.5(π + π_cf)
L_cf = (relu(PPO_CF_MARGIN - js))²
```

| Knob | Suggested | Notes |
|------|-----------|--------|
| `PPO_CF_COEF` | `0.1` | Multiplier on `L_cf` in total loss |
| `PPO_CF_MARGIN` | `0.15` | Nats of JS; hinge — do not maximize unbounded KL |
| `PPO_CF_FRAC` | `0.25` | Fraction of mb samples that get a CF pair |
| Apply to | **slot 0 only** (v1) | Optional later: light weight on slot 1 |

Total update loss:

```
L = L_ppo - ent_coef * H_weighted + vf_coef * L_v + PPO_CF_COEF * L_cf
```

### 7.4 Why hinge JS (not max KL)

Unbounded KL invites two noisy, wait-blind policies. A margin says “be at least this sensitive to prefs,” then stops pressing.

### 7.5 Eval probe (acceptance)

Fixed waits/location/time template; sweep `N` random must-do sets; record argmax `a_0` histogram and mean JS between random pairs. Success: no single ride > ~25–30% share across the sweep; mean pairwise JS ≳ margin.

---

## 8. Behavioral cloning

Heuristic labels are **single actions**. For route BC:

### 8.1 Label synthesis (v1)

From a mined decision at time t with expert action `a*`:

1. Set `route[0] = a*` (if ride); if exit/idle, length-1 label.
2. For slots `1..K-1`, greedily extend with the heuristic’s next picks under a **frozen short rollout** *or* a static proxy: sort remaining candidates by the same heuristic score assuming waits unchanged and location = previous ride node after `route[i-1]`.

Simplest robust v1: **teacher force slot 0 only** from BC data; train slots `1..K-1` with a self-imitation / heuristic continuation head loss only when continuation labels are available. Accept that early PPO will shape the tail.

### 8.2 Alternative

Skip route BC; warm-start **encoder + pointer projections** from current single-action BC/PPO; randomly init GRU decoder; rely on PPO + CF KL. Faster to ship, colder start on tails.

**Recommendation:** alternative warm-start first; add synthetic route BC if slot-0 regresses vs current checkpoint.

---

## 9. Pipeline / consumer changes

| Area | Change |
|------|--------|
| `model.py` | AR pointer decoder; `forward` returns route actions / logits sequence + value |
| `training/features.py` | Keep obs dims; add helpers for CF pref rewrite; route action packing |
| `training/env.py` | Discrete→MultiDiscrete or custom route space; apply `route[0]`; attach shaping rewards |
| `training/ppo_train.py` | Store routes; weighted entropy; CF KL in update; log consist/walk/CF metrics |
| `training/bc_train.py` | Optional route labels or encoder-only warm-start path |
| `training/eval_policy.py` | Pref KPIs + collapse probe (§7.5) + mean planned/realized walk |
| `router/ppo.py` | Decode full route; `act` returns `route[0]` for sim drivers |
| `companion/server/recommend.py` + ONNX | Return `route` list; still commit `route[0]`; keep distribution for slot 0 |
| `docs/training.md`, `docs/companion.md` | Model I/O + recommend schema |
| `AGENTS.md` | Index this plan; note personal planner emits short routes |
| Tests | Decoder no-repeat/mask; consistency formula; CF rewrite leaves waits intact; reward magnitude smoke |

Watch/play drivers keep calling `act` → single commit; UI can later display full route from `act_with_route` without DES changes.

---

## 10. Implementation order

1. **Doc + knobs** — land this plan; add config placeholders (coefs default 0 / unused until wired).  
2. **Model + unit tests** — AR decoder K=6, masks, greedy/sample decode; checkpoint meta version.  
3. **Env wiring** — apply `route[0]`; track `prev_route`; Python shaping for consistency + planned walk; realized walk via exposed walk sec.  
4. **PPO loop** — route log-prob / weighted entropy; log new reward components.  
5. **Counterfactual KL** — minibatch pairs + hinge JS; collapse probe in `eval_policy`.  
6. **Warm-start** — load compatible encoder weights; smoke few PPO days; retune coefs so shaping ≪ must-do.  
7. **Companion export** — ONNX wrapper + API `route` field; update `docs/companion.md`.  
8. **Tune** — `PPO_ENT_COEF`, `PPO_CF_*`, consist/walk coefs against must-do rate + collapse probe.

Do not enable consistency/walk at full suggested coefs until slot-0 must-do rate is in a healthy band after the architecture swap.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|------|-------------|
| Consistency crystallizes a global Fantasyland loop | Front-weighted only; coef ≪ must-do; CF KL on `a_0`; consistency is vs **party’s own** prev plan |
| Walk terms dominate → land camping, ignore prefs | Tiny coefs; normalize by `WALK_NORM_SEC`; watch component logs |
| CF KL too strong → ignore waits | Hinge margin + modest `PPO_CF_COEF`; never unmasked KL maximize |
| Tail slots meaningless / noisy | Low entropy weight on tail; low consist weight; accept tails as soft lookahead |
| BC gap / cold decoder | Encoder warm-start; optional synthetic route BC later |
| ONNX / companion break | Version meta; ship decoder wrapper; refuse old single-logit models |
| Credit assignment longer | Unchanged DES horizon; route is action representation, not open-loop multi-commit |

---

## 12. Out of scope

- Branching / tree-structured itineraries.  
- Softmax temperature annealing or inference-time sampling when top probs are close.  
- Executing multiple rides from one emission without replan.  
- Reintroducing wait variance into the reward.  
- Pathway congestion (future phase).  
- Aux top-pref classification head and must-do choice-time bonus (considered; **not** in v1 — CF KL only).

---

## 13. Acceptance criteria

- Policy outputs length-K ride routes (or length-1 exit/idle); DES/companion execute only `route[0]`.  
- Pref/must-do urgency, completion, and terminal terms still present and dominant in logged return components.  
- Consistency uses shifted prev-plan matching with decaying weights; skipped when previous ride became illegal.  
- Planned and realized walk penalties active at small scale; mean |shaping| per step ≪ mean must-do completion credit.  
- PPO update includes hinge counterfactual JS on slot-0; collapse probe shows no universal opener domination and pairwise JS above margin on average.  
- Entropy bonus uses early-slot emphasis; no temperature anneal.  
- Companion returns full route for UX; docs/config/tests updated; old single-action checkpoints marked incompatible.
