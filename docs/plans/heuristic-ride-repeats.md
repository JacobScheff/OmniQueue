# Plan: Heuristic Ride-Repeat Dampening

**Status:** proposal (not implemented)  
**Target:** `native/src/park_sim.cpp` (`route_one`), `docs/heuristic-router.md`, mirrored knobs in `config.py` / `park_sim.hpp`  
**Related:** `PartyArrays::ride_history` (already incremented on `RideComplete`, unused by routing today)

## Problem

The heuristic is **preference-ordered balking**: walk `preference_order` and take the first open, time-feasible ride whose wait is ≤ `balk_sec[ride]`, skipping only the ride the party is already standing at.

That produces a stable ping-pong:

1. Finish top preference A → not “at A” anymore.
2. Next route picks A again if wait still ≤ balk (often true for popular pairs with similar waits).
3. If A’s wait briefly exceeds balk, pick #2 preference B; after B, return to A.

Parties therefore bounce between the same 1–2 rides all day. Repeats are allowed in real parks, but should be **rare** unless preference or wait conditions justify them.

`ride_history[party][ride]` already counts completions per ride; `route_one` ignores it.

## Goals

| Goal | Detail |
|------|--------|
| Novelty bias | Prefer rides the party has not done (or has done fewer times). |
| Allow justified repeats | Must-do / highly preferred rides; or opportunistic short waits when alternatives are long. |
| Keep structure | Stay inside preference-ordered balking + idle/force-pick fallback; avoid a full scored optimizer unless needed. |
| Docs + tests | Update `docs/heuristic-router.md` (and parties note if needed); add focused tests. |

## Non-goals

- Changing spawn preference / must-do generation.
- Pathway congestion or walk-cost in the score (future phases).
- Rewriting PPO rewards (heuristic BC labels will shift; retrain later if desired).
- Hard ban on all repeats.

## Recommended design: multi-pass routing

Extend `route_one` with ordered passes. Each pass still iterates `preference_order` and reuses existing open / time-budget / “not current ride” checks.

### Pass 1 — Fresh rides (primary)

Accept ride `r` if:

- `ride_history[party][r] == 0`
- `wait[r] ≤ balk_sec[party][r]`

Unfinished must-dos already sort first in `preference_order` and almost always have `history == 0`, so they keep priority.

### Pass 2 — Preferred / limited repeats

Accept ride `r` with `history[r] ≥ 1` only if **both**:

1. **Preference gate** — treat as “highly preferred”:
   - rank in `preference_order` is among the top `K` rides (proposed default **`K = 3`**), **or**
   - normalized `preferences[party][r]` ≥ `REPEAT_PREF_THRESHOLD` (proposed default: top ~15% of that party’s pref mass / a fixed floor after looking at spawn distribution — tune with a small calibration script), **and**
2. **Repeat budget** — `history[r] < max_repeats(r)` where e.g.

   ```
   max_repeats(r) = 1 + floor(REPEAT_PREF_SCALE * preferences[party][r] * kNumRides)
   ```

   Cap at a small absolute max (e.g. **3**) so even favorites don’t monopolize the day.

   And `wait[r] ≤ balk_sec[party][r]` (same balk as today, or optionally slightly tighter: `balk * REPEAT_BALK_FACTOR` with factor `< 1`).

Must-do boost (`MUST_DO_PREF_BOOST`) keeps former must-dos high in preference after completion, so a second ride on a must-do remains plausible under this gate without special-casing cleared must-do flags.

### Pass 3 — Opportunistic short wait (any history)

Accept ride `r` if wait is clearly attractive park-wide for this decision, e.g. **any** of:

- `wait[r] ≤ SHORT_WAIT_SEC` (proposed **10–15 min**), or
- `wait[r] ≤ min_wait_among_feasible + SHORT_WAIT_SLACK_SEC`, and that min is still below a soft cap,

**and** the usual open / time / not-current checks.

This covers “everything else is long, this one is short” without requiring low history. Still prefer lower `history[r]` when several rides qualify (stable tie-break: preference order, then lower history).

### Pass 4 — Existing fallback

Unchanged:

| Probability | Action |
|-------------|--------|
| 50% | Idle wander (`ROUTE_IDLE_CODE`) |
| 50% | Force-pick first feasible ride (ignore balk), **but** prefer `history == 0` when scanning, then lowest history |

Force-pick should also respect novelty so the fallback does not reintroduce ping-pong when all waits exceed balk.

## Why multi-pass (not only a balk penalty)

A pure `effective_balk = balk - c * history` still returns the same top preference whenever its wait stays under the reduced threshold — common when two nearby rides stay ~20–30 min. Multi-pass makes novelty the **default**, and only opens repeats under explicit preference or wait exceptions (matching the requested examples).

## Proposed knobs

Mirror in `config.py` and `native/include/park_sim.hpp` (same pattern as balk constants):

| Constant | Proposed start | Role |
|----------|----------------|------|
| `REPEAT_TOP_K` | `3` | Pass 2 rank gate |
| `REPEAT_PREF_THRESHOLD` | tune | Optional mass/floor gate alongside top-K |
| `REPEAT_PREF_SCALE` | `2.0` | Scales `max_repeats` with preference |
| `REPEAT_MAX` | `3` | Hard cap on completions before Pass 2 refuses |
| `SHORT_WAIT_SEC` | `12 * 60` | Pass 3 absolute short-wait bar |
| `SHORT_WAIT_SLACK_SEC` | `2 * 60` | Pass 3 relative-to-best slack |
| `REPEAT_BALK_FACTOR` | `1.0` (or `0.85`) | Optional tighter balk on repeats |

Export via existing config → header discipline if those values are compile-time in C++; otherwise keep C++ constants mirrored by hand like today’s balk knobs.

## Implementation steps

1. **Wire history into `route_one`**  
   - Signature already has `parties`; read `ride_history[party_id]`.  
   - Factor shared “is candidate feasible?” helper used by all passes (open, time remaining, not current ride).

2. **Implement Pass 1 → 2 → 3 → 4** as above; keep `route_batch` unchanged.

3. **Force-pick novelty** in the ignore-balk scan.

4. **Constants** in `park_sim.hpp` + comments in `config.py`.

5. **Docs**  
   - Replace selection section in `docs/heuristic-router.md`.  
   - Note in `docs/parties.md` that `ride_history` drives routing (table currently omits it).  
   - Remove or mark this plan file as done / delete after ship.

6. **Rebuild** `_park_sim` (`pip install -e .`).

7. **Tests** (`tests/`)  
   - Unit-style: construct / expose a small harness if needed, or add a focused C++-visible test via Python if `route_one` stays internal — prefer a thin test hook or simulate via recorded decisions.  
   - Behavioral: seed day metrics — fraction of routing decisions that are repeats should drop sharply; unique rides per party should rise; must-do completion rate should not collapse.  
   - Scenario assertions (if a test helper can set history + waits):  
     - history A=5, B=0, both under balk → pick B.  
     - all fresh waits over balk, one short repeat → Pass 3 picks short.  
     - top-pref with history=1 and other fresh under balk → prefer fresh (Pass 1).

8. **Smoke** `python3 -m pytest`, short `benchmark.py --seed 42 --runs 1`, optional visualize spot-check that parties diversify.

## Validation metrics

Compare seed-matched days before/after:

| Metric | Expected direction |
|--------|-------------------|
| Mean unique rides / party | ↑ |
| Share of decisions with `history[target] > 0` | ↓ sharply |
| Max completions of a single ride per party | ↓ |
| Must-do fulfillment rate | ≈ flat or ↑ |
| Rides/party, avg wait variance | watch for regressions; tune SHORT_WAIT / REPEAT_TOP_K if variance worsens |

Optional debug counter (temporary): pass-id histogram (1/2/3/4) to confirm Pass 1 dominates and Pass 2/3 are rare.

## Risks

| Risk | Mitigation |
|------|------------|
| Parties idle more when many rides already sampled | Pass 3 + novelty-aware force-pick; tune `SHORT_WAIT_SEC` |
| High-pref parties still loop top-K | `REPEAT_MAX` + Pass 1 always before Pass 2 |
| BC dataset / old checkpoints mismatch | Document; re-run `bc_train` when adopting new heuristic as expert |
| Observation already has `rides_completed` total but not per-ride history for PPO | Out of scope; optional later feat if PPO should learn novelty |

## Alternatives considered

1. **Rescale balk by history only** — simpler, weaker against ping-pong when waits stay mid-range.  
2. **Hard skip any `history > 0` until all rides tried** — too strict; blocks favorite re-rides and short-wait opportunism.  
3. **Full utility score** (`pref - α·wait - β·history`) — more flexible, larger behavior change and harder to document; revisit if multi-pass needs too many knobs.

## Suggested implementation PR scope

Single focused PR:

- `route_one` (+ small helpers)  
- constants + docs (`heuristic-router.md`, `parties.md`)  
- tests + before/after metric note in PR description  
- no training re-run required to merge
