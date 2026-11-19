# Plan: History-aware DP itinerary planner

Status: initial standalone planner and aerial explorer implemented; see
[usage and implemented scope](dp-planner.md). Sections below describe the full
design, including follow-up reliability and integration work. The implementation
also aligns queue entries to its decision grid to avoid timing-phase ambiguity.

Build a separate `dp` planning backend for the companion. It plans from the
party's current location and time through departure, maximizing personalized
completion value while protecting every remaining must-do. Keep PPO as a
baseline and optional search hint. A PPO probability is neither itinerary
utility nor a probability of completing a ride.

The proposed model is time-dependent orienteering with mandatory visits and
time windows. Dynamic programming for orienteering with time windows has an
[established research basis](https://air.unimi.it/handle/2434/6449); stochastic
time-dependent routing is also a
[studied problem](https://biblio.ugent.be/publication/8044012). The architecture,
safety policy, and rollout below are recommendations for OmniQueue.

## 1. Inputs and clock semantics

Use an immutable `PlanningContext`, built once per recommendation:

- Absolute timezone-aware `planning_as_of`, park operating date, timezone,
  opening/closing sessions, and the party's departure deadline.
- Current location, physical walking speed, walking willingness, preferences,
  completion counts, remaining must-dos, and known eligibility restrictions.
- Observed park wait/status history through `history_as_of`, forecast origin,
  forecast distributions, availability information, and model identity.
- Canonical attraction IDs, durations, queue-entry windows, completion
  deadlines, verified walk distances, and the exit location.

Separate **park wait history** from **party completion history**. The current
companion request's `history` field means completion counts, not wait snapshots.
Use `completion_counts` and `wait_history` in the new internal contract.

Clock rules:

1. In replay mode, initialize planning at the selected history cutoff. History
   through 10:00 means `planning_as_of = 10:00`; if opening was 08:00,
   `seconds_since_open = 7200`. There are no fictional 08:00-10:00 ride visits.
2. The 10:00 snapshot is observed input. The existing forecast pipeline's first
   predicted tick is 10:05. **The first forecast tick is not the planning time.**
3. In live mode, use actual current time. If it is 10:12 and observations end at
   10:00, plan from 10:12, forecasting across the stale gap. Expose snapshot age
   and widen uncertainty; never move the guest backward to 10:00.
4. Forecasts consume only observations available at the origin. Reuse the
   existing missing-history and interpolation checks; future interpolation
   endpoints cannot fill an earlier prefix.
5. Preserve actual schedules, operating dates, and overnight hours. Convert to
   seconds since opening only at existing simulator interfaces. The physical
   simulator keeps second resolution; planner discretization is separate.
6. If the guest is already walking, queued, or riding, resolve that committed
   activity before planning the next free decision. Do not teleport them to a
   new attraction or count an unfinished ride as completed.

Use the latest observed value at the origin, then a documented stepwise or
interpolated forecast for arbitrary future arrival times. Do not extrapolate
silently beyond forecast coverage. Keep numeric waits, `300+`, DOWN, CLOSED,
REFURBISHMENT, and UNKNOWN distinct; unknown/overflow waits are not zero or a
known finite upper bound.

## 2. Objective and walking preference

For a candidate route P, maximize:

```text
J(P) = sum(value[j] * time_factor(completion[j]))
       - lambda_walk * total_walking_meters / 1000

value[j] = 100 * preference[j]^gamma / sum(preference[k]^gamma)
time_factor(c) = exp(-kappa * (c - session_start) / reference_duration)
```

Values are normalized over the original catalog, without renormalizing after
each completion. Reuse current preference normalization and sharpening concepts
(`gamma = 2` is a starting point). Keep `session_start` fixed at the user's actual
arrival or first planning request, rather than park opening or each replan.
`kappa = 0` disables the preference for earlier completions. There is no separate
elapsed-time score deduction for staying longer.

Initially J scores the conservative schedule, rather than claiming to maximize
true expected utility. Once joint scenarios are calibrated, scenario-averaged
completion value can rank routes that pass the same reliability requirement.

Must-dos are **hard terminal constraints**, including must-dos with preference
zero. No finite completion bonus can replace this requirement. Award value on
completion, never merely on joining a queue. Initially allow at most one new
completion per attraction; completed attractions have no new first-visit value
and completed must-dos leave the required set. Repeats need a later count-based
state and diminishing rewards.

Offer walking choices such as **Minimize walking / Balanced / Happy to walk**,
mapped to decreasing `lambda_walk`. Express the coefficient as utility points
per kilometer and tune it against recognizable preference tradeoffs. For
example, at two points/km, an extra 1.5 km must earn at least three utility points.
These values are tuning examples, not measured defaults.

Walking willingness and speed are separate: willingness changes reward, while
speed changes travel duration and feasibility. Include the initial walk and
walk to the exit in both cost and time. Charge total distance, not average
distance per hop, which can reward adding unnecessary stops. Start with this
soft cost. A hard remaining-distance budget later adds a resource dimension to
both DPs. A maximum leg rule also needs explicit rest waypoints; otherwise
optional stops could become necessary stepping stones for must-do feasibility.

## 3. Scheduling transition

Keep seconds internally. Start with a five-minute decision grid `Delta = 300`,
with a finer one-minute pass when deadlines are tight or coarse scheduling
cannot find a solution. Grid origin is the request's `planning_as_of`, preserving
an off-grid live start such as 10:12. Forecast buckets stay on their own timeline.

At a decision state `(location i, planned time t)`, consider unvisited ride j:

```text
walk = ceil(verified_distance_m(i, j) / party_speed_mps) + walk_buffer
arrival = t + walk
join = next_allowed_queue_entry(j, arrival)
finish = join + wait_bound(j, join) + duration[j] + service_buffer[j]
next_t = round_up_to_decision_grid(finish)
```

Evaluate wait at **queue entry**, which normally equals arrival, rather than
using the 10:00 wait for every later stop. Opening windows can require explicit
waiting before entry. `wait_bound` is the conservative scheduling estimate
described below, not a claim about a calibrated joint probability.

Reject the transition unless eligibility, route connectivity, entry windows,
any ride completion deadline, and the remaining departure constraints hold.
Queue-entry cutoff and completion deadline are different fields. Reserve time
to reach the exit. Default to finishing before published close and exiting by
the guest's deadline; allowing post-close completion requires a known park/ride
policy and enough guest time. Do not copy the DES soft-close drain into live
planning as an assumed entitlement.

Also allow `WAIT` at the current location until the next decision tick. This
permits a later lower wait or scheduled opening to be useful. A forecast may
drop sharply enough that arriving later appears to finish earlier; do not assume
FIFO travel/service curves or discard every later-time state.

Conservative timestamps describe an executable **scheduled** plan. If the guest
finishes a leg early, they can hold until the scheduled departure or queue-entry
time, or request a fresh validated plan. Simply entering an earlier queue can
have a worse forecast wait. If the product permits flexible entry instead,
bound the worst wait/status across that whole entry interval. Never treat a
later arrival estimate as automatically safer.

## 4. DP A: protect the remaining must-dos

Let M be selected must-dos with `completion_counts[j] == 0`. Build a small,
memoized feasibility DP over subsets B of M, last location i, and decision time t:

```text
CanFinish(B, i, t) =
    CanFinish(B, i, t + Delta)                         # wait, if time remains
    OR any valid required-ride transition to j in B:
        CanFinish(B - {j}, j, next_t(i, j, t))

CanFinish(empty, i, t) = exit_walk(i) fits departure constraints
```

The base case includes exit walking cost/resources if a hard walking budget is
enabled. Store a concrete successful continuation as a witness, not just a
boolean. Its queue entries, completions, and exit must all pass the scheduling
checks. Use the same bounds, grid, and resource rules as the main search.

Before accepting **any optional ride**, require that its successor has a known
feasible required-only continuation. This allows a convenient optional ride
before a must-do without consuming the last available opportunity. Merely
checking that each remaining must-do individually fits is insufficient: they
must fit together, in some order.

With m must-dos and T decision ticks, the usual required-node table costs roughly
`O(T * 2^m * m^2)` time and `O(T * 2^m * m)` space; arbitrary optional endpoints
query a first leg into that table. Profile modest sets, initially about 3-8
must-dos, before choosing service limits. Exactness is relative to the chosen
grid and scheduling assumptions.

For a large required set or a timeout, maintain any fully checked witness already
found and use insertion/search to find further witnesses. A failed bounded
search means **uncertified**, not proven impossible. Never drop must-dos to meet
a candidate or runtime cap. A required-only feasibility witness suffices with
ordinary metric walking and arbitrary voluntary waiting; extend the oracle if
later constraints make intermediate rest stops necessary.

The finite WAIT grid makes this a **sufficient conservative filter**, rather
than a completeness proof for the main optimizer. With a sharply falling wait
curve, an optional stop can change the queue-entry timing phase in a way the
required-only grid cannot reproduce. `true` still supplies a valid witness;
`false` can lose a feasible, higher-value itinerary. Refine waiting/queue-entry
choices around forecast changes and deadline events when needed. Keep the small
exact reference optimizer unfiltered by this oracle so measured quality gaps
include losses from the protection filter as well as the beam.

## 5. DP B: maximize preference reward

Use a time-expanded subset DP:

```text
V[S, i, t] = best accumulated utility after completing S,
             at location i, ready at decision time t

V[S union {j}, j, next_t] = max(
    current value,
    V[S, i, t] + value[j] * time_factor(finish)
                - lambda_walk * distance_m(i,j)/1000
)
```

Initialize at the **current** time/location with empty newly completed S.
Previously completed rides are excluded separately. Every extension passes
`CanFinish(M - S_next, j, next_t)`. A WAIT edge advances time without a score deduction. Valid
terminal states have `M subset_of S` and a valid exit transition, charged for
both walking distance and elapsed time. Keep predecessor pointers to reconstruct
the itinerary. A hard walk budget adds remaining meters to the state and oracle.

Two labels can share S, i, and a five-minute forecast bucket while having
different real readiness times, resources, or path history. Do not merge them
unless their decision state is actually equivalent. On the explicit grid, a
single highest-scoring label per exact state suffices for the best route; retain
multiple paths when generating alternatives. Cross-time dominance needs proof
that the earlier label can wait to reproduce the later state with no worse
score or resources. Otherwise keep both.

Exact subset DP across all 34 attractions has `2^34` subsets and is unsuitable as
the default. Use a bounded sparse label DP in production:

1. Seed the incumbent with the must-do witness, including its exit.
2. Keep a configurable total beam of labels per decision tick; start profiling
   hundreds of labels, not a promised latency target.
3. Retain full 34-bit visited masks even if expansion uses fewer candidates.
   Prefer trying all supported attractions first. If filtering is needed,
   include all must-dos plus high-value, nearby, and short-wait options, and
   search several geographic pools.
4. Rank labels with an optimistic bound such as current J plus all remaining
   first-visit values, ignoring future costs. Protect promising different
   required subsets and early route prefixes from beam collapse.
5. Keep the feasible incumbent outside the beam. A time limit can reduce reward
   quality without deleting the known must-do plan.

Before returning, replay each complete route through the scheduling validator.
Report `search_complete`, candidate coverage, grid resolution, bounds version,
and runtime. Call production output **best found**. Beam limits and filtering
remove a global optimality guarantee. Even complete grid search cannot prove
continuous-time infeasibility when rounding has excluded a tight schedule.

```python
context = build_context(request)           # explicit clocks, catalog, forecasts
oracle = build_required_feasibility(context)
incumbent = oracle.witness_at_start()
if incumbent is None:
    return diagnose_or_refine(context)      # no silent removal of required rides

frontier = seed_current_state(context)
while frontier and within_search_budget():
    label = pop_next_time_label(frontier)
    for action in rides_plus_wait(label):
        successor = schedule_and_score(label, action, context)
        if successor is None:
            continue
        if not oracle.has_witness(successor):
            continue
        keep_label_with_beam_and_diversity(successor)
        retain_valid_terminal_routes(successor)

return validate_rank_and_diversify(incumbent, terminal_routes)
```

## 6. What can be promised about must-dos

There are two separate claims:

- **Feasible under bounds:** every must-do fits if the specified walking/service
  bounds hold, the attractions operate as assumed, and the party follows the
  schedule. This is the first implementation's auditable claim.
- **Estimated joint reliability:** the probability that **all** remaining
  must-dos complete and departure constraints hold. Target, for example, 99%
  before describing a route as extremely likely, with calibration evidence and
  uncertainty shown. This is a later capability, not an existing model property.

First ship conservative schedules using high wait estimates plus calibrated
residual buffers where available, protected deadline slack, availability checks,
and replanning. Label uncalibrated bounds as such. Fragile/early-closing must-dos
should usually be protected earlier; the oracle determines whether deferring
them is safe under the bounds.

**Do not translate p90 waits into a 90% route success claim.** Operating wait
quantiles are conditional on operating. They do not cover closures while
waiting or riding, and optional stops before the final must-do also consume
time and risk. The current `ride_horizon_v3` samples horizon marginals with
availability adjustments; its raw draws are not a validated joint park-day
scenario generator. Cross-ride attention does not establish joint calibration.

For a probability claim, add coherent whole-day scenarios with correlated waits,
walking variation, closures/reopening durations, and ride interruptions. Test
each fixed candidate schedule against them, counting success only when all
required completions and departure constraints succeed. Use scenario batches
independent of search/selection for the final check; adjust statistical
confidence for the number of routes offered. Report a lower confidence bound,
sample count, model assumptions, and held-out day-prefix performance, not just
an optimistic sample percentage. Independent simulations from a misspecified
model still do not establish real-world reliability.

Backtest the deployed replan policy as well as fixed schedules. Replanning must
not grant a fresh failure budget at every decision. Preserve a session-level
risk budget or validate the entire adaptive policy. Absolute guarantees remain
conditional because an attraction can close for the rest of the day.

If no plan can be certified, distinguish a known closure/eligibility conflict,
missing forecast or walk coverage, conservative-bound/grid infeasibility, and
search exhaustion. Explain which required attractions are at risk and show a
clearly labeled best-effort recovery with the unmet set. Never present that
recovery as satisfying all must-dos or automatically relax the user's choices.

## 7. Distinct near-optimal route options

Return up to three independently validated routes, only when useful diversity
exists. Similar PPO action probabilities are not a criterion for route quality.

1. Find the best discovered route with original personalized score J_best.
2. Admit alternatives with `J >= J_best - delta`. Since total first-visit value
   is normalized to 100, try a three-point utility gap initially; this is also
   meaningful when net scores are zero or negative. Describe the threshold in
   points, not as a guaranteed percentage of the unknown global optimum.
3. Preserve several different predecessors per exact DP state, or run extra
   searches with prefix/edge exclusions and diversity penalties. A normal
   one-label DP discards different histories that later converge.
4. Measure both directed attraction-transition/order differences and
   length-weighted walkway overlap. Walkway edge-use counts detect repeated
   backtracking; distance-weighted overlap avoids treating many tiny edges as
   more important than a long shared corridor. Require meaningful differences
   early in the itinerary, not just a different final stop. Shared unavoidable
   entrance segments need not dominate the diversity measure.
5. Select the best original score first, then greedily choose eligible routes
   that maximize minimum difference from already selected routes. Diversity
   penalties are for generation/selection only; rescore eligibility with the
   original objective and the same must-do safety checks.

Labels such as **Best preference fit**, **Less walking**, or **Different order**
must describe measured differences. A lower-walking candidate is shown only if
it still meets the user's original score tolerance and must-do requirements.
Show one option if others are effectively duplicates. Do not create variety by
adding arbitrary walking detours to the same attraction sequence.

For each route show stops, entry/completion windows, walk/wait/ride totals, exit
time, preference value, walking cost, must-do coverage, safety slack, forecast
age, and reliability status. Expected times can be displayed alongside the
conservative schedule, with distinct labels. Until joint uncertainty is modeled,
an itinerary generated from median waits is a central estimate, not the
mathematical expectation of a whole route.

## 8. Replanning and implementation boundaries

Replan after completion, a meaningful wait/status update, a location or
preference change, a delay, or a user-requested change. Commit only the next
activity; retain the full remaining-day witness for protection. Keep a queued
ride committed unless the user exits or an interruption occurs. Reduce the
remaining must-do set only after a real completion. Favor retaining a still-good
plan within a small utility tolerance, while safety changes take priority.

Suggested new modules:

- `Park/planner/context.py`: typed inputs, clock/schedule semantics, canonical
  attraction mapping, forecast provenance, verified walking data.
- `Park/planner/forecast.py`: adapter to reusable prediction inference; cache by
  operating day, observed prefix, schedule, and checkpoint version.
- `Park/planner/schedule.py`: one transition/validation implementation shared
  by the oracle, optimizer, alternatives, and response assembly.
- `Park/planner/required.py`: required-subset feasibility and witnesses.
- `Park/planner/dp.py`: exact reference solver and bounded label search.
- `Park/planner/alternatives.py`: score-gap filter and route diversity.
- `Park/planner/evaluate.py`: origin-based replay and DES policy evaluation.

Existing integration points and limitations:

- `Wait Time Predictions/ml/run.py` already validates forecast boundaries,
  consumes full observed prefixes, and returns `as_of` and `prediction_start`.
  Refactor reusable inference away from CLI/global configuration and retain
  distributions/provenance. Do not parse terminal output or reload a checkpoint
  for every candidate transition. The spaced project directory and its `ml`
  imports need a deliberate package/service boundary, not fragile path imports.
- `Wait Time Predictions/ml/data.py` has schedule-aware expected ticks,
  distinct status tokens, and history completeness checks. Reuse those rules.
- `Park/companion/server/ride_map.py` maps ThemeParks IDs to Park indices. Make
  this join explicit and validate one-to-one coverage: the forecast catalog
  excludes some simulator attractions. Missing must-do coverage is unverified,
  never silently removed or assigned zero wait.
- `Park/park_graph.py` scales walk times by speed; `Park/pathways.py` provides
  meters, geometry, and path node IDs. Require verified connectivity. Its
  straight-line fallback for disconnected paths is unsuitable for a protected
  itinerary. The selected path's distance must match the timing calculation;
  account for native random path variants during DES evaluation.
- `Park/companion/server/app.py` currently derives wall time and builds a live
  PPO observation. Add an explicit planning backend and context builder here;
  keep the full context rather than squeezing forecasts into the PPO flat vector.
- The companion request/state lacks active-activity target, queue-entry time,
  and remaining-activity estimates. Scope the first DP endpoint to parties ready
  for a new decision. Supporting requests while walking, queued, or riding needs
  persisted commitment fields and completion/interruption events first.
- The live board/configuration does not provide ride-specific entry schedules
  or guest eligibility. Add date-specific published-hours inputs or verified
  overrides, plus party-declared restrictions. Unknown required-ride deadlines
  or eligibility remain explicit certification gaps; a park-wide closing time
  alone cannot certify an attraction that might close earlier.
- `Park/companion/server/obs.py` has party state and fixed-window clock helpers.
  New planning uses schedule-aware absolute time; arrival time is separate from
  now. Convert to existing observation formats only for PPO comparisons.
- `Park/training/route_reward.py` penalizes mean planned inter-ride walk. The DP
  needs its own total-distance objective, including initial and exit walks.

Expose `planner_backend="dp"` independently of PPO `model_version`. Return a
versioned planning response with full `routes`, `next_action`, clocks, score
components, required coverage, validation assumptions/status, and search
metadata. DP utility must not populate the existing PPO probability fields.
Any forced stop/pin triggers a complete feasibility recheck. Start in the web
companion, then update native clients and play/watch adapters explicitly; do not
change PPO checkpoints or the C++ crowd router to introduce this backend.

## 9. Delivery and validation

1. **Context and scheduling:** implement clock/history/catalog adapters and a
   deterministic transition validator. Use synthetic forecasts first, then
   connect cached inference. Verify late starts, live stale gaps, overnight
   schedules, exclusions, missing history, queue deadlines, and exit travel.
2. **Exact small solver:** enumerate small attraction sets and compare DP output
   to exhaustive permutations plus grid waiting, without the conservative
   required-only filter. Include a later-arrival /
   shorter-wait case, completed and zero-preference must-dos, and an optional
   detour that would strand a required ride. Separately test an optional stop
   changing the queue-entry timing phase to quantify oracle false negatives.
3. **Protected bounded search:** add the oracle and beam, preserving a checked
   incumbent on timeout. Test an infeasible required set separately from search
   exhaustion and coarse-grid failure. Measure score gap against exact small
   instances; benchmark real 34-ride inputs before setting latency/size limits.
4. **Walking and options:** verify stronger walk aversion reduces distance on a
   controlled tradeoff fixture, speed changes feasibility, all legs are charged,
   and alternatives pass the original score gap and must-do checks. Include two
   different prefixes converging to the same DP state.
5. **Companion/replanning:** show full remaining-day routes and first-action
   execution. Verify every replan uses only its available prefix and preserves
   actual completion history. Display degraded certification explicitly.
6. **Reliability calibration:** backtest many held-out complete days at varied
   origins such as 10:00, 13:00, and 17:00, including breakdown-heavy days. Compare
   PPO, heuristic, a required-first baseline, and DP using the same origins,
   guest preferences, completion state, and geography. Replay actual later waits
   only for outcome evaluation. Use DES runs for controlled interruptions and
   routing feedback; historical posted waits alone do not reveal exact personal
   realized queue times or simulate recommendation-driven crowd changes.

Acceptance metrics: all-must-dos completion rate (not merely average per-ride
coverage), missed deadlines, preference value, completion latency, total walking,
forecast age, certification coverage, alternative diversity/score gaps, planner
p50/p95 latency, memory, and the exact-search comparison where available.
Measure probability calibration separately by start time, forecast horizon, and
outage conditions before promising a numerical success rate.

The first deliverable is a protected, explainable, best-found itinerary. A
calibrated whole-route probability is a distinct follow-up milestone.
