# DP itinerary explorer

Run from the repository root:

```powershell
python Park/plan.py --demo
```

This opens a local browser explorer with synthetic wait curves and the committed
park walkway graph. The default aerial view uses County of Orange public imagery;
walking routes, numbered attractions, must-do markers, and the animated party are
overlaid on it. Click a ride or itinerary stop to highlight its approach. Use
wheel/button zoom and drag to pan, or switch to Schematic for offline use.
Simplified walkway/ride positions can differ locally from the aerial backdrop.

To use the existing trained wait model and cached day history:

```powershell
python Park/plan.py --day 2026-11-15 --start 10:00
```

The date must exist in the wait dataset. `--start` means history **observed
through** that park-local time, so planning starts at 10:00 with predictions from
10:05. Later observations are excluded by the existing ML inference boundary
checks. A blank day in the UI selects the latest cached operating day. Only
five-minute origins with complete validated history are supported. The adapter
uses the wait project's Windows virtualenv when present; override using
`--forecast-python /path/to/python --device cpu` as needed. Cold model inference
is additional to the DP search budget and can take longer than ten seconds.
Forecast generation is cached for four day/start selections.

You can also pass `--forecast result.json`, containing the dictionary returned by
`ml.run.predict(..., print_output=False)`. Forecast timestamps must have offsets.
For exact origin coverage include an `origin_snapshot` in the same shape as a
forecast snapshot. Without one, the planner cannot enter queues until forecast
coverage begins. An optional integer `closing_timestamp` bounds the horizon;
otherwise the last forecast tick is used conservatively. No extrapolation across
missing ticks or beyond forecast coverage is allowed.
The effective planning cutoff is the earlier of requested departure and forecast
end. The explorer displays this cutoff explicitly, including a warning when the
forecast horizon shortens the requested day. For example, a requested 11:50pm
departure with forecasts ending at 10:55pm still requires exiting by 10:55pm.
The response preserves both `requested_leave` and effective `leave`, plus
`forecast_end` and `departure_limited_by_forecast`.

Choose preferences, must-dos, already completed rides, location, departure,
walking willingness and speed. Settings expose the beam, search budget (default
10 seconds), time grid, and an extra per-ride buffer. Changing the origin updates
forecasts; changing preferences reuses them. Completed rides are excluded from
new visits. Replanning is explicit: update location/completions and build again
when free to choose a new activity. Active queue/ride tracking and integration
into the live companion API are not part of this standalone explorer.

The browser uses a local `POST /api/plan` endpoint. In addition to UI fields it
accepts `queue_deadlines`, mapping ride index strings to local `HH:MM` times, for
known date-specific early closures. Without overrides the departure/forecast
horizon is assumed as the ride deadline. These assumed hours and eligibility
are explicitly unverified; published park-wide hours alone do not certify rides.
The service listens only on loopback and prints its actual port. If Windows
reserves the preferred 8767 port it chooses an available port automatically.

## Implemented algorithm

The planner uses absolute integer seconds and a 1- or 5-minute decision grid.
Departures and queue entries align to that grid; guests can hold until a planned
entry even after an early arrival. The scheduler checks walks, future waits,
duration, extra buffer, queue-entry cutoff, completion deadline, and exit walk.
Forecast DOWN/CLOSED/UNKNOWN/refurbishment, overflow waits, missing rides and
disconnected walk paths cannot supply a finite feasible transition. A finite
operating wait with positive operating probability is eligible, even below 90%.
Availability is a soft score cost (`5 * -log(p_operating)` per entry), so otherwise
similar schedules favor more reliable entries. Below-90% entries are displayed
as route warnings instead of making the must-do set infeasible. This score is
not a whole-route reliability estimate. The default wait estimate remains p90.

A memoized required-subset DP finds a concrete must-do continuation. Every
optional successor must retain one. A time-expanded bounded label DP then
maximizes squared normalized preference value with mild completion-time decay,
minus walking and availability costs. There is no per-hour elapsed-time penalty;
time still constrains feasibility. Current defaults for walking cost are 6/2/0 points per
kilometer for Minimize/Balanced/Happy. Initial and exit legs are charged.

The checked required-only route is retained outside the beam. If the search
budget expires, the best known validated route is returned. If the initial
oracle exhausts its budget or finds no conservative schedule, the result says
`search_incomplete` or `uncertified`, rather than silently dropping requirements.
Failure diagnostics name each required ride and distinguish absent forecast
coverage, no usable operating waits, and a schedule conflict.
The five-minute grid can miss tight continuous-time schedules. The result is
best found, not guaranteed globally optimal.

Up to three routes are chosen within three original utility points of the best
found. They must differ in their first three rides and have a directed
attraction-transition Jaccard distance of at least 0.3. Prefixes are retained
separately when labels converge. This first version uses order/transition
diversity; it does not yet optimize length-weighted walkway-edge diversity or
provide a calibrated route-success probability. Every offered route still
includes all remaining must-dos under the same scheduling assumptions.

## Validation

```powershell
python -m unittest Park.tests.test_dp_planner -v
```

The tests cover late starts, optional detours endangering a must-do, reopenings,
queue deadlines, exit walks, completed rides, disconnected paths, walk aversion,
and comparison against exhaustive small-instance permutations. The demo runs
without NumPy, NetworkX, PyTorch or the native simulator. Real inference runs in
the existing ML environment. Actual interruptions and calibrated joint risk,
session persistence, and automatic live replanning remain follow-up work.

See [the design plan](dp-planner-plan.md) for the full proposed architecture.

For automatic replanning during historical full-day evaluation against OmniQueue
RL, see [`compare.py`](compare.md). It refreshes forecasts using only the history
available at each decision, advances the party through archived waits, and updates
completed rides and the ride-exit position before planning again.
