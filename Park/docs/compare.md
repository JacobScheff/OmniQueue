# Historical RL versus DP comparison

Run from the repository root using a Python environment with Park's NumPy,
NetworkX and PyTorch dependencies (plus Pygame for visualization):

```powershell
python compare.py
python compare.py --day 2026-11-18 --headless
python compare.py --replay Park/comparisons/2026-11-18-seed42.json
```

`Park/compare.py` is also an entry point, including when the working directory is
`Park/`. Explicit paths are relative to the working directory; default paths
resolve from the repository. On this Windows checkout the existing ML environment
can run it directly:

```powershell
& '.\Wait Time Predictions\.venv\Scripts\python.exe' compare.py
```

The default selects yesterday in America/Los_Angeles, falling back to the latest
earlier **complete cached day**. Explicit missing/incomplete dates fail rather
than shortening the test. No Firestore or other network access is needed. Refresh
the archive with the wait project's existing `ml/data.py` exporter when needed.
Both agents run from recorded opening through recorded close, including operating
days crossing midnight. Saved JSON in `Park/comparisons/` contains both event
streams, decision histories, wait boards, settings and scores. A partial file
with the completed RL run is saved before DP starts; replay requires both runs.

The default RL checkpoint is `Park/checkpoints/ppo/ppo_final.pt` when its observation
dimensions match the current code, otherwise the compatible deployed
`Park/companion/model/v2.pt`. The selection and any skipped incompatible checkpoint
are printed; the exact selected path is saved. `--model path.pt` / `--checkpoint`
selects an explicit model and rejects dimension mismatches. In the current local
artifacts, `ppo_final.pt` and companion `v3.pt` expect 44 guest features, while this
checkout implements 43, so `v2.pt` is selected by default. No unknown feature is
invented to make an incompatible checkpoint run. RL close-call sampling is disabled
for reproducibility.

## Shared historical environment

The two independent parties share seeded popularity-weighted preferences, explicit
must-dos, physical walking speed, OSM shortest paths, configured ride durations,
no repeats, and a hard walk-out-by-close deadline. They do not add simulated
crowds to the archive. A completion sets that attraction to done and moves the
party's location to its ride node. The map has one point per attraction, so this
also represents the ride exit. Both models receive this updated state.

Every queue entry samples that time's recorded posted wait, then consumes the
configured ride duration. A later unavailable archive bucket interrupts queued
or on-ride visits with no completion credit; the party remains at the ride node
and replans. Unknown/missing, overflow and closed/down/refurbishment observations
cannot supply a usable wait. Both parties abandon unfinished visits when needed
to walk out by close. Queue positions, evacuation durations and unreported
interruptions cannot be recovered from five-minute posted waits. These are
historical **posted-wait replays**, not measured individual guest outcomes.

Completed attractions are excluded for both policies. An exit/idle choice becomes
waiting until the next five-minute observation and trying again, so a full-day
test does not end at an early model exit action. This rule also applies when DP
has no certified itinerary; its failure status is retained without silently
dropping must-dos. Explicit DP planned departure/queue-entry holds are honored.
Walk, queue, ride and idle time are recorded separately. The DP safety buffer
affects its forecast schedule; it is not an artificial delay charged to actual
visits.

## Forecasts and online DP updates

DP loads the existing wait checkpoint in a persistent worker and rebuilds the
remaining full-day itinerary after **each ride completion**, interruption or
failed attempt. The forecast includes only observations known at that decision.
The latest five-minute origin is used, with history growing throughout the day;
the same origin reuses its sampled forecast. Actual waits at future queue entries
remain hidden from both policies until execution. The runner never passes the
archive or future boards into a policy decision.

Archive repairs need particular care: an interpolated tick is hidden from the
current board until its right endpoint has arrived. Forecast history freezes at
the preceding valid prefix until then. There is no interpolation from a future
endpoint into a decision's history. Each DP decision saves `forecast_as_of`,
`forecast_history_ticks`, the current position/completed IDs, proposed full route,
search status, timeout and beam-pruning flags. DP uses forecast availability and
wait quantiles; attractions omitted from the wait model cannot receive a finite
DP forecast, even if they have recorded waits usable by RL.

The result records whether the evaluation date appears in the wait checkpoint's
train/validation/test dates. `outside_recorded_splits` means the date is absent
from those metadata lists; it is not a claim about other possible data sources.
November 18 is outside the current checkpoint's recorded splits.

## Settings

- `--day YYYY-MM-DD`, `--dataset path.json.gz`, `--seed 42`.
- `--model path.pt`, `--device cpu`, `--threads 1`.
- `--forecast-python path/to/python`, `--forecast-checkpoint path.pt`,
  `--forecast-device cpu`, `--samples 100`. The worker automatically uses the wait
  project's `.venv` interpreter when available, otherwise the current interpreter.
- `--dp-seconds 30`, `--dp-beam 1000`, `--dp-step 60`, `--dp-buffer 300`,
  `--quantile p90`, `--walk-cost 2`. Search time is **per replan**; increasing it
  can make a full day take many minutes. The existing bounded DP returns its best
  validated schedule and reports timeouts/pruning, not global optimality.
- `--must-do 0 --must-do 3` assigns required ride IDs (the default set is empty).
  IDs match `Park/config.py` and the replay map. Both agents receive the same set
  and Play's must-do preference boost.
- `--profile path.json` accepts `preferences` (34 raw nonnegative weights),
  `must_dos` (ride IDs) and `speed` (meters/second). `--walking-speed` overrides
  the speed; `--must-do` overrides the profile's must-do set.
- `--output path.json`, `--headless`, `--replay path.json`, `--speed 300`.

## Replay

The Pygame viewer opens on the final results with RL and DP side by side. It
shows animated party markers on their walking paths, completed rides, must-do
rings, cumulative preference scores, squared preference utility, queue/walk/idle
times and completion itineraries. DP also shows its latest forecast origin and
history count. Both maps and score panels share one clock.

Press **Space** to play/pause, **Home** to restart at opening, **End** for final
results, **Left/Right** for five-minute jumps and **+/-** to change playback speed.
Drag the timeline to scrub; hover a ride to see its name and recorded wait.
Use the mouse wheel over either panel to scroll its completion itinerary.
Green rides are completed; a gold ring marks a must-do. The saved JSON exposes
every failed attempt and full proposed DP route for further analysis.
Replay loads no ML models and needs neither Torch nor the native C++ simulator.

## Validation

```powershell
python -m unittest Park.tests.test_compare Park.tests.test_dp_planner -v
```

Tests cover date fallback, incomplete-day rejection, observation boundaries,
interpolation timing, completion/location updates, queue-entry wait sampling,
queue/on-ride interruption, hard departure with exit travel, full-day idle
progress, repeat prevention and DP context updates.
