# Wait time predictions

The ML code and interactive map are in `ml/`.

New training runs learn CLOSED as an output and save an outage-duration estimate fitted only on training days. The ride-specific estimate gives the probability of reopening in the next five minutes, conditional on how long it has already been DOWN. Sparse observations borrow evidence from other rides. Only observed operating-to-DOWN onsets establish training durations; session breaks, interpolated/unknown observations, and DOWN-to-CLOSED changes censor them. An outage already underway at the first history tick has an unknown onset, so its observed age is a lower bound. Forecasted DOWN states advance the duration without using later actual observations.

CLOSED probabilities come from the neural head. Sampling uses conditional reopening probabilities for ongoing breakdowns; deterministic evaluation uses the median remaining outage duration to avoid staying DOWN forever when every individual five-minute reopening probability is below 50%.

The map and terminal show Down or Closed, without a numeric wait, when at least half of sampled paths are unavailable. DOWN and CLOSED probabilities remain separate. New checkpoints store these policies alongside the weights; older checkpoints retain their original output policy until retrained.

An exact raw operating wait of **13 minutes means 5 minutes**. This conversion happens before five-minute rounding; 12 and 14 are rounded normally. Existing token caches cannot distinguish a former 13 from a genuine 15. Rebuild imported days offline with `python ml/backfill.py --reencode-cache` (using saved raw history, with a backup), then refresh collected days with `python ml/data.py --refresh` using Firestore credentials. Retrain after rebuilding the cache.

```text
ml/
  config.py   settings
  data.py     fetch, save, and prepare the dataset
  model.py    causal ride GRU, cross-ride attention, direct horizon forecasts
  train.py    training on complete days
  test.py     evaluate held-out complete days
  run.py      forecasts from a trained checkpoint
  visualize.py local interactive wait-time map
  visualizer/  map HTML, JavaScript, and styles
```

Edit `ml/config.py` to set dates, credentials, paths, and model settings. Run the following commands from `Wait Time Predictions`. Dataset and checkpoint paths default to this folder regardless of where Python is launched.

Fetch and cache observations from Firestore for training and prediction.

```powershell
python ml/data.py
```

Set `START_DATE`, optionally `END_DATE`, and `CREDENTIALS` in the config. `END_DATE = None` includes today. Credentials can also come from `FIRESTORE_SERVICE_ACCOUNT_JSON` or Google application default credentials. Completed days and finished incomplete days are cached. Current-day observations refresh once per five-minute bucket, with a final read after closing. Set `REFRESH = True` to recheck the configured date range.

To add history from before your collector started, save a ThemeParks.wiki API key in `credentials/themeparks-api-key.txt` (only the key), or set `THEMEPARKS_API_KEY`, then run:

```powershell
python ml/backfill.py
```

The backfill reads the key's permitted history window and downloads finalized Disneyland history through the archive's recorded date, which can trail live data by several days. Existing complete days and incomplete Firestore days are preserved. The original dataset is backed up before the first write, and raw responses are cached under `data/themeparks-history/` so interrupted runs can resume. Optional `--since YYYY-MM-DD` and `--until YYYY-MM-DD` restrict the operating dates. Key files, raw responses, and dataset backups are excluded from Git.

Historical change records are sampled onto the published operating schedule every five minutes, using only the latest state at or before each tick. The API's opening state supplies unchanged values; missing observations remain missing, and degraded opening states stop the import. Operating days that cross midnight use both calendar days. Imported days carry source and sampling metadata and train alongside collected days automatically; no change to the Firestore export start date is needed. The importer selects currently listed attractions, using the names in each historical response. Source archive gaps may still exist even when every reconstructed bucket is present. Powered by [ThemeParks.wiki](https://www.themeparks.wiki/).

The exporter shows days checked, elapsed time, estimated time remaining, and its current stage. While reading Firestore, it reports received snapshots against the day's expected count. The first export can take several minutes depending on the number of days, authentication, and network latency. Fully cached exports finish quickly. Each finished day is saved, so a restarted export reuses that work.

An empty Firestore response records an incomplete day and moves on. Firestore reads use a 30-second request timeout with automatic query retries disabled. A failed request reports its error and leaves that day uncached so it can be retried on the next run.

Set `TRAIN_SPLIT`, `VAL_SPLIT`, and `TEST_SPLIT` in `ml/config.py` (defaults: `0.70`, `0.15`, `0.15`, summing to 1). Complete operating days are sorted chronologically: the oldest days train, the next days validate, and the newest days test. A day is assigned to only one group, including any operating hours past midnight. Counts are rounded to whole days, with at least one day per enabled split; the default requires at least three complete days with two snapshots each. Incomplete days are excluded. Set a holdout fraction to zero to disable that group.

Train the model and print training and validation NLL loss, token accuracy, and the validation forecast objective after each epoch. Only training days update weights, and the attraction catalog comes from training days. With validation enabled, `CHECKPOINT` contains the weights with the best validation forecast objective, rather than the final epoch. `waits.latest.pt` retains the latest completed epoch for recovery and inspection. Without validation, `CHECKPOINT` retains the final epoch. `EARLY_STOPPING_PATIENCE` stops an unproductive run (40 epochs by default; zero disables stopping).

Each checkpoint is written to a `.pt.tmp` file before replacing its destination. If Windows temporarily locks either file, replacement retries for up to 10 seconds. If it still fails, training stops with recovery instructions and preserves the completed `.tmp` checkpoint alongside the previous checkpoint. Once the lock clears, move the `.tmp` file over its destination before starting another training run to keep that completed epoch. Training starts from scratch each run; it does not resume from the saved checkpoint.

`ml/attractions.py` excludes the Treehouse, Explorer Canoes, Monorail, Donald's Duck Pond, Frontierland Shootin' Exposition, Goofy's yard, Great Moments with Mr. Lincoln, Main Street Cinema and Vehicles, Minnie's House, Opera House and both A Magical Life listings, Pixar Short Film Spotlight, Sailing Ship Columbia, Sleeping Beauty Castle Walkthrough, Disney Gallery, and Tiki Room. These attractions never enter new models as inputs or targets and are hidden from the interactive map, including recorded waits. Raw collected data stays intact. Existing checkpoints containing these attractions require retraining with `python ml/train.py` before forecasting or testing; removing output rows alone would leave them influencing other rides.

```powershell
python ml/train.py
```

Evaluate the saved checkpoint on its held-out test days:

```powershell
python ml/test.py
```

The evaluator uses the exact test dates saved in the checkpoint, even after the dataset grows or split settings change. It verifies that test dates overlap neither training nor validation and evaluates without updating the model. Older checkpoints without saved holdouts require retraining before testing. Missing saved test days raise an error instead of silently changing the test set.

`test.py` reports forecast accuracy and mean absolute wait-time error (MAE) separately at **5, 15, and 30 minutes; and 1, 2, 4, 6, 8, and 10 hours** ahead. Displayed horizons use minutes below one hour and hours thereafter; returned `horizons` dictionary keys remain in minutes. Each eligible five-minute observation in a test day starts a new forecast using the full observed prefix through that origin. `ride_horizon_v3` uses the normal one-step head at five minutes and queries that fixed prefix's recurrent state directly for longer leads, conditioned on lead time and known future calendar features. No later actual wait/status observation enters a forecast. Older architectures continue to feed their most likely predictions back autoregressively. The evaluator uses deterministic decoding; `run.py` samples distributions. Both apply the same supported output classes, outage-duration policy, and refurbishment/unobserved-attraction locks. `BATCH_SIZE` controls how many forecast origins are scored together.

Each forecast horizon must fit inside a continuous operating session in the recorded day. Forecasts that cross scheduled breaks or extend past its final observation are excluded. Overnight hours within the same operating day remain eligible. An origin is excluded while an interpolated history tick needs an endpoint later than that origin. Short days can contribute to shorter horizons; unavailable horizons display N/A. The output includes each horizon's forecast count because longer horizons have fewer eligible origins.

Accuracy is exact token accuracy at the requested lead time, pooled over ride/forecast pairs with actual numeric wait, overflow, or DOWN targets. Actual CLOSED, REFURBISHMENT, UNKNOWN, and attractions absent from the training catalog are excluded. Wait-time errors require both actual and predicted tokens to be numeric waits from 0 through 300 minutes; `300+` overflow and status tokens have no exact numeric wait. Each horizon prints accuracy, MAE, RMSE, signed mean bias (`predicted - actual`), and eligible numeric pair count out of all actual numeric waits. Error statistics include zero-error matches and use the quantized five-minute tokens. Detailed quartiles, population standard deviations, and over/underprediction statistics remain available in `evaluate()`'s returned `horizons[minutes]["wait_errors"]` dictionary. Individual ride/time error listings are omitted. Training and validation metrics printed during training remain one-step predictions using observed history.

A closure/reopening timing summary appears only at the end of the report. A closure is an operating wait (including overflow) changing to DOWN or CLOSED; reopening is the reverse. For each eligible ride/forecast origin, the evaluator compares the next actual and predicted transition within the same observed window, up to 10 hours. It reports exact five-minute-bucket accuracy out of actual events, mean absolute timing error and signed timing bias for matched events, matched/actual counts, missed events, and false alarms. Positive bias means late; negative means early. Counts represent forecast opportunities, so the same outage can appear from multiple origins. Timing coverage ends at scheduled breaks, the end of the day, or unknown, refurbishment, or interpolated observations; these cannot establish observed event times. No event is inferred before the first observation or after closing. DOWN-to-CLOSED changes remain unavailable and do not count as reopening. These metrics are returned under `evaluate()["transitions"]["closing" or "reopening"]`. All architectures use the production availability policy. CLOSED is a supported output for newly trained checkpoints.

Select the replay day and the park-local time at which predictions begin in `ml/config.py`.

```python
PREDICT_DAY = "2026-11-11"
PREDICT_START_TIME = "12:00"
```

To print the prediction at a particular park-local time using the saved checkpoint, pass that time to the prediction script.

```powershell
python ml/run.py 18:00
```

The argument is the time to display. With `PREDICT_START_TIME = "17:00"`, `python ml/run.py 18:00` uses real observations through 16:55, generates every five-minute forecast from 17:00 through 18:00, and prints the 18:00 prediction. The requested time must be at or after the configured prediction start. It can extend beyond the configured `STEPS` horizon.

`PREDICT_DAY = None` uses the latest cached day, including an incomplete current day. For a noon start, observed data from opening through 11:55 is the history. The noon tick and every following tick are forecasts from that cutoff. Every scheduled observation from opening up to that cutoff must be present or eligible for interpolation using endpoints before the cutoff. An unresolved missing bucket or attraction observation raises an error before forecasting. Later missing observations are allowed. Published schedules distinguish scheduled breaks from missing data. Explicit UNKNOWN statuses and unavailable standby waits remain valid source observations. Later actual wait and status observations are excluded from the forecast input. Times between five-minute buckets round up to the next bucket, and the preceding start bucket must contain an observation. Clock times before opening refer to the following overnight portion of the selected operating day.

For programmatic `predict()` calls, `STEPS = None` forecasts through the final scheduled tick of that day. Set `STEPS = 24` for a two-hour forecast. When the scheduled day has ended, the fallback horizon is two hours. `PREDICT_START_TIME = None` starts immediately after the final observation. Predictions print directly in the terminal, grouped by park-local time, with each ride's median wait, p10/p90 waits, DOWN probability, or refurbishment/unknown status. Optional `HISTORY` accepts that day's observed snapshots in the archive viewer's JSON response format. This file must include `expected_timestamps` or have a matching day or published schedule in the local cache so history completeness can be checked.

`DEVICE = "auto"` uses an available NVIDIA GPU. Set it to `"cuda"` to require the GPU or `"cpu"` to use the CPU. `EPOCHS`, `BATCH_SIZE`, and the other training settings are in the same config file. `BATCH_SIZE` counts full operating days. Prediction uses the model size stored in the checkpoint and always consumes the full observed day.

## Interactive map

Launch the local map from this folder, using the same Python environment as training:

```powershell
python ml/visualize.py
```

The program opens [the local visualizer](http://127.0.0.1:8766) in your browser. It reads the dataset saved by `ml/data.py` and the checkpoint saved by `ml/train.py`; no Firestore credentials or additional Python packages are required for visualization. You can browse recorded waits without a checkpoint. Run the exporter again and reload the page to include newly collected data.

Choose an **operating day** and **Predictions start** time, then click **Generate forecast**. The initial selection uses `PREDICT_DAY` and `PREDICT_START_TIME` from `ml/config.py`. Select **Viewing time** directly or drag the bottom slider in five-minute steps. Play advances the timeline automatically. The visualizer generates through the final scheduled tick once per day/start selection, so dragging the slider does not run the model again. Times between buckets round up. All controls use Disneyland local time, including the overnight portion of an operating day.

Green solid pins show **recorded data** before the prediction start. Purple dashed pins show **model predictions** at and after the start, even when later actual observations exist in the cache. Amber pins identify **interpolated data**. Missing observations and scheduled breaks have explicit labels. An interpolation that needs an endpoint at or after the forecast start cannot enter prediction history. Unresolved earlier gaps produce an error; move the start to a valid history boundary or refresh the dataset.

When a recorded actual is available for the same ride and time, a badge beside the predicted wait shows the signed error (**predicted − recorded**): yellow for 0–5 minutes of absolute error, orange for 6–15, and red above 15. Wait/status token mismatches (including a numeric wait versus Closed or Down) are red; matching status or `300+` tokens show **Match**. Overflow (`300+`) comparisons never invent an exact minute error. Hover a badge or select the ride to see the recorded value. Missing, unknown, and interpolated actuals are excluded. These comparisons only affect the display and never enter prediction history.

Pan and zoom the map, search or sort the attraction list, and select a ride to see its wait history, forecast, p10/p90 interval, and chance of being down. Predicted waits are medians conditional on operating; the down probability is reported separately. The map uses [Leaflet](https://leafletjs.com/reference-1.9.4.html), public [Orange County aerial imagery](https://www.ocgis.com/arcpub/rest/services/Aerial_Imagery_Countywide/OC_Aerial_3in_WGS84/ImageServer), and an optional [OpenStreetMap](https://www.openstreetmap.org/copyright) street layer. These layers require no API key. The aerial layer requests detailed imagery for each visible map tile through native zoom 20, with larger image requests on high-density displays, instead of enlarging the former zoom-16 USGS tiles. Imagery is a geographic backdrop and does not match the selected historical day. Internet access is needed for map tiles and the Leaflet library. Public attraction coordinates from [ThemeParks.wiki](https://api.themeparks.wiki/docs/v1/) are cached in `data/attraction_locations.json`; attractions without coordinates remain available in the list. Only public map/coordinate requests leave the machine; forecasting uses the local dataset and checkpoint.

Map zoom buttons use quarter-level steps. Mouse-wheel zoom uses Leaflet's normal sensitivity and event timing, with quarter-level snapping and animated transitions.

Optional launch settings:

```powershell
python ml/visualize.py --device auto --samples 100 --port 8766
python ml/visualize.py --dataset data/waits.json.gz --checkpoint checkpoints/waits.pt
python ml/visualize.py --refresh-locations --no-browser
```

The server listens only on `127.0.0.1`. Press Ctrl+C in the terminal to stop it. Device, samples, seed, and temperature default to `ml/config.py`. Day and start selections in the browser do not change the config file.

## One-time setup

From this folder, create and activate a Python environment, then install dependencies. The CUDA 12.8 build was verified on the RTX 5080. Other builds are listed in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-model.txt
```

Activate this environment when opening a new terminal. The collector's separate `requirements.txt` keeps its deployment dependencies small.

## How it works

The cache stores complete operating days in `days` and incomplete days in `incomplete_days`. Both retain their timestamps, ride metadata, and token rows. New records also store every expected timestamp from the published operating schedule. Missing attraction observations have a distinct internal missing marker, separate from an explicit UNKNOWN status. Incomplete days retain their reason and refresh metadata.

A day enters the training group after closing when every scheduled five-minute bucket and attraction observation is present, including eligible interpolated ticks. When one to four scheduled ticks are missing through the latest observation, interior gaps are filled by linear interpolation between the known waits immediately before and after each gap. Later scheduled ticks do not count toward this limit, so in-progress days can be repaired before closing. Previously interpolated ticks count toward the daily limit across refreshes. Interpolated waits round to the nearest five minutes, with halfway values rounded up. Status and overflow categories use the nearer endpoint (the later endpoint on a tie). Missing opening or closing ticks, gaps across scheduled breaks, and missing attraction observations are not filled. Days with five or more elapsed missing ticks remain incomplete. The cache records each interpolated tick's endpoints, and forecasts reject history that would require an endpoint at or after the prediction start. Running `python ml/data.py` also repairs eligible cached incomplete days in the configured date range without refetching them.

Incomplete and in-progress days remain available to `run.py`. Sessions are grouped by local opening date and include hours past midnight. Writes are atomic, and fetch failures preserve saved data. Old rejected days are fetched again once to recover observations discarded by earlier versions of the exporter. Existing complete-day records and trained checkpoints remain usable. An attraction omitted by the source for an entire day cannot be detected from this archive.

New training runs use `ride_horizon_v3`, built on the `ride_transition_v2` encoder. A shared GRU maintains a separate 128-unit learned state for each ride. Inputs combine 64-dimensional token and ride-ID embeddings, six next-tick calendar features, and five numeric features: normalized wait, a numeric-wait flag, the latest wait change, a valid-change flag, and log-scaled token persistence. Two additional state channels retain the last token and its capped run length, so streaming and full-day encoding produce the same results. A masked mean of the park's current ride embeddings enters each ride's GRU input, letting it remember earlier conditions at other rides. Two attention blocks with four heads then compare the recurrent ride states at the same tick. Future ride observations never enter these operations.

The output separates operating/DOWN/CLOSED probabilities from the wait distribution conditional on operating. That wait distribution mixes a learned local change distribution (-60 through +60 minutes in five-minute steps) with a global distribution over all wait buckets. A learned gate selects the balance; unavailable, unknown, and overflow inputs use the global head. Local changes accumulate probability at the zero and overflow boundaries. This provides a useful persistence and small-change prior while still allowing large changes and reopening at a different wait. `HIDDEN_SIZE`, `ATTENTION_BLOCKS`, `ATTENTION_HEADS`, and `DROPOUT` remain configurable; hidden size must be divisible by the head count.

For longer forecasts, a lead-conditioned MLP combines the observed origin's GRU state, token and ride-ID embeddings, future calendar, and three continuous lead features (linear, log, and inverse square root). Cross-ride attention and the structured output head produce each lead's distribution directly. This avoids the accumulation of errors from recursively feeding guessed waits back as observations. `HORIZON_STEPS` supervises five minutes through twelve hours by default; each lead draws `HORIZON_ORIGINS` random origins per training day and fixed, evenly spaced origins per validation day. Targets and calendar features are gathered at origin + lead, while the recurrent encoding stops at the origin. Entire windows must be continuous and unpadded. `HORIZON_WEIGHT` controls this loss alongside the ordinary next-tick loss. This follows the direct/recursive distinction in the [forecasting literature](https://proceedings.mlr.press/v32/taieb14.html).

The objective also includes an ordered numeric-wait loss. `ORDINAL_WEIGHT` controls a ranked probability score over the exact numeric wait buckets: a distant wrong wait costs more than an adjacent one. Statuses and overflow have no numeric distance and contribute only to NLL. This uses the ordering principle also discussed in [CORN](https://arxiv.org/abs/2111.08851); the implementation here uses a cumulative-distribution score rather than CORN's conditional classifiers.

Optional generated-input training is available for autoregressive models (`ROLLOUT_WEIGHT`, zero by default). For each day, `ROLLOUT_ORIGINS` observed prefixes branch into forecasts up to `ROLLOUT_STEPS` ticks long. After the first prediction, the model consumes its own sampled tokens and known future calendar features; actual future tokens are used only as loss targets. Recurrent gradients continue through the generated sequence. Windows crossing scheduled breaks or padding are excluded. Sampling uses the same supported statuses, refurbishment/unknown locks, and outage-duration policy as production. The weight ramps up over `ROLLOUT_WARMUP_EPOCHS`. Validation uses fixed origins, deterministic production decoding, and full loss weights from the start. The combined validation objective selects the best checkpoint without consulting test days. Choose `ride_transition_v2` or `ride_gru_attention_v1` and set `ORDINAL_WEIGHT` and `ROLLOUT_WEIGHT` to zero for a one-step NLL ablation.

The model predicts all rides together with one categorical output per ride. Tokens cover five-minute waits from 0 through 300, overflow, DOWN, CLOSED, REFURBISHMENT, and UNKNOWN. Wait, DOWN, and CLOSED targets contribute to new training runs' loss. Unknown statuses and missing standby waits remain UNKNOWN. Observed refurbishment flags stay locked throughout each forecast. V3 samples numeric waits from each lead's marginal distribution, while carrying sampled outage ages between ticks for the calibrated reopening policy. Numeric samples are not a learned joint queue trajectory. V1 and v2 retain autoregressive sampling.

Checkpoints save the architecture, attention settings, training objective, and selection metrics, so inference reconstructs the trained model independently of current config settings through `build_model()`. V1, v2, and older checkpoints without an architecture field remain supported. Changing config alone does not upgrade existing weights. Training saves to `CHECKPOINT`, so select a different path first if you want to keep old weights for comparison.

The included v3 checkpoint was selected at epoch 137 using 21 training and five validation days, then evaluated on the same four held-out days (November 14–17, 2026) as the previous 200-epoch v1 checkpoint. One/two/four-hour wait MAE changed from 6.17/7.22/8.79 to 5.22/5.59/6.21 minutes; four-hour exact token accuracy rose from 25.2% to 32.4%. Numeric-wait coverage fell from 99.3% to 96.0% at four hours. On the identical 14,414 ride/origin pairs where both models predict numeric waits, four-hour MAE still improves from 8.41 to 6.25 minutes (25.7%). Fifteen-minute accuracy is essentially unchanged. The local full results, shared-pair comparison, training log, and old weights are retained under `data/improvement-*.json`, `data/horizon-training.log`, and `data/waits-before-horizon-v3.pt`.

Availability remains a tradeoff: closing-event matches rise from 37 to 1,302 of 5,605 opportunities, but false alarms rise from 6 to 178. Reopening matches rise from 1,030 to 1,753 of 1,902, while matched-event timing MAE rises from 18.73 to 23.40 minutes and false alarms from 72 to 125. These timing MAEs use different matched cohorts. Four held-out days cannot establish general performance across seasons; additional operating days are needed to assess that.

Training uses only the training partition of complete days in `days`; `PREDICT_DAY` selects a replay and does not affect the split. Each day is a full sequence, with each next-tick prediction conditioned on all earlier observations from that day. Shorter days are padded in a batch and padded targets are excluded from loss. Interpolated ticks are masked to UNKNOWN in training inputs and targets, preventing future interpolation endpoints from entering a prefix and excluding invented targets. History carries across scheduled breaks, while transitions across those breaks are excluded from the five-minute prediction loss. Each epoch reports one-step NLL and accuracy separately from the combined forecast objective. Test days are scored only by `test.py`. `run.py` loads the selected checkpoint to generate predictions. Forecasts can extend past park closing.

Run the regression checks from this folder.

```powershell
python -m unittest discover -s tests -v
node --test tests/test_visualizer_errors.cjs
```
