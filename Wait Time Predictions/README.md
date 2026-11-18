# Wait time predictions

The ML code and interactive map are in `ml/`.

```text
ml/
  config.py   settings
  data.py     fetch, save, and prepare the dataset
  model.py    shared ride GRU, cross-ride attention, autoregressive sampling
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

Train the model and print training and validation NLL loss and token accuracy after each epoch. Only training days update weights, and the attraction catalog comes from training days. The training command saves the final model state and metrics after every epoch.

Checkpoints are written to `checkpoints/waits.pt.tmp` before replacing `checkpoints/waits.pt`. If Windows temporarily locks either file, replacement retries for up to 10 seconds. If it still fails, training stops with recovery instructions and preserves the completed `.tmp` checkpoint alongside the previous checkpoint. Once the lock clears, move the `.tmp` file over `waits.pt` before starting another training run to keep that completed epoch. Training starts from scratch each run; it does not resume from the saved checkpoint.

`ml/attractions.py` excludes the Treehouse, Explorer Canoes, Monorail, Donald's Duck Pond, Frontierland Shootin' Exposition, Goofy's yard, Great Moments with Mr. Lincoln, Main Street Cinema and Vehicles, Minnie's House, Opera House and both A Magical Life listings, Pixar Short Film Spotlight, Sailing Ship Columbia, Sleeping Beauty Castle Walkthrough, Disney Gallery, and Tiki Room. These attractions never enter new models as inputs or targets and are hidden from the interactive map, including recorded waits. Raw collected data stays intact. Existing checkpoints containing these attractions require retraining with `python ml/train.py` before forecasting or testing; removing output rows alone would leave them influencing other rides.

```powershell
python ml/train.py
```

Evaluate the saved checkpoint on its held-out test days:

```powershell
python ml/test.py
```

The evaluator uses the exact test dates saved in the checkpoint, even after the dataset grows or split settings change. It verifies that test dates overlap neither training nor validation and evaluates without updating the model. Older checkpoints without saved holdouts require retraining before testing. Missing saved test days raise an error instead of silently changing the test set.

`test.py` reports autoregressive forecast accuracy and mean absolute wait-time error (MAE) separately at **5, 15, 30, 60, 120, and 240 minutes** ahead. Each eligible five-minute observation in a test day starts a new forecast using the full observed prefix through that origin. After that point, the model receives its own most likely predicted tokens, never the later actual wait/status observations. Future calendar features are known inputs. This is a deterministic greedy rollout; `run.py` continues to sample multiple possible paths. The evaluator applies the same output-category exclusions and refurbishment/unobserved-attraction locks as the production forecaster. `BATCH_SIZE` controls how many forecast origins roll out together during testing.

Each forecast horizon must fit inside a continuous operating session in the recorded day. Forecasts that cross scheduled breaks or extend past its final observation are excluded. Overnight hours within the same operating day remain eligible. An origin is excluded while an interpolated history tick needs an endpoint later than that origin. Short days can contribute to shorter horizons; unavailable horizons display N/A. The output includes each horizon's forecast count because longer horizons have fewer eligible origins.

Accuracy is exact token accuracy at the requested lead time, pooled over ride/forecast pairs with actual numeric wait, overflow, or DOWN targets. Actual CLOSED, REFURBISHMENT, UNKNOWN, and attractions absent from the training catalog are excluded. Wait-time errors require both actual and predicted tokens to be numeric waits from 0 through 300 minutes; `300+` overflow and status tokens have no exact numeric wait. Each horizon prints accuracy, MAE, RMSE, signed mean bias (`predicted - actual`), and eligible numeric pair count out of all actual numeric waits. Error statistics include zero-error matches and use the quantized five-minute tokens. Detailed quartiles, population standard deviations, and over/underprediction statistics remain available in `evaluate()`'s returned `horizons[minutes]["wait_errors"]` dictionary. Individual ride/time error listings are omitted. Training and validation metrics printed during training remain one-step predictions using observed history.

A closure/reopening timing summary appears only at the end of the report. A closure is an operating wait (including overflow) changing to DOWN or CLOSED; reopening is the reverse. For each eligible ride/forecast origin, the evaluator compares the next actual and predicted transition within the same observed window, up to 240 minutes. It reports exact five-minute-bucket accuracy out of actual events, mean absolute timing error and signed timing bias for matched events, matched/actual counts, missed events, and false alarms. Positive bias means late; negative means early. Counts represent forecast opportunities, so the same outage can appear from multiple origins. Timing coverage ends at scheduled breaks, the end of the day, or unknown, refurbishment, or interpolated observations; these cannot establish observed event times. No event is inferred before the first observation or after closing. DOWN-to-CLOSED changes remain unavailable and do not count as reopening. These metrics are returned under `evaluate()["transitions"]["closing" or "reopening"]`; the original horizon scores and display are unchanged. Forecasting remains autoregressive and uses the production output masks; a DOWN forecast can represent an unavailable ride even though CLOSED itself is masked.

Select the replay day and the park-local time at which predictions begin in `ml/config.py`.

```python
PREDICT_DAY = "2026-11-11"
PREDICT_START_TIME = "12:00"
```

To print the prediction at a particular park-local time using the saved checkpoint, pass that time to the prediction script.

```powershell
python ml/run.py 18:00
```

The argument is the time to display. With `PREDICT_START_TIME = "17:00"`, `python ml/run.py 18:00` uses real observations through 16:55, generates every five-minute tick from 17:00 through 18:00 autoregressively, and prints the 18:00 prediction. The requested time must be at or after the configured prediction start. It can extend beyond the configured `STEPS` horizon.

`PREDICT_DAY = None` uses the latest cached day, including an incomplete current day. For a noon start, observed data from opening through 11:55 is the history. The noon tick and every following tick are generated autoregressively from the model's own sampled outputs. Every scheduled observation from opening up to that cutoff must be present or eligible for interpolation using endpoints before the cutoff. An unresolved missing bucket or attraction observation raises an error before forecasting. Later missing observations are allowed. Published schedules distinguish scheduled breaks from missing data. Explicit UNKNOWN statuses and unavailable standby waits remain valid source observations. Later actual wait and status observations are excluded from the forecast input. Times between five-minute buckets round up to the next bucket, and the preceding start bucket must contain an observation. Clock times before opening refer to the following overnight portion of the selected operating day.

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

A day enters the training group after closing when every scheduled five-minute bucket and attraction observation is present, including eligible interpolated ticks. When one to four scheduled ticks are missing across the whole operating day, interior gaps are filled by linear interpolation between the known waits immediately before and after each gap. Interpolated waits round to the nearest five minutes, with halfway values rounded up. Status and overflow categories use the nearer endpoint (the later endpoint on a tie). Missing opening or closing ticks, gaps across scheduled breaks, and missing attraction observations are not filled. Days with five or more missing ticks remain incomplete. The cache records each interpolated tick's endpoints, and forecasts reject history that would require an endpoint at or after the prediction start. Running `python ml/data.py` also repairs eligible cached incomplete days in the configured date range without refetching them.

Incomplete and in-progress days remain available to `run.py`. Sessions are grouped by local opening date and include hours past midnight. Writes are atomic, and fetch failures preserve saved data. Old rejected days are fetched again once to recover observations discarded by earlier versions of the exporter. Existing complete-day records and trained checkpoints remain usable. An attraction omitted by the source for an entire day cannot be detected from this archive.

New training runs use `ride_gru_attention_v1`: a shared single-layer GRU with a separate 128-unit hidden state for each ride, followed by two cross-ride attention blocks with four heads each. Each ride's input combines a 64-dimensional wait/status embedding, a learned 64-dimensional ride-ID embedding, and six next-tick calendar features. The input projection maps these to 128 units with ReLU. Each attention block includes residual connections, layer normalization, and a 256-unit GELU feed-forward layer, with 0.1 dropout; final layer normalization and a shared categorical head produce each ride's next-tick distribution. Attention mixes rides only at the same tick, preserving causality. `HIDDEN_SIZE`, `ATTENTION_BLOCKS`, `ATTENTION_HEADS`, and `DROPOUT` are configurable in `ml/config.py`; hidden size must be divisible by the head count.

The model predicts all rides together with one categorical output per ride. Tokens cover five-minute waits from 0 through 300, overflow, DOWN, CLOSED, REFURBISHMENT, and UNKNOWN. Only wait and DOWN targets contribute to loss. Unknown statuses and missing standby waits remain UNKNOWN. Observed refurbishment flags stay locked throughout each forecast. Sampling remains autoregressive: each future tick consumes the preceding generated tokens.

Checkpoints save the architecture and attention settings, so inference reconstructs the trained model independently of current config settings. Existing checkpoints without an architecture field continue to load the original park-wide GRU. Run `python ml/train.py` to train the new architecture; changing config alone does not upgrade an existing checkpoint. Training saves to `CHECKPOINT`, so select a different path first if you want to keep the old weights for comparison.

Training uses only the training partition of complete days in `days`; `PREDICT_DAY` selects a replay and does not affect the split. Each day is a full sequence, with each next-tick prediction conditioned on all earlier observations from that day. Shorter days are padded in a batch and padded targets are excluded from loss. History carries across scheduled breaks, while transitions across those breaks are excluded from the five-minute prediction loss. Each epoch reports training and validation loss and token accuracy and saves the latest model. Test days are scored only by `test.py`. `run.py` loads the saved model to generate predictions. Forecasts can extend past park closing.

Run the regression checks from this folder.

```powershell
python -m unittest discover -s tests -v
node --test tests/test_visualizer_errors.cjs
```
