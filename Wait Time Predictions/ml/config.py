"""Settings for data.py, train.py, test.py, and run.py."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shared paths and runtime. ROOT is the Wait Time Predictions folder.
PARK = "disneyland"
DATASET = ROOT / "data" / PARK / "waits.json.gz"
CHECKPOINT = ROOT / "checkpoints" / PARK / "waits.pt"
DEVICE = "cuda"  # "auto", "cuda", or "cpu"
SEED = 42

# Firestore export. Dates are inclusive local opening dates.
START_DATE = None  # None fetches all history available to the API key for this park.
END_DATE = None  # None includes today, whose observations are available for prediction.
CREDENTIALS = ROOT / "credentials/credentials.json"
if not CREDENTIALS.exists():
    CREDENTIALS = None  # Fall back to FIRESTORE_SERVICE_ACCOUNT_JSON or Google application credentials.
PROJECT = "omniqueue-9ce1b"
REFRESH = False  # True rechecks the selected dates, including previously discarded days.
KEY_FILE = ROOT / "credentials/themeparks-api-key.txt"
LIVE_DAYS = 14  # Firestore lookback when no archive coverage is available.

# Training.
# Fractions must sum to 1. Whole days are randomly assigned using SEED;
# observations within each day stay chronological and never cross splits.
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
EPOCHS = 300
BATCH_SIZE = 64  # Full operating days per batch.
EMBEDDING_SIZE = 64
MODEL_ARCHITECTURE = "ride_horizon_v3"
HIDDEN_SIZE = 128  # Per-ride recurrent state, shared GRU weights.
ATTENTION_BLOCKS = 2  # Cross-ride attention at each observed tick.
ATTENTION_HEADS = 4  # HIDDEN_SIZE must be divisible by this count.
DROPOUT = 0.1
LEARNING_RATE = 0.001
ORDINAL_WEIGHT = 2.0  # Penalize distance between numeric wait distributions.
ROLLOUT_WEIGHT = 0.0  # Optional generated-input training for autoregressive v1/v2.
ROLLOUT_STEPS = 12  # Up to one hour, never across a break or padding.
ROLLOUT_ORIGINS = 2  # Random training / fixed validation origins per day.
ROLLOUT_WARMUP_EPOCHS = 20  # Linearly introduce rollout training.
EARLY_STOPPING_PATIENCE = 10  # 0 disables stopping; best validation weights are kept.
HORIZON_WEIGHT = 1.0  # Direct forecasts avoid compounding five-minute errors.
HORIZON_STEPS = (1, 3, 6, 12, 24, 48, 96, 144)  # 5 min through 12 h.
HORIZON_ORIGINS = 8  # Random training / fixed validation origins per day and lead.

# Forecasting. Used by run.py.
PREDICT_DAY = None # None selects the latest cached day, including an incomplete day.
PREDICT_START_TIME = None # Park-local HH:MM. None starts after the last observation.
HISTORY = None  # Optional Path to recent snapshots in the archive viewer's JSON format.
STEPS = None  # None predicts the rest of the recorded day, or 24 ticks for live history.
SAMPLES = 100
TEMPERATURE = 1.0
