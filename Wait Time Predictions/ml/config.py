"""Settings for data.py, train.py, test.py, and run.py."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shared paths and runtime. ROOT is the Wait Time Predictions folder.
DATASET = ROOT / "data" / "waits.json.gz"
CHECKPOINT = ROOT / "checkpoints" / "waits.pt"
DEVICE = "cuda"  # "auto", "cuda", or "cpu"
SEED = 42

# Firestore export. Dates are inclusive local opening dates.
START_DATE = "2026-11-10"
END_DATE = None  # None includes today, whose observations are available for prediction.
CREDENTIALS = None  # Path("C:/path/to/service-account.json"), or use environment credentials.
PROJECT = "omniqueue-9ce1b"
REFRESH = False  # True rechecks the selected dates, including previously discarded days.

# Training.
# Fractions must sum to 1. Whole days are split chronologically:
# oldest days train, then validation, then newest days test.
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
EPOCHS = 200
BATCH_SIZE = 64  # Full operating days per batch.
EMBEDDING_SIZE = 64
MODEL_ARCHITECTURE = "ride_gru_attention_v1"
HIDDEN_SIZE = 128  # Per-ride recurrent state, shared GRU weights.
ATTENTION_BLOCKS = 2  # Cross-ride attention at each observed tick.
ATTENTION_HEADS = 4  # HIDDEN_SIZE must be divisible by this count.
DROPOUT = 0.1
LEARNING_RATE = 0.001

# Forecasting. Used by run.py.
PREDICT_DAY = "2026-11-15" # None selects the latest cached day, including an incomplete day.
PREDICT_START_TIME = "20:30" # Park-local HH:MM. None starts after the last observation.
HISTORY = None  # Optional Path to recent snapshots in the archive viewer's JSON format.
STEPS = None  # None predicts the rest of the recorded day, or 24 ticks for live history.
SAMPLES = 100
TEMPERATURE = 1.0
