# ---------------------------------------------------------------------------
# Padded decision sizes (action / obs contract)
# ---------------------------------------------------------------------------

# Max free vehicles in one coordinated wave (padded).
MAX_VEHICLES = 64
# Max pending request candidates pointed at in one decision (padded).
MAX_REQUESTS = 128
# Special actions after request slots: STAY, IDLE (no chargers in v1).
NUM_SPECIAL_ACTIONS = 2
ACTION_STAY = MAX_REQUESTS  # index of STAY
ACTION_IDLE = MAX_REQUESTS + 1  # index of IDLE
NUM_ACTIONS = MAX_REQUESTS + NUM_SPECIAL_ACTIONS

# ---------------------------------------------------------------------------
# Feature dimensions (must match C++ FleetEnv / _fleet_sim)
# ---------------------------------------------------------------------------

VEHICLE_DYNAMIC_FEAT_DIM = 8
INTERSECTION_DYNAMIC_FEAT_DIM = 2  # Num waiting at intersection, total waiting time
REQUEST_DYNAMIC_FEAT_DIM = 8
PAIRWISE_DYNAMIC_FEAT_DIM = 4  # Pairwise vehicle ↔ request (drive time / dist / energy / reachable).
ENV_DYNAMIC_FEAT_DIM = 4  # Global env: time-of-day, backlog, fleet SOC mean, free-vehicle fraction.

# Flat obs from FleetEnv (single deciding vehicle V=1). Must match FleetEnv.hpp.
FLAT_OBS_DIM = (
    VEHICLE_DYNAMIC_FEAT_DIM
    + MAX_REQUESTS * REQUEST_DYNAMIC_FEAT_DIM
    + MAX_REQUESTS * PAIRWISE_DYNAMIC_FEAT_DIM
    + ENV_DYNAMIC_FEAT_DIM
    + MAX_REQUESTS  # request mask
    + NUM_ACTIONS  # action mask
    + 1  # vehicle node index
    + MAX_REQUESTS  # request origin indices
    + MAX_REQUESTS  # request dest indices
)

# ---------------------------------------------------------------------------
# Model architecture defaults
# ---------------------------------------------------------------------------

D_MODEL = 128
NUM_TRANSFORMER_LAYERS = 2
NUM_ATTN_HEADS = 4
NUM_GNN_LAYERS = 2
LAPLACIAN_PE_DIM = 16  # Laplacian positional encoding: # of smallest non-zero eigenmodes per node
MAX_SHORTEST_PATH_DIST = 64
# Upper bound for optional node-id embedding fallback when GNN is disabled.
MAX_NODES = 4096

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

PPO_SAVE_DIR = "checkpoints/ppo"
PPO_SAVE_EVERY = 50
PPO_LEARNING_RATE = 3e-3
PPO_ANNEAL_LR = True
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_NUM_MINIBATCHES = 4
PPO_UPDATE_EPOCHS = 2
PPO_CLIP_COEF = 0.2
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_MAX_STEPS_PER_EPISODE = 50_000
PPO_TARGET_KL = 0.02  # early-stop PPO epochs when approx KL exceeds this
PPO_NUM_ENVS = 8
PPO_LOG_EVERY = 1

# Default sim scale (override via CLI)
PPO_NUM_VEHICLES = 30
PPO_NUM_REQUESTS = 120
PPO_NUM_INTERSECTIONS = 80
PPO_HORIZON_SEC = 3600
