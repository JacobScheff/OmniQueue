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
# Feature dimensions (must match obs adapters / future C++ build_observation)
# ---------------------------------------------------------------------------

VEHICLE_DYNAMIC_FEAT_DIM = 8
INTERSECTION_DYNAMIC_FEAT_DIM = 2 # Num waiting at intersection, total waiting time
REQUEST_DYNAMIC_FEAT_DIM = 8

# ---------------------------------------------------------------------------
# Model architecture defaults
# ---------------------------------------------------------------------------

D_MODEL = 256
NUM_TRANSFORMER_LAYERS = 8
NUM_ATTN_HEADS = 8
NUM_GNN_LAYERS = 8
LAPLACIAN_PE_DIM = 16 # Laplacian positional encoding: # of smallest non-zero eigenmodes per node
MAX_SHORTEST_PATH_DIST = 64
# Upper bound for optional node-id embedding fallback when GNN is disabled.
MAX_NODES = 4096