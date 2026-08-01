"""Park configuration: rides, graph topology, spawn parameters, router settings."""

from __future__ import annotations

import math

# --- Time ---
DAY_START_HOUR = 8
DAY_END_HOUR = 23  # 11 PM close; last second is 22:59:59 -> 54000 seconds
DAY_SECONDS = (DAY_END_HOUR - DAY_START_HOUR) * 3600  # 54000

# --- Population ---
TOTAL_GUESTS_MEAN = 50_000
TOTAL_GUESTS_STD = 2_500
PARTY_SIZE_MEAN = 3.2
PARTY_SIZE_STD = 1.0
# Arrival mixture: rope-drop rush + remainder through the operating day.
SPAWN_RUSH_FRACTION = 0.65          # share of guests in the opening rush
SPAWN_RUSH_MEAN_SEC = 8 * 60        # rush peak ~8 min after open
SPAWN_RUSH_STD_SEC = 12 * 60
SPAWN_RUSH_CLAMP_SEC = 2 * 3600     # rush arrivals clamped to first 2 hours
SPAWN_DAY_MEAN_SEC = 6 * 3600       # non-rush peak ~2:00 PM
SPAWN_DAY_STD_SEC = int(3.5 * 3600)
DWELL_MEAN_SEC = 14 * 3600          # longer stays → busier near close
DWELL_STD_SEC = int(2.5 * 3600)
MIN_DWELL_SEC = 2 * 3600
# After official close, parties already in queue/on-ride finish, then exit.
CLOSE_DRAIN_SEC = 3 * 3600

# --- Walking ---
# Physical walking pace in meters/second. Pathway edge lengths from OSM are in
# meters; legacy Euclidean fallback uses the same numeric speed on display units.
BASE_WALKING_SPEED = 1.4
MEMBER_SPEED_LOG_MU = math.log(1.4)
MEMBER_SPEED_LOG_SIGMA = 0.25

# Near-shortest path randomization (spreads corridor load without density maps).
# When enabled, each walk samples among OSM paths within LENGTH_SLACK of shortest;
# pick probability ∝ exp(-(walk_sec - shortest_sec) / SOFTMAX_TAU_SEC).
WALK_PATH_RANDOM = True
WALK_PATH_MAX_VARIANTS = 6
WALK_PATH_LENGTH_SLACK = 0.15  # allow paths up to 15% longer than shortest
WALK_PATH_SOFTMAX_TAU_SEC = 45.0  # larger → more uniform among near-ties

# --- Breakdown / evacuation ---
BREAKDOWN_REPAIR_MIN_SEC = 15 * 60
BREAKDOWN_REPAIR_MAX_SEC = 60 * 60
EVAC_INTERVAL_SEC = 4

# --- Heuristic router ---
ROUTER = "heuristic"  # "heuristic" | "ppo"
BASE_BALK_SEC = 40 * 60   # 40 min floor; typical rides stay near this
BALK_SCALE = 5 * 60       # +0–5 min by preference^BALK_PREF_EXP (max ~45 min)
BALK_PREF_EXP = 1.5
MUST_DO_PREF_BOOST = 10.0
# Default spawn (play/watch/visualize/heuristic days):
#   raw[r] = popularity[r] * U(1-noise, 1+noise), must-do boost, L1-normalize.
PREF_POPULARITY_NOISE = 0.25
# Training-only spawn (BC / personal PPO): i.i.d. U(PREF_RAW_EPS, 1), must-do boost, L1-normalize.
PREF_RAW_EPS = 1e-3
IDLE_WALK_PROB = 0.5
IDLE_MAX_HOPS = 2
MAX_ROUTE_BATCH = 256
# Ride-repeat dampening (mirrored in native/include/park_sim.hpp)
REPEAT_TOP_K = 3                 # Pass 2: allow repeats for top-K preference ranks
REPEAT_PREF_THRESHOLD = 0.04     # Pass 2: or if normalized pref >= this
REPEAT_PREF_SCALE = 2.0          # max_repeats = 1 + floor(scale * pref * NUM_RIDES)
REPEAT_MAX = 3                   # hard cap on preferred-repeat completions
REPEAT_BALK_FACTOR = 1.0         # multiply balk threshold for Pass 2 (1.0 = same as fresh)
SHORT_WAIT_SEC = 12 * 60         # Pass 3: absolute short-wait bar
SHORT_WAIT_SLACK_SEC = 2 * 60    # Pass 3: accept within slack of best feasible wait

# --- Metrics ---
METRICS_SAMPLE_INTERVAL_SEC = 300

# --- Graph node ids ---
NODE_ENTRANCE = 0
NODE_MAIN_HUB = 1
NODE_GALAXY_HUB = 2
NODE_CRITTER_HUB = 3
NODE_NEW_ORLEANS_HUB = 4
NODE_ADVENTURE_HUB = 5
NODE_FRONTIER_HUB = 6
NODE_FANTASY_HUB = 7
NODE_TOONTOWN_HUB = 8
NODE_TOMORROW_HUB = 9
# Ride nodes start at 100 + ride_id
RIDE_NODE_OFFSET = 100

NUM_RIDES = 34

# coords in abstract park units (1000x1000-ish layout)
# Relative demand weights (not absolute guest counts). Sourced from the legacy
# Disneyland popularity table; used at spawn to bias party preferences / must-dos.
RIDES: list[dict] = [
    {"name": "Star Wars: Rise of the Resistance", "capacity_per_hour": 1200, "duration_sec": 18 * 60, "breakdown_prob_per_hour": 0.003, "popularity": 220, "coords": (100, 250)},
    {"name": "Millennium Falcon: Smugglers Run", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.001, "popularity": 140, "coords": (200, 200)},
    {"name": "Tiana's Bayou Adventure", "capacity_per_hour": 1800, "duration_sec": 10 * 60, "breakdown_prob_per_hour": 0.0015, "popularity": 150, "coords": (100, 600)},
    {"name": "The Many Adventures of Winnie the Pooh", "capacity_per_hour": 1000, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 40, "coords": (60, 650)},
    {"name": "Davy Crockett's Explorer Canoes", "capacity_per_hour": 600, "duration_sec": 10 * 60, "breakdown_prob_per_hour": 0.0001, "popularity": 15, "coords": (120, 650)},
    {"name": "Haunted Mansion", "capacity_per_hour": 2000, "duration_sec": 9 * 60, "breakdown_prob_per_hour": 0.0005, "popularity": 180, "coords": (150, 500)},
    {"name": "Pirates of the Caribbean", "capacity_per_hour": 2800, "duration_sec": 15 * 60, "breakdown_prob_per_hour": 0.0005, "popularity": 230, "coords": (200, 550)},
    {"name": "Indiana Jones Adventure", "capacity_per_hour": 1600, "duration_sec": 4 * 60, "breakdown_prob_per_hour": 0.002, "popularity": 190, "coords": (250, 650)},
    {"name": "Jungle Cruise", "capacity_per_hour": 1800, "duration_sec": 8 * 60, "breakdown_prob_per_hour": 0.0005, "popularity": 130, "coords": (320, 620)},
    {"name": "Walt Disney's Enchanted Tiki Room", "capacity_per_hour": 1200, "duration_sec": 15 * 60, "breakdown_prob_per_hour": 0.0001, "popularity": 30, "coords": (380, 650)},
    {"name": "Big Thunder Mountain Railroad", "capacity_per_hour": 2000, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.001, "popularity": 160, "coords": (350, 450)},
    {"name": "Mark Twain Riverboat", "capacity_per_hour": 1500, "duration_sec": 14 * 60, "breakdown_prob_per_hour": 0.0001, "popularity": 40, "coords": (280, 500)},
    {"name": "Sailing Ship Columbia", "capacity_per_hour": 1200, "duration_sec": 14 * 60, "breakdown_prob_per_hour": 0.0001, "popularity": 35, "coords": (280, 450)},
    {"name": "Matterhorn Bobsleds", "capacity_per_hour": 1500, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0015, "popularity": 120, "coords": (650, 400)},
    {"name": "Peter Pan's Flight", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 90, "coords": (480, 450)},
    {"name": "Mr. Toad's Wild Ride", "capacity_per_hour": 800, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 45, "coords": (520, 450)},
    {"name": "Snow White's Enchanted Wish", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 50, "coords": (460, 470)},
    {"name": "Pinocchio's Daring Journey", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 40, "coords": (440, 440)},
    {"name": "King Arthur Carrousel", "capacity_per_hour": 1000, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0001, "popularity": 30, "coords": (500, 400)},
    {"name": "Dumbo the Flying Elephant", "capacity_per_hour": 700, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0001, "popularity": 55, "coords": (550, 350)},
    {"name": "Mad Tea Party", "capacity_per_hour": 900, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0001, "popularity": 50, "coords": (600, 430)},
    {"name": "Alice in Wonderland", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 60, "coords": (580, 380)},
    {"name": "Casey Jr. Circus Train", "capacity_per_hour": 700, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 35, "coords": (430, 350)},
    {"name": "Storybook Land Canal Boats", "capacity_per_hour": 700, "duration_sec": 6 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 30, "coords": (480, 330)},
    {"name": "it's a small world", "capacity_per_hour": 2500, "duration_sec": 14 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 135, "coords": (500, 250)},
    {"name": "Mickey and Minnie's Runaway Railway", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.002, "popularity": 150, "coords": (500, 150)},
    {"name": "Roger Rabbit's Car Toon Spin", "capacity_per_hour": 1200, "duration_sec": 4 * 60, "breakdown_prob_per_hour": 0.0005, "popularity": 70, "coords": (430, 180)},
    {"name": "Chip 'n' Dale's GADGETcoaster", "capacity_per_hour": 800, "duration_sec": 60, "breakdown_prob_per_hour": 0.0005, "popularity": 45, "coords": (570, 180)},
    {"name": "Space Mountain", "capacity_per_hour": 1800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0015, "popularity": 190, "coords": (850, 550)},
    {"name": "Star Tours", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.001, "popularity": 110, "coords": (750, 580)},
    {"name": "Buzz Lightyear Astro Blasters", "capacity_per_hour": 2000, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.0005, "popularity": 130, "coords": (700, 630)},
    {"name": "Astro Orbitor", "capacity_per_hour": 600, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0002, "popularity": 35, "coords": (750, 650)},
    {"name": "Autopia", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.0005, "popularity": 90, "coords": (850, 450)},
    {"name": "Finding Nemo Submarine Voyage", "capacity_per_hour": 1000, "duration_sec": 13 * 60, "breakdown_prob_per_hour": 0.0005, "popularity": 70, "coords": (780, 480)},
]

ENTRANCE_COORDS = (500, 900)

# Hub coordinates for macro graph
HUB_COORDS: dict[int, tuple[float, float]] = {
    NODE_ENTRANCE: ENTRANCE_COORDS,
    NODE_MAIN_HUB: (500, 800),
    NODE_GALAXY_HUB: (150, 220),
    NODE_CRITTER_HUB: (90, 620),
    NODE_NEW_ORLEANS_HUB: (175, 520),
    NODE_ADVENTURE_HUB: (300, 640),
    NODE_FRONTIER_HUB: (310, 470),
    NODE_FANTASY_HUB: (520, 380),
    NODE_TOONTOWN_HUB: (500, 165),
    NODE_TOMORROW_HUB: (780, 560),
}

# Each ride maps to its land hub (by index in RIDES list)
RIDE_HUB: list[int] = [
    NODE_GALAXY_HUB, NODE_GALAXY_HUB,
    NODE_CRITTER_HUB, NODE_CRITTER_HUB, NODE_CRITTER_HUB,
    NODE_NEW_ORLEANS_HUB, NODE_NEW_ORLEANS_HUB,
    NODE_ADVENTURE_HUB, NODE_ADVENTURE_HUB, NODE_ADVENTURE_HUB,
    NODE_FRONTIER_HUB, NODE_FRONTIER_HUB, NODE_FRONTIER_HUB,
    NODE_FANTASY_HUB, NODE_FANTASY_HUB, NODE_FANTASY_HUB, NODE_FANTASY_HUB,
    NODE_FANTASY_HUB, NODE_FANTASY_HUB, NODE_FANTASY_HUB, NODE_FANTASY_HUB,
    NODE_FANTASY_HUB, NODE_FANTASY_HUB, NODE_FANTASY_HUB, NODE_FANTASY_HUB,
    NODE_TOONTOWN_HUB, NODE_TOONTOWN_HUB, NODE_TOONTOWN_HUB,
    NODE_TOMORROW_HUB, NODE_TOMORROW_HUB, NODE_TOMORROW_HUB,
    NODE_TOMORROW_HUB, NODE_TOMORROW_HUB, NODE_TOMORROW_HUB,
]

# Macro edges: (node_a, node_b) bidirectional idle-wander topology.
# Walk times use OSM pathway meters when data/pathways.json is present;
# otherwise Euclidean distance on display coords.
MACRO_EDGES: list[tuple[int, int]] = [
    (NODE_ENTRANCE, NODE_MAIN_HUB),
    (NODE_MAIN_HUB, NODE_GALAXY_HUB),
    (NODE_MAIN_HUB, NODE_NEW_ORLEANS_HUB),
    (NODE_MAIN_HUB, NODE_FRONTIER_HUB),
    (NODE_MAIN_HUB, NODE_FANTASY_HUB),
    (NODE_MAIN_HUB, NODE_TOONTOWN_HUB),
    (NODE_MAIN_HUB, NODE_TOMORROW_HUB),
    (NODE_NEW_ORLEANS_HUB, NODE_CRITTER_HUB),
    (NODE_NEW_ORLEANS_HUB, NODE_ADVENTURE_HUB),
    (NODE_NEW_ORLEANS_HUB, NODE_FRONTIER_HUB),
    (NODE_ADVENTURE_HUB, NODE_CRITTER_HUB),
    (NODE_FRONTIER_HUB, NODE_FANTASY_HUB),
    (NODE_FANTASY_HUB, NODE_TOONTOWN_HUB),
    (NODE_GALAXY_HUB, NODE_CRITTER_HUB),
    (NODE_GALAXY_HUB, NODE_NEW_ORLEANS_HUB),
    (NODE_TOMORROW_HUB, NODE_FANTASY_HUB),
    (NODE_TOMORROW_HUB, NODE_FRONTIER_HUB),
    # Waypoints for river / castle chokepoints
]

# Extra waypoint nodes to avoid unrealistic straight-line shortcuts
NODE_RIVER_CROSSING = 10
NODE_CENTRAL_PLAZA = 11

HUB_COORDS[NODE_RIVER_CROSSING] = (260, 540)
HUB_COORDS[NODE_CENTRAL_PLAZA] = (500, 550)

MACRO_EDGES.extend([
    (NODE_NEW_ORLEANS_HUB, NODE_RIVER_CROSSING),
    (NODE_RIVER_CROSSING, NODE_FRONTIER_HUB),
    (NODE_RIVER_CROSSING, NODE_ADVENTURE_HUB),
    (NODE_MAIN_HUB, NODE_CENTRAL_PLAZA),
    (NODE_CENTRAL_PLAZA, NODE_FANTASY_HUB),
    (NODE_CENTRAL_PLAZA, NODE_TOMORROW_HUB),
    (NODE_CENTRAL_PLAZA, NODE_NEW_ORLEANS_HUB),
])

# Display coords are overlaid from data/pathways.json in park_graph.get_park_graph().
# Fallback abstract coords above remain for environments without the pathways file.

# ---------------------------------------------------------------------------
# Behavioral Cloning (Phase 2) training defaults
# ---------------------------------------------------------------------------
BC_SAVE_DIR = "checkpoints/bc"
BC_SAVE_EVERY = 5_000      # save checkpoint every N optimizer steps
BC_EPOCHS = 10
# Batch size is in individual routing decisions (single-party model, no G axis).
BC_BATCH_SIZE = 256
BC_LR = 3e-4

# ---------------------------------------------------------------------------
# PPO (Phase 3) training defaults — personal focal planner
# Override individual values here rather than via command-line flags.
# ---------------------------------------------------------------------------
PPO_SAVE_DIR = "checkpoints/ppo"
PPO_SAVE_EVERY = 5_000           # save checkpoint every N routing steps
PPO_LEARNING_RATE = 1e-4
PPO_ANNEAL_LR = True              # linearly decay LR over total_days
PPO_GAMMA = 0.999                 # discount factor
PPO_GAE_LAMBDA = 0.95             # GAE lambda
PPO_NUM_MINIBATCHES = 8           # PPO minibatch count per update
PPO_UPDATE_EPOCHS = 2             # PPO epochs per day
PPO_CLIP_COEF = 0.1               # PPO clipping epsilon
PPO_ENT_COEF = 0.03               # entropy bonus coefficient (early route slots weighted)
PPO_VF_COEF = 0.5                 # value loss coefficient
PPO_MAX_GRAD_NORM = 0.5           # gradient clipping norm
PPO_SUBSAMPLE_SIZE = 262_144      # random transitions per day used for update
PPO_MAX_ROUTING_STEPS = 600_000   # safety cap on routing decisions per day
# N focal parties trained against a heuristic crowd on the same day.
PPO_NUM_FOCALS = 24
# C++ may return up to this many pending focal observations per exchange.
PPO_INFERENCE_BATCH_SIZE = 256
# Transitions packed into one neural forward / optimizer step during PPO update.
PPO_UPDATE_MB_SIZE = 256
# Pause after each optimizer step so Windows can composite the desktop.
PPO_UPDATE_YIELD_SEC = 0.05
PPO_LOG_EVERY = 5_000             # rollout progress log interval (0 = disabled)

# Multi-ride route output (see docs/route-plan-output-plan.md)
PPO_ROUTE_K = 6
PPO_ROUTE_CONSIST_COEF = 0.02
PPO_ROUTE_CONSIST_WEIGHTS = (1.0, 0.5, 0.25, 0.1, 0.05)  # len K-1; new[i] vs old[i+1]
PPO_ROUTE_PLANNED_WALK_COEF = 0.01
PPO_ROUTE_REALIZED_WALK_COEF = 0.02
PPO_ROUTE_WALK_NORM_SEC = 600.0
# Entropy weights per route slot (early slots explore more)
PPO_ROUTE_ENTROPY_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.15, 0.1)
# Counterfactual preference KL (hinge on JS divergence of slot-0 dists)
PPO_CF_COEF = 0.1
PPO_CF_MARGIN = 0.15
PPO_CF_FRAC = 0.25
MODEL_ARCH_VERSION = "route_v1"

# PPO reward shaping (mirrored in park_sim.hpp)
# Objective: preferred + must-do rides done quickly per party (guest-weighted).
# Wait variance is KPI/logging only — not in the training reward.
#
# On RideComplete (pending, flushed on that party's next routing step):
#   time_factor = max(0, 1 - PPO_TIME_DECAY * (now - spawn) / DAY_SECONDS)
#   bonus = (PPO_PREF_REWARD_SCALE * preference[ride]
#            + PPO_MUST_DO_COMPLETION_BONUS if must-do) * time_factor
#   if PPO_WEIGHT_BY_PARTY_SIZE: bonus *= party_size
# Every routing step (party-local):
#   -PPO_MUST_DO_URGENCY_COEF * remaining_must_dos
#   -PPO_PREF_URGENCY_COEF * remaining_pref_mass  (prefs for rides with history==0)
# Terminal (focals only in personal mode):
#   -PPO_UNFULFILLED_MUST_DO_PENALTY * (focal_remaining / max(1, focal_assigned))
#   + flush remaining pending bonuses for learners
# Python route shaping (added on top of C++ delta; see training/route_reward.py):
#   + PPO_ROUTE_CONSIST_COEF * Σ w_i * 1[new[i]==old[i+1]]
#   - PPO_ROUTE_PLANNED_WALK_COEF * mean_inter_ride_walk / WALK_NORM
#   - PPO_ROUTE_REALIZED_WALK_COEF * walk_to_commit / WALK_NORM
PPO_PREF_REWARD_SCALE = 0.05
PPO_MUST_DO_COMPLETION_BONUS = 0.15
PPO_TIME_DECAY = 0.75
PPO_MUST_DO_URGENCY_COEF = 2e-5
PPO_PREF_URGENCY_COEF = 1e-5
PPO_WEIGHT_BY_PARTY_SIZE = True
PPO_UNFULFILLED_MUST_DO_PENALTY = 2.0


def ride_node_id(ride_id: int) -> int:
    return RIDE_NODE_OFFSET + ride_id


def get_ride_configs() -> list[dict]:
    """Return ride configs enriched with derived fields."""
    configs = []
    for i, ride in enumerate(RIDES):
        cfg = dict(ride)
        cfg["ride_id"] = i
        cfg["node_id"] = ride_node_id(i)
        cfg["hub_id"] = RIDE_HUB[i]
        cfg["capacity_per_sec"] = ride["capacity_per_hour"] / 3600.0
        cfg["breakdown_prob_sec"] = 1.0 - (1.0 - ride["breakdown_prob_per_hour"]) ** (1.0 / 3600.0)
        configs.append(cfg)
    return configs
