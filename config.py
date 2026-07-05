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
SPAWN_MEAN_SEC = 3 * 3600  # peak arrivals ~11 AM (3h after open)
SPAWN_STD_SEC = 2 * 3600
DWELL_MEAN_SEC = 10 * 3600
DWELL_STD_SEC = 2 * 3600
MIN_DWELL_SEC = 2 * 3600

# --- Walking ---
BASE_WALKING_SPEED = 1.4  # graph units per second (~typical walking pace scale)
MEMBER_SPEED_LOG_MU = math.log(1.4)
MEMBER_SPEED_LOG_SIGMA = 0.25

# --- Breakdown / evacuation ---
BREAKDOWN_REPAIR_MIN_SEC = 15 * 60
BREAKDOWN_REPAIR_MAX_SEC = 60 * 60
EVAC_INTERVAL_SEC = 4

# --- Heuristic router ---
ROUTER = "heuristic"  # "heuristic" | "ppo"
BASE_BALK_SEC = 600
BALK_SCALE = 2400
BALK_PREF_EXP = 1.5
MUST_DO_PREF_BOOST = 10.0
IDLE_WALK_PROB = 0.5
IDLE_MAX_HOPS = 2
FORCE_PICK_IDLE_SEC = 60  # nominal time for forced pick before re-eval
MAX_ROUTE_BATCH = 256

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

NUM_RIDES = 35

# coords in abstract park units (1000x1000-ish layout)
RIDES: list[dict] = [
    {"name": "Star Wars: Rise of the Resistance", "capacity_per_hour": 1200, "duration_sec": 18 * 60, "breakdown_prob_per_hour": 0.003, "coords": (100, 250)},
    {"name": "Millennium Falcon: Smugglers Run", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.001, "coords": (200, 200)},
    {"name": "Tiana's Bayou Adventure", "capacity_per_hour": 1800, "duration_sec": 10 * 60, "breakdown_prob_per_hour": 0.0015, "coords": (100, 600)},
    {"name": "The Many Adventures of Winnie the Pooh", "capacity_per_hour": 1000, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (60, 650)},
    {"name": "Davy Crockett's Explorer Canoes", "capacity_per_hour": 600, "duration_sec": 10 * 60, "breakdown_prob_per_hour": 0.0001, "coords": (120, 650)},
    {"name": "Haunted Mansion", "capacity_per_hour": 2000, "duration_sec": 9 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (150, 500)},
    {"name": "Pirates of the Caribbean", "capacity_per_hour": 2800, "duration_sec": 15 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (200, 550)},
    {"name": "Indiana Jones Adventure", "capacity_per_hour": 1600, "duration_sec": 4 * 60, "breakdown_prob_per_hour": 0.002, "coords": (250, 650)},
    {"name": "Jungle Cruise", "capacity_per_hour": 1800, "duration_sec": 8 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (320, 620)},
    {"name": "Walt Disney's Enchanted Tiki Room", "capacity_per_hour": 1200, "duration_sec": 15 * 60, "breakdown_prob_per_hour": 0.0001, "coords": (380, 650)},
    {"name": "Big Thunder Mountain Railroad", "capacity_per_hour": 2000, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.001, "coords": (350, 450)},
    {"name": "Mark Twain Riverboat", "capacity_per_hour": 1500, "duration_sec": 14 * 60, "breakdown_prob_per_hour": 0.0001, "coords": (280, 500)},
    {"name": "Sailing Ship Columbia", "capacity_per_hour": 1200, "duration_sec": 14 * 60, "breakdown_prob_per_hour": 0.0001, "coords": (280, 450)},
    {"name": "Matterhorn Bobsleds", "capacity_per_hour": 1500, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0015, "coords": (650, 400)},
    {"name": "Peter Pan's Flight", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (480, 450)},
    {"name": "Mr. Toad's Wild Ride", "capacity_per_hour": 800, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (520, 450)},
    {"name": "Snow White's Enchanted Wish", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (460, 470)},
    {"name": "Pinocchio's Daring Journey", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (440, 440)},
    {"name": "King Arthur Carrousel", "capacity_per_hour": 1000, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0001, "coords": (500, 400)},
    {"name": "Dumbo the Flying Elephant", "capacity_per_hour": 700, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0001, "coords": (550, 350)},
    {"name": "Mad Tea Party", "capacity_per_hour": 900, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0001, "coords": (600, 430)},
    {"name": "Alice in Wonderland", "capacity_per_hour": 800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (580, 380)},
    {"name": "Casey Jr. Circus Train", "capacity_per_hour": 700, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (430, 350)},
    {"name": "Storybook Land Canal Boats", "capacity_per_hour": 700, "duration_sec": 6 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (480, 330)},
    {"name": "it's a small world", "capacity_per_hour": 2500, "duration_sec": 14 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (500, 250)},
    {"name": "Mickey and Minnie's Runaway Railway", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.002, "coords": (500, 150)},
    {"name": "Roger Rabbit's Car Toon Spin", "capacity_per_hour": 1200, "duration_sec": 4 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (430, 180)},
    {"name": "Chip 'n' Dale's GADGETcoaster", "capacity_per_hour": 800, "duration_sec": 60, "breakdown_prob_per_hour": 0.0005, "coords": (570, 180)},
    {"name": "Hyperspace Mountain", "capacity_per_hour": 1800, "duration_sec": 3 * 60, "breakdown_prob_per_hour": 0.0015, "coords": (850, 550)},
    {"name": "Star Tours", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.001, "coords": (750, 580)},
    {"name": "Buzz Lightyear Astro Blasters", "capacity_per_hour": 2000, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (700, 630)},
    {"name": "Astro Orbitor", "capacity_per_hour": 600, "duration_sec": 2 * 60, "breakdown_prob_per_hour": 0.0002, "coords": (750, 650)},
    {"name": "Autopia", "capacity_per_hour": 1800, "duration_sec": 5 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (850, 450)},
    {"name": "Finding Nemo Submarine Voyage", "capacity_per_hour": 1000, "duration_sec": 13 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (780, 480)},
    {"name": "Disneyland Monorail", "capacity_per_hour": 1000, "duration_sec": 15 * 60, "breakdown_prob_per_hour": 0.0005, "coords": (800, 500)},
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
    NODE_TOMORROW_HUB, NODE_TOMORROW_HUB, NODE_TOMORROW_HUB, NODE_TOMORROW_HUB,
]

# Macro edges: (node_a, node_b) bidirectional; weight = Euclidean distance
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


def ride_node_id(ride_id: int) -> int:
    return RIDE_NODE_OFFSET + ride_id


def ride_id_from_node(node_id: int) -> int | None:
    if node_id >= RIDE_NODE_OFFSET:
        return node_id - RIDE_NODE_OFFSET
    return None


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
