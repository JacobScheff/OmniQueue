#!/usr/bin/env python3
"""Export Park catalog + walk times into the iOS app bundle.

Run from the OmniQueue repo root:

    python3 "Companion app/export_park_data.py"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Park import config  # noqa: E402
from Park.companion.server.ride_map import (  # noqa: E402
    DISNEYLAND_PARK_ENTITY_ID,
    KNOWN_ENTITY_IDS,
    NAME_ALIASES,
    hub_display_name,
)
from Park.park_graph import get_park_graph  # noqa: E402
from Park.training.features import (  # noqa: E402
    ENV_DYNAMIC_FEAT_DIM,
    FLAT_OBS_DIM,
    GUEST_FEAT_DIM,
    NUM_ACTIONS,
    NUM_RIDES,
    RIDE_DYNAMIC_FEAT_DIM,
)

OUT = Path(__file__).resolve().parent / "OmniQueueCompanion" / "Resources" / "ParkData.json"
WEIGHT_SLIDER_MAX = 250.0


def main() -> None:
    park = get_park_graph()
    prefs = [float(min(max(r["popularity"], 0.0), WEIGHT_SLIDER_MAX)) for r in config.RIDES]
    action_labels = [r["name"] for r in config.RIDES] + ["Exit park", "Idle wander"]

    hubs = []
    seen: set[str] = set()
    for hub_id, _coords in sorted(config.HUB_COORDS.items()):
        key = "entrance" if hub_id == config.NODE_ENTRANCE else f"hub:{hub_id}"
        if key in seen:
            continue
        seen.add(key)
        walks = park.walk_times_to_rides(hub_id, config.BASE_WALKING_SPEED).tolist()
        hubs.append(
            {
                "id": hub_id,
                "key": key,
                "name": hub_display_name(hub_id),
                "kind": "entrance" if hub_id == config.NODE_ENTRANCE else "hub",
                "node_id": hub_id,
                "node_idx": int(park.node_to_idx(hub_id)),
                "at_ride": -1,
                "walk_sec": [int(x) for x in walks],
            }
        )

    rides = []
    for i, ride in enumerate(config.RIDES):
        node_id = config.ride_node_id(i)
        walks = park.walk_times_to_rides(node_id, config.BASE_WALKING_SPEED).tolist()
        rides.append(
            {
                "id": i,
                "name": ride["name"],
                "hub_id": config.RIDE_HUB[i],
                "hub_name": hub_display_name(config.RIDE_HUB[i]),
                "location_key": f"ride:{i}",
                "popularity": ride["popularity"],
                "duration_sec": ride["duration_sec"],
                "duration_min": ride["duration_sec"] / 60.0,
                "capacity_per_hour": ride["capacity_per_hour"],
                "node_id": node_id,
                "node_idx": int(park.node_to_idx(node_id)),
                "entity_id": KNOWN_ENTITY_IDS.get(i),
                "walk_sec": [int(x) for x in walks],
            }
        )

    payload = {
        "num_rides": NUM_RIDES,
        "num_actions": NUM_ACTIONS,
        "guest_feat_dim": GUEST_FEAT_DIM,
        "ride_feat_dim": RIDE_DYNAMIC_FEAT_DIM,
        "env_feat_dim": ENV_DYNAMIC_FEAT_DIM,
        "flat_obs_dim": FLAT_OBS_DIM,
        "route_k": int(config.PPO_ROUTE_K),
        "day_start_hour": config.DAY_START_HOUR,
        "day_end_hour": config.DAY_END_HOUR,
        "day_seconds": int(config.DAY_SECONDS),
        "close_drain_sec": int(config.CLOSE_DRAIN_SEC),
        "base_walking_speed": float(config.BASE_WALKING_SPEED),
        "pref_reward_exp": float(config.PPO_PREF_REWARD_EXP),
        "weight_slider_max": float(WEIGHT_SLIDER_MAX),
        "num_nodes": int(park.num_nodes),
        "park_entity_id": DISNEYLAND_PARK_ENTITY_ID,
        "wait_api_base": "https://api.themeparks.wiki/v1",
        "default_preference_weights": prefs,
        "hubs": hubs,
        "rides": rides,
        "name_aliases": NAME_ALIASES,
        "action_labels": action_labels,
        "model": {
            "id": "v2",
            "filename": "v2.onnx",
            "route_k": 5,
            "ride_feat_dim": 11,
            "supports_force_any_slot": True,
        },
    }
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
