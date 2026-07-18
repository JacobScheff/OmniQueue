"""Map ThemeParks.wiki live entities onto Park.config.RIDES indices."""

from __future__ import annotations

import re
import unicodedata

from Park import config

# Disneyland Park (Anaheim) on ThemeParks.wiki
DISNEYLAND_PARK_ENTITY_ID = "7340550b-c14d-4def-80bb-acdb51d49a66"

# Stable entity ids when names drift (™, punctuation, subtitle changes).
KNOWN_ENTITY_IDS: dict[int, str] = {
    0: "34b1d70f-11c4-42df-935e-d5582c9f1a8e",  # Rise of the Resistance
    1: "b2c2549c-e9da-4fdd-98ea-1dcff596fed7",  # Smugglers Run
    2: "a9076acd-7630-4bad-a8da-e6bd689ddcac",  # Tiana's
    3: "52a8ef64-d54c-4974-883f-027c3026e3f1",  # Pooh
    4: "5bd95ae8-181d-449c-8f04-a621e2448961",  # Canoes
    5: "ff52cb64-c1d5-4feb-9d43-5dbd429bac81",  # Haunted Mansion
    6: "82aeb29b-504a-416f-b13f-f41fa5b766aa",  # Pirates
    7: "2aedc657-1ee2-4545-a1ce-14753f28cc66",  # Indiana Jones Adventure
    8: "1b83fda8-d60e-48e4-9a3d-90ddcbcd1001",  # Jungle Cruise
    9: "106c1e5a-a5e7-42d7-96ab-bc100d8faf71",  # Tiki Room
    10: "0de1413a-73ee-46cf-af2e-c491cc7c7d3b",  # Big Thunder
    11: "6c30d5b0-8c0a-406f-9258-0b6c55d4a5e4",  # Mark Twain
    12: "c9e39189-7e99-4e0a-97e0-4a0d5654d257",  # Columbia
    13: "faaa8be9-cc1e-4535-ac20-04a535654bd0",  # Matterhorn
    14: "c23af6ba-8515-406a-8a48-d0818ba0bfc9",  # Peter Pan
    15: "9d401ad3-49b2-469f-ac73-93eb429428fb",  # Mr Toad
    16: "4f0053e7-b8db-4833-b02f-35e1c91b4523",  # Snow White
    17: "90ee50d4-7cc9-4824-b29d-2aac801acc29",  # Pinocchio
    18: "f7904912-3f08-4563-b99e-fd59f43cc9f2",  # Carrousel
    19: "cc980e8e-192f-48b6-848c-27784084e54b",  # Dumbo
    20: "e0cfed11-96d7-40f3-907f-5cfed172592a",  # Mad Tea
    21: "a07f3110-013e-43bb-a182-e66bb8b5e28d",  # Alice
    22: "8e686e4c-f3db-4d9c-a185-2d54b1fa8899",  # Casey Jr
    23: "cb929138-d77a-4dd2-983c-f651bbd1bd92",  # Storybook
    24: "3638ac09-9fce-4a43-8c79-8ebbe17afce2",  # small world
    25: "cd670bff-81d1-4f34-8676-7bafdf49220a",  # Mickey & Minnie's Runaway Railway
    26: "6ce9cdd1-0a43-459e-83cd-f4cace9cfa7b",  # Roger Rabbit
    27: "59647168-d239-4161-8b24-92eb128e96fb",  # GADGETcoaster
    28: "9167db1d-e5e7-46da-a07f-ae30a87bc4c4",  # Space Mountain
    29: "cc718d11-fa15-44ee-87d0-ded989ad61bc",  # Star Tours
    30: "88197808-3c56-4198-a5a4-6066541251cf",  # Buzz
    31: "6c225598-91c9-44a3-95e2-7c423475db61",  # Astro Orbitor
    32: "1da85181-bf0f-4ccc-b98e-243142f7347b",  # Autopia
    33: "64d44aaa-6857-4693-b24b-bcff6c6dcfa1",  # Nemo
}

# Extra API name variants → config ride index
NAME_ALIASES: dict[str, int] = {
    "indiana jones adventure": 7,
    "indiana jones™ adventure": 7,
    "mickey & minnie's runaway railway": 25,
    "mickey and minnie's runaway railway": 25,
    "star tours - the adventures continue": 29,
    "star tours–the adventures continue": 29,
    '"it\'s a small world"': 24,
    "it's a small world": 24,
}


def _normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    text = text.replace("™", "").replace("®", "")
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_config_name_index() -> dict[str, int]:
    index: dict[str, int] = {}
    for i, ride in enumerate(config.RIDES):
        index[_normalize_name(ride["name"])] = i
    for alias, idx in NAME_ALIASES.items():
        index[_normalize_name(alias)] = idx
    return index


def resolve_ride_index(entity_id: str | None, name: str, name_index: dict[str, int]) -> int | None:
    """Return config ride index for a live attraction, or None if not in our catalog."""
    if entity_id:
        for ride_id, known in KNOWN_ENTITY_IDS.items():
            if known == entity_id:
                return ride_id
    key = _normalize_name(name)
    if key in name_index:
        return name_index[key]
    # Prefix / containment fallback for subtitle drift
    for cand, idx in name_index.items():
        if cand in key or key in cand:
            return idx
    return None


def hub_display_name(hub_id: int) -> str:
    return {
        config.NODE_ENTRANCE: "Entrance",
        config.NODE_MAIN_HUB: "Main Street Hub",
        config.NODE_GALAXY_HUB: "Galaxy's Edge",
        config.NODE_CRITTER_HUB: "Critter Country / Bayou",
        config.NODE_NEW_ORLEANS_HUB: "New Orleans Square",
        config.NODE_ADVENTURE_HUB: "Adventureland",
        config.NODE_FRONTIER_HUB: "Frontierland",
        config.NODE_FANTASY_HUB: "Fantasyland",
        config.NODE_TOONTOWN_HUB: "Toontown",
        config.NODE_TOMORROW_HUB: "Tomorrowland",
        config.NODE_RIVER_CROSSING: "River Crossing",
        config.NODE_CENTRAL_PLAZA: "Central Plaza",
    }.get(hub_id, f"Hub {hub_id}")
