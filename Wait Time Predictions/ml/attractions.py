"""Attractions omitted from modeling and display; raw archives stay intact."""

import unicodedata


# IDs also cover historical renames and both listings of A Magical Life.
EXCLUDED_ATTRACTIONS = {
    "e27b1db8-9ec9-4f8b-9ca6-fd6377de66ee": "Adventureland Treehouse",
    "5bd95ae8-181d-449c-8f04-a621e2448961": "Davy Crockett's Explorer Canoes",
    "56d0bd6d-5106-4420-8f60-0005475c04c3": "Disneyland Monorail",
    "ca32ef8a-ae0d-4c24-bf4f-e59192122e01": "Donald's Duck Pond",
    "835deb74-de54-4d7f-9dc4-dacab90fcb60": "Frontierland Shootin' Exposition",
    "b0eca3d3-a519-47f9-a5a7-9911126da2df": "Goofy's How-to-Play Yard",
    "1b23667a-d8fb-436d-8952-c3e3f2e56d13": "Great Moments with Mr. Lincoln",
    "f5bb0d14-7eee-4ede-9230-eb256ce3664c": "Main Street Cinema",
    "bcfd1e17-3eab-4203-b597-6257a257d427": "Main Street Vehicles",
    "c02fb82d-0860-4e95-8c61-899fa594d20e": "Minnie's House",
    "6e2af074-3a67-4f38-9bcf-2f5a572c87df": "Opera House - Walt Disney - A Magical Life",
    "05b17c11-fd4c-4b85-ab45-3e269dc56559": "Walt Disney - A Magical Life",
    "526ada11-4688-442b-96d4-4077eac4f988": "Pixar Short Film Spotlight",
    "c9e39189-7e99-4e0a-97e0-4a0d5654d257": "Sailing Ship Columbia",
    "90d5a091-478c-4df1-adfe-c605b4005013": "Sleeping Beauty Castle Walkthrough",
    "37858b2a-9c11-4ac6-9f8d-db7fa9088703": "The Disney Gallery",
    "106c1e5a-a5e7-42d7-96ab-bc100d8faf71": "Walt Disney's Enchanted Tiki Room",
}


def normalized_name(name):
    return "".join(char for char in unicodedata.normalize("NFKD", name).casefold() if char.isalnum())


# Name matching covers alternate source IDs, punctuation, and shortened titles.
EXCLUDED_NAME_PARTS = tuple(map(normalized_name, (
    "Treehouse", "Canoes", "Monorail", "Donald Duck Pond", "Donald's Duck Pond",
    "Frontierland Shootin", "Goofy Yard", "Goofy's How-to-Play Yard",
    "Great Moments with Lincoln", "Great Moments with Mr. Lincoln",
    "Main Street Cinema", "Main Street Vehicles", "Minnie House", "Minnie's House",
    "Walt Disney Magic Life", "Walt Disney A Magical Life", "Opera House",
    "Pixar Short Film", "Sailing Ship Columbia", "Sleeping Beauty Walkthrough",
    "Sleeping Beauty Castle Walkthrough", "Disney Gallery", "Tiki Room",
    # Walkthroughs, play areas, transport and untimed presentations in the six parks.
    # Shows with a real posted standby queue (e.g. Philharmagic) remain eligible.
    "Redwood Creek Challenge", "Animation Academy", "Sorcerer's Workshop",
    "Walt Disney Imagineering Blue Sky", "Disney Junior Dance",
    "Country Bear", "Hall of Presidents", "Carousel of Progress",
    "Swiss Family Treehouse", "Tom Sawyer Island", "Frontierland Shootin",
    "Liberty Square Riverboat", "Walt Disney World Railroad",
    "Gorilla Falls", "Maharajah Jungle Trek", "Discovery Island Trails",
    "Oasis Exhibits", "Rafiki", "Affection Section", "Wilderness Explorers",
    "Conservation Station", "Wildlife Express Train", "Boneyard",
    "Indiana Jones Epic Stunt", "Beauty and the Beast Live on Stage",
    "Fantasmic", "Walt Disney Presents", "Vacation Fun", "Lightning McQueen Racing Academy",
    "Awesome Planet", "Canada Far and Wide", "Impressions de France",
    "American Adventure", "Reflections of China", "Disney and Pixar Short Film",
    "Journey of Water", "ImageWorks", "House of the Whispering Willows",
)))


def is_excluded(ride):
    name = normalized_name(ride.get("name", ""))
    return ride["id"] in EXCLUDED_ATTRACTIONS or any(part in name for part in EXCLUDED_NAME_PARTS)


def modeling_day(day, eligible_ids=None):
    """Project ride metadata and token columns together without changing the cache."""
    indices = [index for index, ride in enumerate(day["rides"])
               if not is_excluded(ride) and (eligible_ids is None or ride["id"] in eligible_ids)]
    if len(indices) == len(day["rides"]):
        return day
    return {**day, "rides": [day["rides"][index] for index in indices],
            "tokens": [[row[index] for index in indices] for row in day["tokens"]]}


def validate_checkpoint_rides(rides):
    """Old model dimensions must be retrained rather than silently retaining rides."""
    if any(is_excluded(ride) for ride in rides):
        raise ValueError("This checkpoint includes excluded attractions. "
                         "Retrain with python ml/train.py before forecasting or testing.")
