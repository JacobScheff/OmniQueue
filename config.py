# config.py

# Park dimensions for visualization
PARK_WIDTH = 1000
PARK_HEIGHT = 1000
ENTRANCE_COORDS = (500, 900)

# Reduced total guests by 20% (originally 43000)
TOTAL_GUESTS = 34400
TOTAL_MINUTES = 900  # 8:00 AM to 11:00 PM (15 hours * 60 minutes)
WALKING_SPEED = 30   # coordinate units per minute

# Guests arrive on a bell curve, peaking around 11:00 AM (Tick 180)
# They stay for a bell curve amount of time, averaging 8 hours (480 mins)
GUEST_SPAWN_MEAN = 180
GUEST_SPAWN_STD_DEV = 120
GUEST_DURATION_MEAN = 480
GUEST_DURATION_STD_DEV = 120

# ALL DISNEYLAND RIDES CONFIGURATION
# Added "popularity" weight. Higher number = more guests actively want to go here.
RIDES_CONFIG = {
    # == Star Wars: Galaxy's Edge ==
    "Star Wars: Rise of the Resistance": {"capacity": 1200, "duration": 18, "breakdown_prob": 0.003, "popularity": 220, "coords": (100, 250)},
    "Millennium Falcon: Smugglers Run": {"capacity": 1800, "duration": 5, "breakdown_prob": 0.001, "popularity": 140, "coords": (200, 200)},

    # == Critter Country / Bayou ==
    "Tiana's Bayou Adventure": {"capacity": 1800, "duration": 10, "breakdown_prob": 0.0015, "popularity": 150, "coords": (100, 600)},
    "The Many Adventures of Winnie the Pooh": {"capacity": 1000, "duration": 3, "breakdown_prob": 0.0002, "popularity": 40, "coords": (60, 650)},
    "Davy Crockett's Explorer Canoes": {"capacity": 600, "duration": 10, "breakdown_prob": 0.0001, "popularity": 15, "coords": (120, 650)},

    # == New Orleans Square ==
    "Haunted Mansion": {"capacity": 2000, "duration": 9, "breakdown_prob": 0.0005, "popularity": 180, "coords": (150, 500)},
    "Pirates of the Caribbean": {"capacity": 2800, "duration": 15, "breakdown_prob": 0.0005, "popularity": 230, "coords": (200, 550)},

    # == Adventureland ==
    "Indiana Jones Adventure": {"capacity": 1600, "duration": 4, "breakdown_prob": 0.002, "popularity": 190, "coords": (250, 650)},
    "Jungle Cruise": {"capacity": 1800, "duration": 8, "breakdown_prob": 0.0005, "popularity": 130, "coords": (320, 620)},
    "Walt Disney's Enchanted Tiki Room": {"capacity": 1200, "duration": 15, "breakdown_prob": 0.0001, "popularity": 30, "coords": (380, 650)},

    # == Frontierland ==
    "Big Thunder Mountain Railroad": {"capacity": 2000, "duration": 3, "breakdown_prob": 0.001, "popularity": 160, "coords": (350, 450)},
    "Mark Twain Riverboat": {"capacity": 1500, "duration": 14, "breakdown_prob": 0.0001, "popularity": 40, "coords": (280, 500)},
    "Sailing Ship Columbia": {"capacity": 1200, "duration": 14, "breakdown_prob": 0.0001, "popularity": 35, "coords": (280, 450)},

    # == Fantasyland ==
    "Matterhorn Bobsleds": {"capacity": 1500, "duration": 3, "breakdown_prob": 0.0015, "popularity": 140, "coords": (650, 400)},
    "Peter Pan's Flight": {"capacity": 800, "duration": 3, "breakdown_prob": 0.0002, "popularity": 110, "coords": (480, 450)},
    "Mr. Toad's Wild Ride": {"capacity": 800, "duration": 2, "breakdown_prob": 0.0002, "popularity": 55, "coords": (520, 450)},
    "Snow White's Enchanted Wish": {"capacity": 800, "duration": 3, "breakdown_prob": 0.0002, "popularity": 60, "coords": (460, 470)},
    "Pinocchio's Daring Journey": {"capacity": 800, "duration": 3, "breakdown_prob": 0.0002, "popularity": 50, "coords": (440, 440)},
    "King Arthur Carrousel": {"capacity": 1000, "duration": 2, "breakdown_prob": 0.0001, "popularity": 40, "coords": (500, 400)},
    "Dumbo the Flying Elephant": {"capacity": 700, "duration": 2, "breakdown_prob": 0.0001, "popularity": 70, "coords": (550, 350)},
    "Mad Tea Party": {"capacity": 900, "duration": 2, "breakdown_prob": 0.0001, "popularity": 60, "coords": (600, 430)},
    "Alice in Wonderland": {"capacity": 800, "duration": 3, "breakdown_prob": 0.0002, "popularity": 75, "coords": (580, 380)},
    "Casey Jr. Circus Train": {"capacity": 700, "duration": 3, "breakdown_prob": 0.0002, "popularity": 45, "coords": (430, 350)},
    "Storybook Land Canal Boats": {"capacity": 700, "duration": 6, "breakdown_prob": 0.0002, "popularity": 45, "coords": (480, 330)},
    "It's a Small World": {"capacity": 2500, "duration": 14, "breakdown_prob": 0.0002, "popularity": 160, "coords": (500, 250)},

    # == Mickey's Toontown ==
    "Mickey & Minnie's Runaway Railway": {"capacity": 1800, "duration": 5, "breakdown_prob": 0.002, "popularity": 150, "coords": (500, 150)},
    "Roger Rabbit's Car Toon Spin": {"capacity": 1200, "duration": 4, "breakdown_prob": 0.0005, "popularity": 70, "coords": (430, 180)},
    "Chip 'n' Dale's GADGETcoaster": {"capacity": 800, "duration": 1, "breakdown_prob": 0.0005, "popularity": 45, "coords": (570, 180)},

    # == Tomorrowland ==
    "Space Mountain": {"capacity": 1800, "duration": 3, "breakdown_prob": 0.0015, "popularity": 190, "coords": (850, 550)},
    "Star Tours": {"capacity": 1800, "duration": 5, "breakdown_prob": 0.001, "popularity": 110, "coords": (750, 580)},
    "Buzz Lightyear Astro Blasters": {"capacity": 2000, "duration": 5, "breakdown_prob": 0.0005, "popularity": 130, "coords": (700, 630)},
    "Astro Orbitor": {"capacity": 600, "duration": 2, "breakdown_prob": 0.0002, "popularity": 35, "coords": (750, 650)},
    "Autopia": {"capacity": 1800, "duration": 5, "breakdown_prob": 0.0005, "popularity": 90, "coords": (850, 450)},
    "Finding Nemo Submarine Voyage": {"capacity": 1000, "duration": 13, "breakdown_prob": 0.0005, "popularity": 70, "coords": (780, 480)},
    "Disneyland Monorail": {"capacity": 1000, "duration": 15, "breakdown_prob": 0.0005, "popularity": 30, "coords": (800, 500)},

    # == Main Street, U.S.A. ==
    "Disneyland Railroad": {"capacity": 1500, "duration": 20, "breakdown_prob": 0.0002, "popularity": 70, "coords": (500, 850)},
}