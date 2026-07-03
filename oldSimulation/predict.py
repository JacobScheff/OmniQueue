# predict.py
import math
import random
import csv
import collections
import time
import datetime
from oldSimulation.config import *
from oldSimulation.guest_logic import choose_next_ride
from oldSimulation.getWaitTimes import getRideStatuses

class Ride:
    def __init__(self, name, config):
        self.name = name
        self.capacity_per_min = config['capacity'] / 60.0
        self.duration = config['duration']
        self.breakdown_prob = config['breakdown_prob']
        self.coords = config['coords']
        
        self.queue = collections.deque()
        self.status = 'OPEN'
        self.broken_until = 0
        self.capacity_fraction = 0.0 

    def get_wait_time(self):
        if self.capacity_per_min == 0: return 999
        return len(self.queue) / self.capacity_per_min

class Guest:
    def __init__(self, g_id, force_spawn_tick=None):
        self.id = g_id
        # If we specify a spawn tick (e.g., for live guests currently in the park), use it
        if force_spawn_tick is not None:
            self.spawn_tick = force_spawn_tick
        else:
            self.spawn_tick = max(0, min(TOTAL_MINUTES - 120, int(random.gauss(GUEST_SPAWN_MEAN, GUEST_SPAWN_STD_DEV))))
            
        self.duration_in_park = max(120, int(random.gauss(GUEST_DURATION_MEAN, GUEST_DURATION_STD_DEV)))
        self.preferences = {} 
        self.location = ENTRANCE_COORDS

def calc_walk_time(loc1, loc2):
    dist = math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])
    return max(1, math.ceil(dist / WALKING_SPEED))

def get_current_tick():
    """Calculates the current simulation tick based on your local computer time."""
    now = datetime.datetime.now()
    start_of_day = now.replace(hour=8, minute=0, second=0, microsecond=0)
    delta = now - start_of_day
    tick = int(delta.total_seconds() / 60)
    
    if tick < 0: return 0
    if tick >= TOTAL_MINUTES: return TOTAL_MINUTES - 1
    return tick

def get_average_walkers(target_tick, filename='walking_guests_stats.csv'):
    closest_tick = 0
    avg_walkers = 0
    min_diff = 9999
    
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            avg_idx = header.index('Average')
            tick_idx = header.index('Tick')
            
            for row in reader:
                row_tick = int(row[tick_idx])
                diff = abs(row_tick - target_tick)
                if diff < min_diff:
                    min_diff = diff
                    closest_tick = row_tick
                    avg_walkers = float(row[avg_idx])
                    
        return int(avg_walkers)
    except FileNotFoundError:
        print("Warning: 'walking_guests_stats.csv' not found. Defaulting to 3000 walkers.")
        return 3000

def print_progress(iteration, total):
    if total <= 0: return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(50 * iteration // total)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    print(f'\rSimulating Future |{bar}| {percent}% Complete', end='\r')
    if iteration == total: print()

def run_prediction():
    print("Fetching LIVE wait times from Disneyland API...")
    live_statuses = getRideStatuses()
    if not live_statuses:
        print("Failed to fetch live data. Ensure getWaitTimes.py is working.")
        return

    current_tick = get_current_tick()
    num_walkers = get_average_walkers(current_tick)
    
    hour = 8 + (current_tick // 60)
    minute = current_tick % 60
    am_pm = "AM" if hour < 12 else "PM"
    hr_12 = hour if hour <= 12 else hour - 12
    if hr_12 == 0: hr_12 = 12
    print(f"Current Time: {hr_12:02d}:{minute:02d} {am_pm} (Tick {current_tick})")
    print(f"Historically, ~{num_walkers} guests are actively walking right now.")
    print("Initializing state and predicting future flow...")

    # --- 1. INITIALIZE PARK ---
    rides = {name: Ride(name, cfg) for name, cfg in RIDES_CONFIG.items()}
    guest_id_counter = 0
    
    spawn_schedule = collections.defaultdict(list)
    walking_arrivals = collections.defaultdict(list)
    ride_finishers = collections.defaultdict(list)
    deciding_guests =[]

    # --- 2. POPULATE FUTURE ARRIVALS ---
    for _ in range(TOTAL_GUESTS):
        temp_spawn = max(0, min(TOTAL_MINUTES - 120, int(random.gauss(GUEST_SPAWN_MEAN, GUEST_SPAWN_STD_DEV))))
        if temp_spawn >= current_tick:
            g = Guest(guest_id_counter, force_spawn_tick=temp_spawn)
            spawn_schedule[g.spawn_tick].append(g)
            guest_id_counter += 1

    # --- 3. POPULATE LIVE RIDE QUEUES ---
    for name, ride in rides.items():
        if name in live_statuses:
            is_open, wait_time = live_statuses[name]
            ride.status = 'OPEN' if is_open else 'BROKEN'
            
            if not is_open:
                # If it is broken in real life right now, we estimate it reopens in 15-60 mins
                # (Otherwise it would stay broken forever since random future breakdowns are disabled)
                ride.broken_until = current_tick + random.randint(15, 60)
            else:
                # Convert minutes of wait time into actual number of Guest objects standing in line
                queue_length = int(wait_time * ride.capacity_per_min)
                for _ in range(queue_length):
                    past_spawn = max(0, current_tick - random.randint(30, 240))
                    g = Guest(guest_id_counter, force_spawn_tick=past_spawn)
                    ride.queue.append(g)
                    guest_id_counter += 1

    # --- 4. POPULATE CURRENTLY WALKING GUESTS ---
    rides_state = {
        name: {'wait_time': r.get_wait_time(), 'status': r.status, 'queue_length': len(r.queue), 'coords': r.coords}
        for name, r in rides.items()
    }
    
    ride_names = list(rides.keys())
    with open('guest_log.csv', 'w', newline='') as g_file, \
         open('rides_log.csv', 'w', newline='') as r_file:
         
        g_writer = csv.writer(g_file)
        r_writer = csv.writer(r_file)
        
        g_writer.writerow(['tick', 'guest_id', 'event', 'start_x', 'start_y', 'end_x', 'end_y', 'arrival_tick', 'target'])
        r_writer.writerow(['tick', 'ride_name', 'status', 'queue_length', 'wait_time'])

        # Backfill empty log data from 8:00 AM to 'now' so the visualizer doesn't error out
        for t in range(0, current_tick):
            for name, cfg in RIDES_CONFIG.items():
                r_writer.writerow([t, name, 'OPEN', 0, 0])

        for _ in range(num_walkers):
            start_ride = random.choice(ride_names)
            sx, sy = rides[start_ride].coords
            
            past_spawn = max(0, current_tick - random.randint(30, 240))
            g = Guest(guest_id_counter, force_spawn_tick=past_spawn)
            g.location = (sx, sy)
            guest_id_counter += 1
            
            target_name = choose_next_ride(g.id, g.preferences, rides_state, g.location)
            target = rides.get(target_name) if target_name else None
            
            target_coords = target.coords if target else ENTRANCE_COORDS
            walk_time = calc_walk_time(g.location, target_coords)
            arrival_tick = current_tick + walk_time
            
            walking_arrivals[arrival_tick].append((g, target))
            g_writer.writerow([current_tick, g.id, 'START_WALK', sx, sy, target_coords[0], target_coords[1], arrival_tick, target.name if target else 'ENTRANCE'])

        # --- 5. RUN PREDICTION ENGINE (Normal Sim Loop) ---
        start_time = time.time()
        sim_ticks = TOTAL_MINUTES - current_tick
        
        for tick in range(current_tick, TOTAL_MINUTES):
            # Spawn new
            for g in spawn_schedule[tick]:
                deciding_guests.append(g)
                g_writer.writerow([tick, g.id, 'SPAWN', g.location[0], g.location[1], '', '', '', 'ENTRANCE'])

            # Finish rides
            for g, ride in ride_finishers[tick]:
                g.location = ride.coords
                deciding_guests.append(g)
                g_writer.writerow([tick, g.id, 'END_RIDE', g.location[0], g.location[1], '', '', '', ride.name])

            # Finish walks
            for g, target_ride in walking_arrivals[tick]:
                if target_ride is None:
                    g_writer.writerow([tick, g.id, 'LEFT_PARK', g.location[0], g.location[1], '', '', '', ''])
                else:
                    g.location = target_ride.coords
                    if target_ride.status == 'BROKEN':
                        deciding_guests.append(g)
                    else:
                        target_ride.queue.append(g)
                        g_writer.writerow([tick, g.id, 'ENTER_QUEUE', g.location[0], g.location[1], '', '', '', target_ride.name])

            # Process Rides (FUTURE BREAKDOWNS REMOVED)
            rides_state = {}
            for name, ride in rides.items():
                if ride.status == 'BROKEN':
                    if tick >= ride.broken_until:
                        ride.status = 'OPEN'
                
                # Note: The random breakdown generator logic that used to be here is completely removed.
                # Rides will remain open unless they were already broken in the live API fetch.
                
                if ride.status == 'OPEN':
                    ride.capacity_fraction += ride.capacity_per_min
                    process_count = int(ride.capacity_fraction)
                    ride.capacity_fraction -= process_count
                    
                    while ride.queue and process_count > 0:
                        queued_g = ride.queue.popleft()
                        finish_tick = tick + ride.duration
                        ride_finishers[finish_tick].append((queued_g, ride))
                        process_count -= 1
                
                rides_state[name] = {
                    'wait_time': ride.get_wait_time(),
                    'status': ride.status,
                    'queue_length': len(ride.queue),
                    'coords': ride.coords
                }
                r_writer.writerow([tick, name, ride.status, len(ride.queue), round(ride.get_wait_time(), 1)])

            # Decide next actions
            for g in deciding_guests:
                if tick - g.spawn_tick > g.duration_in_park:
                    target = None
                else:
                    target_name = choose_next_ride(g.id, g.preferences, rides_state, g.location)
                    target = rides.get(target_name) if target_name else None
                
                target_coords = target.coords if target else ENTRANCE_COORDS
                walk_time = calc_walk_time(g.location, target_coords)
                arrival_tick = tick + walk_time
                
                walking_arrivals[arrival_tick].append((g, target))
                g_writer.writerow([tick, g.id, 'START_WALK', g.location[0], g.location[1], target_coords[0], target_coords[1], arrival_tick, target.name if target else 'ENTRANCE'])

            deciding_guests.clear()
            
            # Update Progress Bar
            ticks_processed = tick - current_tick
            if ticks_processed % 10 == 0:
                print_progress(ticks_processed, sim_ticks)
                
        print_progress(sim_ticks, sim_ticks)
        print(f"Prediction completed in {round(time.time() - start_time, 2)} seconds. Logs saved.")

if __name__ == "__main__":
    run_prediction()