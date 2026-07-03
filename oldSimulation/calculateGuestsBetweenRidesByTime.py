# calculateGuestsBetweenRidesByTime.py
import math
import random
import csv
import collections
import time
from oldSimulation.config import *
from oldSimulation.guest_logic import choose_next_ride

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
    def __init__(self, g_id):
        self.id = g_id
        self.spawn_tick = max(0, min(TOTAL_MINUTES - 120, int(random.gauss(GUEST_SPAWN_MEAN, GUEST_SPAWN_STD_DEV))))
        self.duration_in_park = max(120, int(random.gauss(GUEST_DURATION_MEAN, GUEST_DURATION_STD_DEV)))
        self.preferences = {} 
        self.location = ENTRANCE_COORDS

def calc_walk_time(loc1, loc2):
    dist = math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])
    # Ensure walks always take at least 1 minute
    return max(1, math.ceil(dist / WALKING_SPEED))

def run_analytical_simulations(num_runs, interval_mins):
    print(f"Starting {num_runs} randomized simulations...")
    print(f"Tracking walking guests every {interval_mins} minutes.\n")
    
    # Generate the checkpoints (e.g. 0, 10, 20, 30... 900)
    intervals = list(range(0, TOTAL_MINUTES, interval_mins))
    
    # Dictionary to hold lists of counts: { tick:[run1_count, run2_count, ...] }
    results = {t:[] for t in intervals}
    overall_start_time = time.time()
    
    for run_idx in range(num_runs):
        print(f"--- Running Simulation {run_idx + 1}/{num_runs} ---")
        
        # Park state initialization
        rides = {name: Ride(name, cfg) for name, cfg in RIDES_CONFIG.items()}
        guests = [Guest(i) for i in range(TOTAL_GUESTS)]
        
        spawn_schedule = collections.defaultdict(list)
        walking_arrivals = collections.defaultdict(list)
        ride_finishers = collections.defaultdict(list)
        
        for g in guests:
            spawn_schedule[g.spawn_tick].append(g)

        deciding_guests =[]
        currently_walking = 0  # ACTIVE TRACKER
        
        # Sim Loop
        for tick in range(TOTAL_MINUTES):
            # 1. Spawn newly arriving guests
            for g in spawn_schedule[tick]:
                deciding_guests.append(g)

            # 2. Finish rides
            for g, ride in ride_finishers[tick]:
                g.location = ride.coords
                deciding_guests.append(g)

            # 3. Finish walks
            for g, target_ride in walking_arrivals[tick]:
                currently_walking -= 1 # Guest has arrived, no longer walking
                if target_ride is not None:
                    g.location = target_ride.coords
                    if target_ride.status == 'BROKEN':
                        deciding_guests.append(g)
                    else:
                        target_ride.queue.append(g)

            # 4. Process Rides (Throughput & Breakdowns)
            rides_state = {}
            for name, ride in rides.items():
                if ride.status == 'BROKEN':
                    if tick >= ride.broken_until:
                        ride.status = 'OPEN'
                else:
                    if random.random() < ride.breakdown_prob:
                        ride.status = 'BROKEN'
                        ride.broken_until = tick + random.randint(15, 60)
                
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

            # 5. Decide next actions
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
                currently_walking += 1 # Guest begins walking
                
            deciding_guests.clear()
            
            # 6. Capture Data Snapshot
            if tick in results:
                results[tick].append(currently_walking)
                
    print(f"\nAll {num_runs} simulations completed in {round(time.time() - overall_start_time, 2)} seconds.")
    
    # ================= EXPORT DATA =================
    output_filename = 'walking_guests_stats.csv'
    print(f"Calculating averages and exporting data to '{output_filename}'...")
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Generate Header
        header =['Time', 'Tick'] + [f'Run_{i+1}' for i in range(num_runs)] + ['Average']
        writer.writerow(header)
        
        # Write Rows
        for tick in intervals:
            # Format time beautifully (8:00 AM)
            hour = 8 + (tick // 60)
            minute = tick % 60
            am_pm = "AM" if hour < 12 else "PM"
            hr_12 = hour if hour <= 12 else hour - 12
            if hr_12 == 0: hr_12 = 12
            time_str = f"{hr_12:02d}:{minute:02d} {am_pm}"
            
            run_data = results[tick]
            average = sum(run_data) / len(run_data) if run_data else 0
            
            # Append Row: Time, Tick, Run 1, Run 2... Run 20, Average
            row_data = [time_str, tick] + run_data + [round(average, 2)]
            writer.writerow(row_data)
            
    print("Export Complete! You can now open it in Excel/Sheets.")

if __name__ == "__main__":
    run_analytical_simulations(num_runs=200, interval_mins=10) # For 34400 guests