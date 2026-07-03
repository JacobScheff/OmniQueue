# simulation.py
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
    # BUG FIX: Force at least 1 minute of transition time to prevent event loop dropout
    return max(1, math.ceil(dist / WALKING_SPEED))

def print_progress(iteration, total):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(50 * iteration // total)
    bar = '█' * filled_length + '-' * (50 - filled_length)
    print(f'\rSimulating |{bar}| {percent}% Complete', end='\r')
    if iteration == total: print()

def run_simulation():
    print("Initializing Park and Guests...")
    rides = {name: Ride(name, cfg) for name, cfg in RIDES_CONFIG.items()}
    guests =[Guest(i) for i in range(TOTAL_GUESTS)]
    
    spawn_schedule = collections.defaultdict(list)
    walking_arrivals = collections.defaultdict(list)
    ride_finishers = collections.defaultdict(list)
    
    for g in guests:
        spawn_schedule[g.spawn_tick].append(g)

    deciding_guests =[]
    
    with open('guest_log.csv', 'w', newline='') as g_file, \
         open('rides_log.csv', 'w', newline='') as r_file:
         
        g_writer = csv.writer(g_file)
        r_writer = csv.writer(r_file)
        
        g_writer.writerow(['tick', 'guest_id', 'event', 'start_x', 'start_y', 'end_x', 'end_y', 'arrival_tick', 'target'])
        r_writer.writerow(['tick', 'ride_name', 'status', 'queue_length', 'wait_time'])

        print("Running Simulation Loop...")
        start_time = time.time()
        
        for tick in range(TOTAL_MINUTES):
            # 1. Spawn newly arriving guests
            for g in spawn_schedule[tick]:
                deciding_guests.append(g)
                g_writer.writerow([tick, g.id, 'SPAWN', g.location[0], g.location[1], '', '', '', 'ENTRANCE'])

            # 2. Finish rides
            for g, ride in ride_finishers[tick]:
                g.location = ride.coords
                deciding_guests.append(g)
                g_writer.writerow([tick, g.id, 'END_RIDE', g.location[0], g.location[1], '', '', '', ride.name])

            # 3. Finish walks
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

            # 4. Process Rides
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
                r_writer.writerow([tick, name, ride.status, len(ride.queue), round(ride.get_wait_time(), 1)])

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
                g_writer.writerow([tick, g.id, 'START_WALK', g.location[0], g.location[1], target_coords[0], target_coords[1], arrival_tick, target.name if target else 'ENTRANCE'])

            deciding_guests.clear()
            
            if tick % 10 == 0:
                print_progress(tick, TOTAL_MINUTES)
                
        print_progress(TOTAL_MINUTES, TOTAL_MINUTES)
        print(f"Simulation completed in {round(time.time() - start_time, 2)} seconds. Logs saved.")

if __name__ == "__main__":
    run_simulation()