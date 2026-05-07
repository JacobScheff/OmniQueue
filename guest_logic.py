# guest_logic.py
import random
import math
from config import RIDES_CONFIG

def choose_next_ride(guest_id, guest_preferences, rides_state, current_location):
    """
    Determines where a guest goes next based on popularity, wait times, AND distance.
    """
    
    # 1. Filter out broken rides, AND filter out the ride the guest is currently at
    available_rides =[
        name for name, state in rides_state.items() 
        if state['status'] != 'BROKEN' and state['coords'] != current_location
    ]
    
    # If no rides are available, wait/leave
    if not available_rides:
        return None
        
    # 2. Calculate dynamic weights for each available ride
    weights =[]
    for name in available_rides:
        base_popularity = RIDES_CONFIG[name]['popularity']
        current_wait = rides_state[name]['wait_time']
        coords = rides_state[name]['coords']
        
        # Penalty 1: High wait times deter guests
        wait_penalty = 1 + (current_wait / 30.0)
        
        # Penalty 2: Walking distance deters guests
        dist = math.hypot(current_location[0] - coords[0], current_location[1] - coords[1])
        dist_penalty = 1 + (dist / 400.0) 
        
        # Final combined weight for this ride
        dynamic_weight = base_popularity / (wait_penalty * dist_penalty)
        weights.append(dynamic_weight)

    # 3. Perform a weighted random selection
    chosen_ride = random.choices(available_rides, weights=weights, k=1)[0]
    
    return chosen_ride