# visualize.py
import pygame
import csv
import sys
import math
from oldSimulation.config import PARK_WIDTH, PARK_HEIGHT, ENTRANCE_COORDS, RIDES_CONFIG, TOTAL_MINUTES

# Setup Pygame with an expanded width for the Sidebar
SIDEBAR_WIDTH = 300
SCREEN_WIDTH = PARK_WIDTH + SIDEBAR_WIDTH
SCREEN_HEIGHT = PARK_HEIGHT

pygame.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Disneyland Queue Simulation & Guest Tracker")

# Fonts
SMALL_FONT = pygame.font.SysFont('Arial', 11, bold=False)
WAIT_FONT = pygame.font.SysFont('Arial', 13, bold=True)
LARGE_FONT = pygame.font.SysFont('Arial', 24, bold=True)
UI_FONT = pygame.font.SysFont('Arial', 16, bold=True)

TICKS_PER_SECOND = 10 
FPS = 60

def load_data():
    print("Loading simulation logs (this may take a few seconds)...")
    ride_states = {} 
    walks_at_tick = {t: list() for t in range(TOTAL_MINUTES)}
    
    # New Guest Tracking Data Structures
    guest_profiles = {}

    with open('rides_log.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row['tick'])
            if t not in ride_states: ride_states[t] = {}
            ride_states[t][row['ride_name']] = {
                'status': row['status'],
                'wait': float(row['wait_time']),
                'queue': int(row['queue_length'])
            }

    print("Pre-computing walking paths and guest itineraries...")
    with open('guest_log.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            g_id = row['guest_id']
            t = int(row['tick'])
            e_type = row['event']
            
            if g_id not in guest_profiles:
                guest_profiles[g_id] = {'events': list(), 'history': list(), 'total_rides': 0}
                
            # Store event for location tracking
            guest_profiles[g_id]['events'].append({
                'tick': t,
                'event': e_type,
                'x': float(row['start_x']) if row['start_x'] else 0,
                'y': float(row['start_y']) if row['start_y'] else 0,
                'ex': float(row['end_x']) if row['end_x'] else 0,
                'ey': float(row['end_y']) if row['end_y'] else 0,
                'arr_tick': int(row['arrival_tick']) if row['arrival_tick'] else 0,
                'target': row['target']
            })
            
            if e_type == 'END_RIDE':
                guest_profiles[g_id]['history'].append((t, row['target']))
                guest_profiles[g_id]['total_rides'] += 1

            # Populate general visual crowd walking dots
            if e_type == 'START_WALK':
                st = t
                et = int(row['arrival_tick'])
                w_data = (
                    float(row['start_x']), float(row['start_y']),
                    float(row['end_x']), float(row['end_y']),
                    st, et
                )
                for map_t in range(st, min(et, TOTAL_MINUTES)):
                    walks_at_tick[map_t].append(w_data)

    # Sort guests by total rides completed in the entire day (Highest to Lowest initially)
    sorted_guests = sorted(guest_profiles.keys(), key=lambda g: guest_profiles[g]['total_rides'], reverse=True)
                    
    print("Data loaded successfully!")
    return ride_states, walks_at_tick, guest_profiles, sorted_guests

def get_guest_state(g_id, current_tick_float, guest_profiles):
    events = guest_profiles[g_id]['events']
    if not events: return None
    
    # Find latest event
    current_event = events[0]
    for e in events:
        if e['tick'] <= current_tick_float:
            current_event = e
        else:
            break
            
    if current_event['event'] == 'LEFT_PARK':
        return {'status': 'Left Park', 'pos': None}
        
    if current_event['event'] == 'START_WALK':
        st = current_event['tick']
        et = current_event['arr_tick']
        duration = et - st
        if duration <= 0: 
            pos = (current_event['ex'], current_event['ey'])
        else:
            progress = max(0.0, min(1.0, (current_tick_float - st) / duration))
            cx = current_event['x'] + (current_event['ex'] - current_event['x']) * progress
            cy = current_event['y'] + (current_event['ey'] - current_event['y']) * progress
            pos = (cx, cy)
        return {'status': f"Walking to {current_event['target']}", 'pos': pos, 'dest': (current_event['ex'], current_event['ey'])}
        
    if current_event['event'] == 'ENTER_QUEUE':
        return {'status': f"In Line / Riding {current_event['target']}", 'pos': (current_event['x'], current_event['y'])}
        
    if current_event['event'] == 'SPAWN':
        return {'status': "Just Arrived", 'pos': (current_event['x'], current_event['y'])}
        
    return {'status': f"Finished {current_event['target']}", 'pos': (current_event['x'], current_event['y'])}

def draw_sidebar(SCREEN, tick_float, guest_profiles, sorted_guests, list_scroll_y, selected_guest, sort_descending):
    # Sidebar Background
    pygame.draw.rect(SCREEN, (35, 40, 45), (PARK_WIDTH, 0, SIDEBAR_WIDTH, PARK_HEIGHT))
    pygame.draw.line(SCREEN, (100, 100, 100), (PARK_WIDTH, 0), (PARK_WIDTH, PARK_HEIGHT), 2)
    
    # Title
    title = LARGE_FONT.render("GUEST DIRECTORY", True, (255, 255, 255))
    SCREEN.blit(title, (PARK_WIDTH + 20, 20))
    
    # --- SORT TOGGLE BUTTON ---
    sort_btn_rect = pygame.Rect(PARK_WIDTH + 20, 50, 150, 22)
    pygame.draw.rect(SCREEN, (50, 60, 70), sort_btn_rect, border_radius=4)
    sort_label = "↓ Highest to Lowest" if sort_descending else "↑ Lowest to Highest"
    lbl_surf = SMALL_FONT.render(sort_label, True, (230, 230, 230))
    SCREEN.blit(lbl_surf, (PARK_WIDTH + 30, 54))

    # --- SCROLLABLE LIST ---
    list_rect = pygame.Rect(PARK_WIDTH + 20, 80, SIDEBAR_WIDTH - 40, 300)
    pygame.draw.rect(SCREEN, (25, 30, 35), list_rect, border_radius=5)
    
    start_idx = int(list_scroll_y // 30)
    end_idx = min(len(sorted_guests), start_idx + (300 // 30) + 1)
    
    for i in range(start_idx, end_idx):
        g_id = sorted_guests[i]
        total = guest_profiles[g_id]['total_rides']
        y_pos = 80 + (i * 30) - list_scroll_y
        
        # Highlight if selected
        color = (60, 150, 255) if g_id == selected_guest else (200, 200, 200)
        
        if 80 <= y_pos <= 350:
            txt = WAIT_FONT.render(f"#{g_id}   ({total} Rides)", True, color)
            SCREEN.blit(txt, (PARK_WIDTH + 30, y_pos + 8))

    pygame.draw.rect(SCREEN, (100, 100, 100), list_rect, 2, border_radius=5) # Border

    # --- SELECTED GUEST TRACKER ---
    pygame.draw.line(SCREEN, (100, 100, 100), (PARK_WIDTH + 20, 400), (SCREEN_WIDTH - 20, 400), 2)
    
    if selected_guest:
        g_state = get_guest_state(selected_guest, tick_float, guest_profiles)
        total_day_rides = guest_profiles[selected_guest]['total_rides']
        
        # Calculate rides completed SO FAR
        history = guest_profiles[selected_guest]['history']
        rides_so_far =[r_name for r_tick, r_name in history if r_tick <= tick_float]
        
        t1 = LARGE_FONT.render(f"Tracking: #{selected_guest}", True, (255, 215, 0))
        SCREEN.blit(t1, (PARK_WIDTH + 20, 420))
        
        t2 = UI_FONT.render(f"Current Status:", True, (150, 150, 150))
        SCREEN.blit(t2, (PARK_WIDTH + 20, 460))
        
        status_color = (100, 255, 100) if g_state and g_state['pos'] else (255, 100, 100)
        status_text = g_state['status'] if g_state else "Unknown"
        
        # Wrap status text if too long
        if len(status_text) > 30: status_text = status_text[:27] + "..."
        t3 = UI_FONT.render(status_text, True, status_color)
        SCREEN.blit(t3, (PARK_WIDTH + 30, 485))

        t4 = UI_FONT.render(f"Itinerary So Far ({len(rides_so_far)}/{total_day_rides}):", True, (150, 150, 150))
        SCREEN.blit(t4, (PARK_WIDTH + 20, 530))
        
        # Show last 10 rides completed so far
        display_rides = rides_so_far[-10:]
        for idx, r_name in enumerate(display_rides):
            if len(r_name) > 30: r_name = r_name[:27] + "..."
            r_txt = WAIT_FONT.render(f"✓ {r_name}", True, (200, 200, 200))
            SCREEN.blit(r_txt, (PARK_WIDTH + 30, 560 + (idx * 22)))
            
    else:
        t1 = UI_FONT.render("Click a guest above to track", True, (150, 150, 150))
        SCREEN.blit(t1, (PARK_WIDTH + 20, 420))


def draw_park(tick_float, ride_states_at_tick, current_walks, playing, guest_profiles, sorted_guests, list_scroll_y, selected_guest, sort_descending):
    SCREEN.fill((230, 240, 245)) 
    
    # Draw Entrance
    pygame.draw.rect(SCREEN, (34, 139, 34), (ENTRANCE_COORDS[0]-40, ENTRANCE_COORDS[1]-20, 80, 40))
    ent_text = WAIT_FONT.render("ENTRANCE", True, (255, 255, 255))
    SCREEN.blit(ent_text, (ENTRANCE_COORDS[0] - ent_text.get_width()//2, ENTRANCE_COORDS[1] - 8))

    # Draw Current Time
    int_tick = int(tick_float)
    hour = 8 + (int_tick // 60)
    minute = int_tick % 60
    am_pm = "AM" if hour < 12 else "PM"
    if hour > 12: hour -= 12
    if hour == 0: hour = 12
    time_text = LARGE_FONT.render(f"Time: {hour:02d}:{minute:02d} {am_pm}", True, (0, 0, 0))
    SCREEN.blit(time_text, (20, 20))

    # Draw General Walking Guests
    for walk in current_walks:
        sx, sy, ex, ey, st, et = walk
        duration = et - st
        if duration <= 0: continue
        progress = max(0.0, min(1.0, (tick_float - st) / duration))
        curr_x = sx + (ex - sx) * progress
        curr_y = sy + (ey - sy) * progress
        pygame.draw.circle(SCREEN, (120, 120, 120), (int(curr_x), int(curr_y)), 2)

    # Draw Attractions
    for name, cfg in RIDES_CONFIG.items():
        x, y = cfg['coords']
        state = ride_states_at_tick.get(name, {'status': 'OPEN', 'wait': 0, 'queue': 0})
        is_broken = (state['status'] == 'BROKEN')
        
        radius = 16
        color = (200, 40, 40) if is_broken else (40, 120, 220)
        
        pygame.draw.circle(SCREEN, color, (x, y), radius)
        pygame.draw.circle(SCREEN, (0, 0, 0), (x, y), radius, 1) 
        
        if is_broken:
            wait_txt = WAIT_FONT.render("X", True, (255, 255, 255))
        else:
            wait_txt = WAIT_FONT.render(str(int(state['wait'])), True, (255, 255, 255))
        SCREEN.blit(wait_txt, (x - wait_txt.get_width()//2, y - wait_txt.get_height()//2))
        
        words = name.split()
        name_short = " ".join(words[:2]) + "..." if len(words) > 3 else name
        name_txt = SMALL_FONT.render(name_short, True, (50, 50, 50))
        SCREEN.blit(name_txt, (x - name_txt.get_width()//2, y + radius + 2))

    # ================= TRACKED GUEST OVERLAY =================
    if selected_guest:
        g_state = get_guest_state(selected_guest, tick_float, guest_profiles)
        if g_state and g_state['pos']:
            gx, gy = int(g_state['pos'][0]), int(g_state['pos'][1])
            
            # Draw line to destination if walking
            if 'dest' in g_state:
                dx, dy = int(g_state['dest'][0]), int(g_state['dest'][1])
                pygame.draw.line(SCREEN, (255, 215, 0), (gx, gy), (dx, dy), 2)
            
            # Draw Guest Marker (Pulsing Gold Star/Circle)
            pulse = abs(math.sin(pygame.time.get_ticks() / 200.0)) * 5
            pygame.draw.circle(SCREEN, (255, 215, 0), (gx, gy), 10 + pulse)
            pygame.draw.circle(SCREEN, (0, 0, 0), (gx, gy), 10 + pulse, 2)
            
            lbl = WAIT_FONT.render(f"#{selected_guest}", True, (0, 0, 0), (255, 215, 0))
            SCREEN.blit(lbl, (gx + 15, gy - 10))

    # ================= UI BOTTOM CONTROL BAR =================
    pygame.draw.rect(SCREEN, (40, 40, 45), (0, 930, PARK_WIDTH, 70))
    
    btn_color = (100, 200, 100) if playing else (200, 100, 100)
    pygame.draw.rect(SCREEN, btn_color, (30, 950, 80, 30), border_radius=5)
    btn_text = WAIT_FONT.render("PAUSE" if playing else "PLAY", True, (255, 255, 255))
    SCREEN.blit(btn_text, (30 + 40 - btn_text.get_width()//2, 950 + 15 - btn_text.get_height()//2))
    
    pygame.draw.rect(SCREEN, (100, 100, 100), (150, 960, 800, 10), border_radius=5)
    slider_progress = tick_float / (TOTAL_MINUTES - 1)
    pygame.draw.rect(SCREEN, (100, 150, 255), (150, 960, int(800 * slider_progress), 10), border_radius=5)
    pygame.draw.circle(SCREEN, (255, 255, 255), (150 + int(800 * slider_progress), 965), 10)

    # Draw Sidebar Layer on top
    draw_sidebar(SCREEN, tick_float, guest_profiles, sorted_guests, list_scroll_y, selected_guest, sort_descending)

    pygame.display.flip()

def run_visualizer():
    ride_states, walks_at_tick, guest_profiles, sorted_guests = load_data()
    clock = pygame.time.Clock()
    
    float_tick = 0.0 
    playing = True
    dragging = False
    
    # UI States
    list_scroll_y = 0
    selected_guest = None
    sort_descending = True

    play_btn_rect = pygame.Rect(30, 950, 80, 30)
    slider_rect = pygame.Rect(130, 940, 840, 50) 
    sidebar_list_rect = pygame.Rect(PARK_WIDTH + 20, 80, SIDEBAR_WIDTH - 40, 300)
    sort_btn_rect = pygame.Rect(PARK_WIDTH + 20, 50, 150, 22)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    if play_btn_rect.collidepoint(mouse_x, mouse_y):
                        playing = not playing
                    elif slider_rect.collidepoint(mouse_x, mouse_y):
                        dragging = True
                        playing = False
                    elif sort_btn_rect.collidepoint(mouse_x, mouse_y):
                        # Reverse sort order and reset scroll to top
                        sort_descending = not sort_descending
                        sorted_guests.reverse()
                        list_scroll_y = 0
                    elif sidebar_list_rect.collidepoint(mouse_x, mouse_y):
                        # Calculate which guest was clicked
                        click_y = mouse_y - 80 + list_scroll_y
                        clicked_idx = int(click_y // 30)
                        if 0 <= clicked_idx < len(sorted_guests):
                            selected_guest = sorted_guests[clicked_idx]
                            
                elif event.button == 4: # Scroll Up
                    if mouse_x > PARK_WIDTH:
                        list_scroll_y = max(0, list_scroll_y - 30)
                elif event.button == 5: # Scroll Down
                    if mouse_x > PARK_WIDTH:
                        max_scroll = max(0, (len(sorted_guests) * 30) - 300)
                        list_scroll_y = min(max_scroll, list_scroll_y + 30)
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

        if dragging:
            progress = max(0.0, min(1.0, (mouse_x - 150) / 800.0))
            float_tick = progress * (TOTAL_MINUTES - 1)
        elif playing:
            float_tick += dt * TICKS_PER_SECOND
            if float_tick >= TOTAL_MINUTES - 1:
                float_tick = TOTAL_MINUTES - 1
                playing = False

        int_tick = int(float_tick)
        current_rides = ride_states.get(int_tick, {})
        current_walks = walks_at_tick.get(int_tick, list())

        draw_park(float_tick, current_rides, current_walks, playing, guest_profiles, sorted_guests, list_scroll_y, selected_guest, sort_descending)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_visualizer()