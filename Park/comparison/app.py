"""Synchronized Pygame route replay with scores and decision inspection."""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
import math

from Park import config
from Park.comparison.history import ZONE
from Park.planner.data import WalkingMap

BG = (14, 20, 30)
PANEL = (23, 33, 46)
TEXT = (227, 236, 245)
MUTED = (146, 165, 185)
COLORS = [(72, 186, 255), (255, 184, 86)]


def point_on_path(points, fraction):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    remaining = sum(lengths)*max(0, min(1, fraction))
    for a, b, length in zip(points, points[1:], lengths):
        if remaining <= length and length:
            f = remaining/length
            return [a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])]
        remaining -= length
    return points[-1]


def completed_at(run, moment):
    return [e for e in run["events"] if e["kind"] == "complete" and e["end"] <= moment]


def run_replay(payload, *, speed=300, max_frames=None, screenshot=None):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((1440, 1000), pygame.RESIZABLE)
    pygame.display.set_caption("OmniQueue | RL vs DP | historical replay")
    font = pygame.font.SysFont("segoeui", 18)
    small = pygame.font.SysFont("segoeui", 15)
    title = pygame.font.SysFont("segoeui", 26, bold=True)
    large = pygame.font.SysFont("segoeui", 32, bold=True)
    walks = WalkingMap()
    opening, closing = payload["opening"], payload["closing"]
    # Start at final results; Home begins the full-day animation.
    moment, paused, running, frames = float(closing), True, True, 0
    clock = pygame.time.Clock()
    backgrounds = {}
    itinerary_offsets = [0, 0]

    def text(value, x, y, color=TEXT, face=font):
        screen.blit(face.render(str(value), True, color), (x, y))

    def clipped(value, width, face=font):
        while value and face.size(value)[0] > width:
            value = value[:-2]+"…"
        return value

    def clock_text(t):
        return datetime.fromtimestamp(t, ZONE).strftime("%H:%M:%S")

    try:
        while running:
            dt = min(clock.tick(30)/1000, 0.2)
            width, height = screen.get_size()
            timeline = pygame.Rect(24, height-66, width-48, 16)
            panel_w = (width-72)//2
            map_size = min(panel_w-32, height-530)
            map_size = max(220, map_size)
            map_top = 212
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((max(1000, ev.w), max(820, ev.h)), pygame.RESIZABLE)
                    backgrounds.clear()
                elif ev.type == pygame.MOUSEWHEEL:
                    column = 0 if pygame.mouse.get_pos()[0] < width/2 else 1
                    itinerary_offsets[column] -= ev.y
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        if moment >= closing:
                            moment = float(opening)
                        paused = not paused
                    elif ev.key == pygame.K_HOME:
                        moment, paused = float(opening), False
                        itinerary_offsets = [0, 0]
                    elif ev.key == pygame.K_END:
                        moment, paused = float(closing), True
                        itinerary_offsets = [0, 0]
                    elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        moment += -300 if ev.key == pygame.K_LEFT else 300
                        paused = True
                    elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        speed = min(7200, speed*2)
                    elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed = max(1, speed/2)
            mouse = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0] and timeline.inflate(0, 22).collidepoint(mouse):
                moment = opening+(closing-opening)*max(0, min(1, (mouse[0]-timeline.x)/timeline.w))
                paused = True
            if not paused:
                moment += dt*speed
            moment = max(opening, min(closing, moment))
            if moment >= closing:
                paused = True
            screen.fill(BG)
            text("OmniQueue / RL vs DP", 24, 13, face=title)
            text(f"{payload['day']}   {clock_text(opening)} – {clock_text(closing)}   recorded posted waits", 24, 49, MUTED)
            text(clock_text(moment), width-190, 13, face=large)
            text(f"{'Paused' if paused else 'Playing'} · {speed:g}x", width-190, 52, MUTED)
            tick_index = max(0, bisect_right(payload["timestamps"], moment)-1)
            board = payload["waits"][tick_index] if moment < closing else [None]*config.NUM_RIDES
            hover = None
            for column, run in enumerate(payload["runs"]):
                color = COLORS[column]
                x = 24+column*(panel_w+24)
                pygame.draw.rect(screen, PANEL, (x, 88, panel_w, height-182), border_radius=12)
                visits = completed_at(run, moment)
                done = {e["ride_id"] for e in visits}
                metrics = run["metrics"]
                prefs = payload["profile"]["preferences"]
                mass = sum(p*p for p in prefs)
                utility = 100*sum(prefs[j]**2 for j in done)/mass
                required = set(payload["profile"]["must_dos"])
                text(f"{run['name']}   {len(done)} rides", x+16, 98, color, title)
                text(f"Preference {sum(prefs[j] for j in done):.3f}   Utility {utility:.1f}%   "
                     f"Must-do {len(done & required)}/{len(required)}", x+16, 136)
                totals = {kind: sum(max(0, min(moment, e["end"])-e["start"]) for e in run["events"]
                                    if e["kind"] == kind and e["start"] <= moment)
                          for kind in ("walk", "queue", "idle")}
                text(f"Queue {totals['queue']/60:.0f}m   Walk {totals['walk']/60:.0f}m   "
                     f"Idle {totals['idle']/60:.0f}m   Day walking {metrics['walk_m']/1000:.2f}km", x+16, 165, MUTED, small)
                map_rect = pygame.Rect(x+(panel_w-map_size)//2, map_top, map_size, map_size)

                def xy(point):
                    return (int(map_rect.x+point[0]*map_size/1000), int(map_rect.y+point[1]*map_size/1000))

                if map_size not in backgrounds:
                    surface = pygame.Surface((map_size, map_size))
                    surface.fill((17, 29, 38))
                    for edge in walks.data["edges"]:
                        points = [(int(p[0]*map_size/1000), int(p[1]*map_size/1000)) for p in edge["geometry"]]
                        if len(points) >= 2:
                            pygame.draw.lines(surface, (48, 66, 77), False, points, 2)
                    backgrounds[map_size] = surface
                screen.blit(backgrounds[map_size], map_rect)
                active = None
                location = config.NUM_RIDES
                for e in run["events"]:
                    if e["start"] > moment:
                        break
                    location = e["location"]
                    if e["kind"] == "walk":
                        poly = walks.paths[e["source"], e["target"]]
                        if e["end"] <= moment:
                            pygame.draw.lines(screen, tuple(int(c*0.6) for c in color), False, [xy(p) for p in poly], 2)
                            location = e["target"]
                    if e["start"] <= moment < e["end"]:
                        active = e
                for rid, point in enumerate(walks.points[:-1]):
                    pos = xy([point["x"], point["y"]])
                    fill = (66, 177, 129) if rid in done else ((66, 93, 113) if board[rid] is not None else (73, 62, 73))
                    pygame.draw.circle(screen, fill, pos, 8)
                    if rid in required:
                        pygame.draw.circle(screen, (255, 219, 98), pos, 11, 2)
                    label = small.render(str(rid), True, TEXT)
                    screen.blit(label, (pos[0]+9, pos[1]-16))
                    if math.dist(mouse, pos) <= 13:
                        hover = rid
                if active and active["kind"] == "walk":
                    poly = walks.paths[active["source"], active["target"]]
                    pygame.draw.lines(screen, color, False, [xy(p) for p in poly], 3)
                    position = point_on_path(poly, (moment-active["start"])/max(1, active["end"]-active["start"]))
                else:
                    point = walks.points[location]
                    position = [point["x"], point["y"]]
                pygame.draw.circle(screen, TEXT, xy(position), 10)
                pygame.draw.circle(screen, color, xy(position), 7)
                state_text = active["kind"].upper() if active else "FINISHED"
                if active and "ride_id" in active:
                    state_text += " · "+config.RIDES[active["ride_id"]]["name"]
                text(clipped(state_text, panel_w-32), x+16, map_top-28, color, small)
                y = map_rect.bottom+10
                decisions = [d for d in run["decisions"] if d["now"] <= moment]
                if decisions:
                    d = decisions[-1]
                    if "forecast_as_of" in d:
                        status = d["planner"]
                        line = (f"Forecast as of {clock_text(d['forecast_as_of'])} · {d['forecast_history_ticks']} history ticks · "
                                f"{'budget reached' if status.get('timed_out') else status['status']}")
                    else:
                        line = f"Last decision {clock_text(d['now'])} · {len(d['completed'])} rides already done"
                    text(clipped(line, panel_w-32, small), x+16, y, MUTED, small)
                    y += 24
                all_visits = [e for e in run["events"] if e["kind"] == "complete"]
                current_index = len(visits)
                text("Completed / upcoming itinerary · wheel to scroll", x+16, y, face=small)
                y += 23
                rows = max(1, int((height-118-y)//23))
                base = max(0, min(current_index-3, len(all_visits)-rows))
                start = max(0, min(base+itinerary_offsets[column], max(0, len(all_visits)-rows)))
                itinerary_offsets[column] = start-base
                for e in all_visits[start:start+rows]:
                    past = e["end"] <= moment
                    rid = e["ride_id"]
                    line = f"{clock_text(e['end'])}  {'done' if past else 'next'}  {rid:02}  {config.RIDES[rid]['name']}"
                    text(clipped(line, panel_w-32, small), x+16, y, color if past else MUTED, small)
                    y += 23
            if hover is not None:
                wait = board[hover]
                label = f"{hover}: {config.RIDES[hover]['name']} | {'unavailable' if wait is None else str(wait//60)+' min'}"
            else:
                label = "Green: completed · Gold ring: must-do · Hover a ride for its recorded wait · Paths show the route so far"
            text(clipped(label, width-48, small), 24, height-95, MUTED, small)
            pygame.draw.rect(screen, (51, 68, 86), timeline, border_radius=8)
            filled = max(1, int(timeline.w*(moment-opening)/(closing-opening)))
            pygame.draw.rect(screen, COLORS[0], (timeline.x, timeline.y, filled, timeline.h), border_radius=8)
            text("Space play/pause   Home restart   End results   ←/→ 5 min   +/- speed   Drag timeline   Esc close", 24, height-39, MUTED, small)
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                if screenshot:
                    pygame.image.save(screen, str(screenshot))
                break
    finally:
        pygame.quit()
