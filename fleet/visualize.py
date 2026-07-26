"""Pygame visualization of a recorded fleet day (PPO policy).

Usage:
    python visualize.py --seed 42 --checkpoint checkpoints/ppo/ppo_final.pt
    python visualize.py --seed 42 --checkpoint checkpoints/ppo/ppo_final.pt --speed 120
"""

from __future__ import annotations

import argparse
import bisect
import heapq
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Parent of the fleet/ package dir must be on sys.path for `import fleet.*`.
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from fleet.simulator import native_backend_name, record_day_ppo

# Layout (city coords map into a square panel; display scaled ~15% down)
UI_SCALE = 0.85
MAP_LOGICAL = 1000
MAP_WIDTH = int(MAP_LOGICAL * UI_SCALE)
MAP_HEIGHT = int(MAP_LOGICAL * UI_SCALE)
SIDEBAR_WIDTH = int(320 * UI_SCALE)
CONTROL_HEIGHT = int(70 * UI_SCALE)
SCREEN_WIDTH = MAP_WIDTH + SIDEBAR_WIDTH
SCREEN_HEIGHT = MAP_HEIGHT + CONTROL_HEIGHT
FPS = 60
MAX_PENDING_DOTS = 400
MAX_FRAME_DT = 1.0 / 30.0

# RequestStatus mirror (C++ enum as uint8)
STATUS_SCHEDULED = 0
STATUS_PENDING = 1
STATUS_ASSIGNED = 2
STATUS_PICKED_UP = 3
STATUS_COMPLETED = 4
STATUS_CANCELLED = 5

TRIP_KIND_NAMES = {0: "pickup", 1: "dropoff", 2: "reposition"}


def _s(v: float) -> int:
    """Scale a layout length from unscaled pixels to the display size."""
    return max(1, int(round(v * UI_SCALE)))


def _xy(x: float, y: float) -> tuple[int, int]:
    """Map layout coordinates onto the scaled display."""
    return int(x * UI_SCALE), int(y * UI_SCALE)


@dataclass
class ReplayState:
    """Indexed recording for scrubbing and vehicle tracking."""

    recording: object
    node_coords: list[tuple[float, float]]
    city_width: float
    city_height: float
    trips: list
    requests: list
    samples: list
    metrics: object
    horizon_sec: int
    num_vehicles: int
    adj: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    path_cache: dict[tuple[int, int], tuple[list[tuple[float, float]], list[float], float]] = field(
        default_factory=dict
    )
    trips_by_vehicle: dict[int, list[int]] = field(default_factory=dict)
    trips_by_minute: dict[int, list[int]] = field(default_factory=dict)
    sample_secs: list[int] = field(default_factory=list)
    sorted_vehicle_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_recording(cls, recording) -> ReplayState:
        trips = list(recording.trips)
        requests = list(recording.requests)
        samples = list(recording.samples)
        city = recording.city
        node_coords = [(float(x), float(y)) for x, y in city.nodes]

        # Street graph with Euclidean edge weights (matches City.hpp Dijkstra).
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(node_coords))}
        for u, v in city.edges:
            ui, vi = int(u), int(v)
            ax, ay = node_coords[ui]
            bx, by = node_coords[vi]
            # Match City.hpp: rounded Euclidean length, minimum 1.
            w = float(max(1, int(round(math.hypot(bx - ax, by - ay)))))
            adj[ui].append((vi, w))
            adj[vi].append((ui, w))

        trips_by_vehicle: dict[int, list[int]] = {}
        trips_by_minute: dict[int, list[int]] = {}
        for i, t in enumerate(trips):
            vid = int(t.vehicle_id)
            trips_by_vehicle.setdefault(vid, []).append(i)
            start_m = int(t.start_sec) // 60
            end_m = max(start_m, (max(int(t.end_sec), int(t.start_sec) + 1) - 1) // 60)
            for m in range(start_m, end_m + 1):
                trips_by_minute.setdefault(m, []).append(i)
        for idxs in trips_by_vehicle.values():
            idxs.sort(key=lambda i: int(trips[i].start_sec))

        num_vehicles = int(recording.num_vehicles)
        return cls(
            recording=recording,
            node_coords=node_coords,
            city_width=float(city.width) if city.width > 1 else 1.0,
            city_height=float(city.height) if city.height > 1 else 1.0,
            trips=trips,
            requests=requests,
            samples=samples,
            metrics=recording.metrics,
            horizon_sec=int(recording.horizon_sec),
            num_vehicles=num_vehicles,
            adj=adj,
            trips_by_vehicle=trips_by_vehicle,
            trips_by_minute=trips_by_minute,
            sample_secs=[int(s.sec) for s in samples],
            sorted_vehicle_ids=list(range(num_vehicles)),
        )


def format_clock(sec: float) -> str:
    """Format elapsed seconds as a 24h clock starting at midnight."""
    total = max(0, int(sec))
    hour = total // 3600
    minute = (total % 3600) // 60
    second = total % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def metric_sample_at(state: ReplayState, sec: float) -> object | None:
    if not state.samples:
        return None
    i = bisect.bisect_right(state.sample_secs, int(sec)) - 1
    if i < 0:
        i = 0
    return state.samples[i]


def active_trips_at(state: ReplayState, sec: float) -> list:
    """Return trips active at ``sec`` (start <= sec < end, or instantaneous)."""
    minute = int(sec) // 60
    candidates = state.trips_by_minute.get(minute, ())
    out = []
    for ti in candidates:
        t = state.trips[ti]
        start = float(t.start_sec)
        end = float(t.end_sec)
        if end <= start:
            if abs(sec - start) < 1e-6:
                out.append(t)
        elif start <= sec < end:
            out.append(t)
    return out


def shortest_path_nodes(state: ReplayState, src: int, dst: int) -> list[int]:
    """Dijkstra node path; falls back to [src, dst] if unreachable."""
    if src == dst:
        return [src]
    dist = {src: 0.0}
    prev: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in state.adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dst not in prev and src != dst:
        return [src, dst]
    path = [dst]
    cur = dst
    while cur != src:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def trip_polyline(state: ReplayState, from_node: int, to_node: int) -> tuple[
    list[tuple[float, float]], list[float], float
]:
    """Cached street-following polyline with cumulative arc lengths."""
    key = (from_node, to_node)
    cached = state.path_cache.get(key)
    if cached is not None:
        return cached

    nodes = state.node_coords
    if from_node == to_node:
        pt = nodes[from_node]
        cached = ([pt], [0.0], 0.0)
        state.path_cache[key] = cached
        return cached

    path = shortest_path_nodes(state, from_node, to_node)
    poly = [nodes[i] for i in path]
    cum = [0.0]
    for i in range(1, len(poly)):
        ax, ay = poly[i - 1]
        bx, by = poly[i]
        cum.append(cum[-1] + math.hypot(bx - ax, by - ay))
    total = cum[-1] if cum else 0.0
    cached = (poly, cum, total)
    state.path_cache[key] = cached
    return cached


def point_along_polyline(
    poly: list[tuple[float, float]], cum: list[float], total: float, frac: float
) -> tuple[float, float]:
    """Interpolate along a polyline by fraction of total length in [0, 1]."""
    if not poly:
        return 0.0, 0.0
    if total <= 1e-9 or frac <= 0.0:
        return poly[0]
    if frac >= 1.0:
        return poly[-1]
    target = frac * total
    i = bisect.bisect_right(cum, target) - 1
    i = max(0, min(i, len(poly) - 2))
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 1e-9 else (target - cum[i]) / seg
    ax, ay = poly[i]
    bx, by = poly[i + 1]
    return ax + t * (bx - ax), ay + t * (by - ay)


def trip_position(state: ReplayState, trip, sec: float) -> tuple[float, float]:
    """Lerp along the shortest street path (not a straight chord)."""
    fn = int(trip.from_node)
    tn = int(trip.to_node)
    poly, cum, total = trip_polyline(state, fn, tn)
    start = float(trip.start_sec)
    end = float(trip.end_sec)
    if end <= start or sec <= start:
        return poly[0]
    if sec >= end:
        return poly[-1]
    frac = (sec - start) / (end - start)
    return point_along_polyline(poly, cum, total, frac)


def trip_path_tail(
    state: ReplayState, trip, sec: float
) -> list[tuple[float, float]]:
    """Remaining route from current position to trip destination (for accent line)."""
    fn = int(trip.from_node)
    tn = int(trip.to_node)
    poly, cum, total = trip_polyline(state, fn, tn)
    start = float(trip.start_sec)
    end = float(trip.end_sec)
    if not poly:
        return []
    if end <= start or sec <= start:
        return poly
    if sec >= end:
        return [poly[-1]]
    frac = (sec - start) / (end - start)
    cur = point_along_polyline(poly, cum, total, frac)
    target = frac * total
    i = bisect.bisect_right(cum, target) - 1
    i = max(0, min(i, len(poly) - 2))
    return [cur, *poly[i + 1 :]]


def request_status_at(req, sec: float) -> int:
    """Derive request status at ``sec`` from recorded timestamps."""
    final = int(req.status)
    spawn = float(req.spawn_sec)
    if sec < spawn:
        return STATUS_SCHEDULED
    assign = float(req.assign_sec)
    pickup = float(req.pickup_sec)
    dropoff = float(req.dropoff_sec)
    if dropoff >= 0 and sec >= dropoff:
        return STATUS_COMPLETED
    if pickup >= 0 and sec >= pickup:
        return STATUS_PICKED_UP
    if assign >= 0 and sec >= assign:
        return STATUS_ASSIGNED
    if final == STATUS_CANCELLED and dropoff < 0 and pickup < 0:
        # Cancelled at horizon — treat as pending/assigned until end of recording use.
        if assign >= 0:
            return STATUS_ASSIGNED
        return STATUS_PENDING
    return STATUS_PENDING


def waiting_requests_at(state: ReplayState, sec: float) -> list[tuple[object, int]]:
    """Guests still at origin: Pending or Assigned (not yet picked up)."""
    out: list[tuple[object, int]] = []
    for r in state.requests:
        st = request_status_at(r, sec)
        if st in (STATUS_PENDING, STATUS_ASSIGNED):
            out.append((r, st))
    return out


def vehicle_state_at(state: ReplayState, vehicle_id: int, sec: float) -> dict:
    """Position / status for one vehicle at ``sec``."""
    trips = state.trips_by_vehicle.get(vehicle_id, ())
    active = None
    last_ended = None
    for ti in trips:
        t = state.trips[ti]
        start = float(t.start_sec)
        end = float(t.end_sec)
        if end <= start:
            end = start
        if start <= sec < end or (end == start and abs(sec - start) < 1e-6):
            active = t
            break
        if start <= sec:
            last_ended = t
        if start > sec:
            break

    if active is not None:
        x, y = trip_position(state, active, sec)
        try:
            kind_i = int(active.kind)
        except Exception:
            kind_i = int(getattr(active.kind, "value", 0))
        eta = max(0.0, float(active.end_sec) - sec)
        return {
            "pos": (x, y),
            "status": TRIP_KIND_NAMES.get(kind_i, "busy"),
            "dest": state.node_coords[int(active.to_node)],
            "request_id": int(active.request_id),
            "eta": eta,
            "trip": active,
        }

    if last_ended is not None:
        x, y = state.node_coords[int(last_ended.to_node)]
        return {
            "pos": (x, y),
            "status": "idle",
            "dest": None,
            "request_id": -1,
            "eta": 0.0,
            "trip": None,
        }

    # Never moved (or before first trip) — use recorded start node.
    starts = getattr(state.recording, "vehicle_start_nodes", None) or ()
    if vehicle_id < len(starts):
        node = int(starts[vehicle_id])
        x, y = state.node_coords[node]
    elif trips:
        t0 = state.trips[trips[0]]
        x, y = state.node_coords[int(t0.from_node)]
    elif state.node_coords:
        x, y = state.node_coords[0]
    else:
        x, y = 0.0, 0.0
    return {
        "pos": (x, y),
        "status": "idle",
        "dest": None,
        "request_id": -1,
        "eta": 0.0,
        "trip": None,
    }


def map_city_to_screen(state: ReplayState, x: float, y: float) -> tuple[int, int]:
    """Map city coordinates into the map panel (with margin)."""
    margin = 24.0
    draw_w = MAP_LOGICAL - 2 * margin
    draw_h = MAP_LOGICAL - 2 * margin
    sx = x / max(1.0, state.city_width - 1.0)
    sy = y / max(1.0, state.city_height - 1.0)
    px = margin + sx * draw_w
    py = margin + sy * draw_h
    return _xy(px, py)


def run_visualizer(
    seed: int = 42,
    checkpoint: str = "checkpoints/ppo/ppo_final.pt",
    speed: float = 60.0,
    sample_interval: int = 60,
    horizon_sec: int = 3600,
    num_vehicles: int = 30,
    num_requests: int = 120,
    num_intersections: int = 80,
    city_width: int | None = None,
    city_height: int | None = None,
    vehicle_speed: float = 2.0,
    device: str = "cpu",
    max_seconds: int | None = None,
    screenshot_path: str | None = None,
    screenshot_sec: float = 1800,
) -> None:
    import pygame

    if native_backend_name() != "native":
        raise SystemExit("Native simulator required. Run: pip install -e .")

    print(f"Recording PPO day (seed={seed}, checkpoint={checkpoint})...")
    recording = record_day_ppo(
        seed=seed,
        checkpoint=checkpoint,
        sample_interval_sec=sample_interval,
        device=device,
        city_width=city_width,
        city_height=city_height,
        num_intersections=num_intersections,
        num_vehicles=num_vehicles,
        num_requests=num_requests,
        horizon_sec=horizon_sec,
        vehicle_speed=vehicle_speed,
    )
    state = ReplayState.from_recording(recording)
    day_end = float(max_seconds if max_seconds is not None else state.horizon_sec)

    print(
        f"Ready: {state.num_vehicles} vehicles, {len(state.requests)} requests, "
        f"{len(state.trips)} trips, {len(state.samples)} samples, "
        f"{state.metrics.requests_completed} completed"
    )

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("OmniQueue — Fleet PPO Visualizer")
    clock = pygame.time.Clock()

    small_font = pygame.font.SysFont("DejaVu Sans", _s(11))
    wait_font = pygame.font.SysFont("DejaVu Sans", _s(13), bold=True)
    large_font = pygame.font.SysFont("DejaVu Sans", _s(22), bold=True)
    ui_font = pygame.font.SysFont("DejaVu Sans", _s(15), bold=True)

    BG = (28, 36, 44)
    STREET = (55, 72, 88)
    NODE = (70, 95, 110)
    VEHICLE_IDLE = (120, 180, 120)
    VEHICLE_PICKUP = (90, 170, 255)
    VEHICLE_DROPOFF = (240, 190, 60)
    VEHICLE_IDLE_TRIP = (180, 140, 220)
    PENDING_DOT = (220, 100, 100)
    ASSIGNED_DOT = (230, 150, 70)  # matched, still waiting at origin for pickup
    PANEL = (22, 28, 34)
    ACCENT = (240, 190, 60)
    TEXT = (230, 235, 240)
    MUTED = (140, 150, 160)

    # Static city backdrop
    city_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
    city_surface.fill(BG)
    for u, v in state.recording.city.edges:
        ax, ay = state.node_coords[int(u)]
        bx, by = state.node_coords[int(v)]
        pygame.draw.line(
            city_surface,
            STREET,
            map_city_to_screen(state, ax, ay),
            map_city_to_screen(state, bx, by),
            max(1, _s(2)),
        )
    for x, y in state.node_coords:
        pygame.draw.circle(city_surface, NODE, map_city_to_screen(state, x, y), _s(4))

    float_sec = 0.0
    playing = True
    dragging = False
    list_scroll_y = 0
    selected_vehicle: int | None = 0 if state.num_vehicles > 0 else None
    sort_by_busy = True
    sim_speed = speed

    play_btn = pygame.Rect(_s(24), MAP_HEIGHT + _s(20), _s(90), _s(32))
    slider = pygame.Rect(_s(140), MAP_HEIGHT + _s(28), MAP_WIDTH - _s(280), _s(14))
    speed_minus = pygame.Rect(MAP_WIDTH - _s(120), MAP_HEIGHT + _s(20), _s(36), _s(32))
    speed_plus = pygame.Rect(MAP_WIDTH - _s(50), MAP_HEIGHT + _s(20), _s(36), _s(32))
    sidebar_list = pygame.Rect(MAP_WIDTH + _s(16), _s(200), SIDEBAR_WIDTH - _s(32), _s(220))
    sort_btn = pygame.Rect(MAP_WIDTH + _s(16), _s(172), _s(200), _s(22))
    row_h = _s(28)

    def vehicle_busy_key(vid: int) -> float:
        total = 0.0
        for ti in state.trips_by_vehicle.get(vid, ()):
            t = state.trips[ti]
            total += max(0.0, float(t.end_sec) - float(t.start_sec))
        return total

    def resort() -> None:
        state.sorted_vehicle_ids = sorted(
            range(state.num_vehicles),
            key=vehicle_busy_key,
            reverse=sort_by_busy,
        )

    resort()

    status_colors = {
        "idle": VEHICLE_IDLE,
        "pickup": VEHICLE_PICKUP,
        "dropoff": VEHICLE_DROPOFF,
        "reposition": VEHICLE_IDLE_TRIP,
        "busy": VEHICLE_PICKUP,
    }

    def draw() -> None:
        screen.blit(city_surface, (0, 0))

        sample = metric_sample_at(state, float_sec)
        if sample is not None:
            pending_n = int(sample.pending)
            free_n = int(sample.free_vehicles)
            busy_n = int(sample.busy_vehicles)
            completed_n = int(sample.completed)
            mean_wait = float(sample.mean_wait)
        else:
            pending_n = sum(
                1 for r in state.requests if request_status_at(r, float_sec) == STATUS_PENDING
            )
            free_n = busy_n = completed_n = 0
            mean_wait = 0.0

        clock_txt = large_font.render(format_clock(float_sec), True, TEXT)
        screen.blit(clock_txt, (_s(16), _s(12)))
        meta = ui_font.render(
            f"seed {seed}  |  PPO  |  done {completed_n}/{state.metrics.requests_spawned}  |  "
            f"wait {mean_wait:.0f}s",
            True,
            MUTED,
        )
        screen.blit(meta, (_s(16), _s(42)))

        # Guests waiting at origin (unassigned = red, assigned awaiting pickup = orange)
        waiting = waiting_requests_at(state, float_sec)
        step = max(1, len(waiting) // MAX_PENDING_DOTS)
        for i in range(0, len(waiting), step):
            r, st = waiting[i]
            ox, oy = state.node_coords[int(r.origin)]
            color = PENDING_DOT if st == STATUS_PENDING else ASSIGNED_DOT
            pygame.draw.circle(
                screen, color, map_city_to_screen(state, ox, oy), max(2, _s(7))
            )

        # Vehicles
        for vid in range(state.num_vehicles):
            if vid == selected_vehicle:
                continue  # draw selected on top
            g = vehicle_state_at(state, vid, float_sec)
            color = status_colors.get(g["status"], VEHICLE_IDLE)
            px, py = map_city_to_screen(state, *g["pos"])
            pygame.draw.circle(screen, color, (px, py), _s(6))
            pygame.draw.circle(screen, (10, 10, 10), (px, py), _s(6), 1)

        if selected_vehicle is not None:
            g = vehicle_state_at(state, selected_vehicle, float_sec)
            color = status_colors.get(g["status"], VEHICLE_IDLE)
            gx, gy = map_city_to_screen(state, *g["pos"])
            trip = g.get("trip")
            if trip is not None:
                tail = trip_path_tail(state, trip, float_sec)
                if len(tail) >= 2:
                    pygame.draw.lines(
                        screen,
                        ACCENT,
                        False,
                        [map_city_to_screen(state, x, y) for x, y in tail],
                        max(1, _s(2)),
                    )
            pulse = abs(math.sin(pygame.time.get_ticks() / 200.0)) * 5
            pygame.draw.circle(screen, color, (gx, gy), _s(8))
            pygame.draw.circle(screen, ACCENT, (gx, gy), _s(10 + pulse), max(1, _s(2)))
            lbl = wait_font.render(f"V{selected_vehicle}", True, (20, 20, 20), ACCENT)
            screen.blit(lbl, (gx + _s(14), gy - _s(10)))

        # Control bar
        pygame.draw.rect(screen, PANEL, (0, MAP_HEIGHT, MAP_WIDTH, CONTROL_HEIGHT))
        btn_color = (70, 160, 90) if playing else (180, 80, 80)
        pygame.draw.rect(screen, btn_color, play_btn, border_radius=_s(5))
        btn_lbl = wait_font.render("PAUSE" if playing else "PLAY", True, TEXT)
        screen.blit(
            btn_lbl,
            (
                play_btn.centerx - btn_lbl.get_width() // 2,
                play_btn.centery - btn_lbl.get_height() // 2,
            ),
        )

        pygame.draw.rect(screen, (70, 80, 90), slider, border_radius=_s(5))
        progress = float_sec / max(1.0, day_end - 1)
        pygame.draw.rect(
            screen,
            (90, 150, 230),
            (slider.x, slider.y, int(slider.w * progress), slider.h),
            border_radius=_s(5),
        )
        pygame.draw.circle(
            screen,
            TEXT,
            (slider.x + int(slider.w * progress), slider.centery),
            _s(8),
        )

        pygame.draw.rect(screen, (70, 80, 90), speed_minus, border_radius=_s(4))
        pygame.draw.rect(screen, (70, 80, 90), speed_plus, border_radius=_s(4))
        screen.blit(
            wait_font.render("−", True, TEXT),
            (speed_minus.centerx - _s(4), speed_minus.centery - _s(7)),
        )
        screen.blit(
            wait_font.render("+", True, TEXT),
            (speed_plus.centerx - _s(5), speed_plus.centery - _s(7)),
        )
        spd = small_font.render(f"{sim_speed:.0f}x", True, MUTED)
        screen.blit(spd, (MAP_WIDTH - _s(95), MAP_HEIGHT + _s(6)))

        # Sidebar
        pygame.draw.rect(screen, PANEL, (MAP_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))
        pygame.draw.line(
            screen, (80, 90, 100), (MAP_WIDTH, 0), (MAP_WIDTH, SCREEN_HEIGHT), max(1, _s(2))
        )
        title = large_font.render("FLEET", True, TEXT)
        screen.blit(title, (MAP_WIDTH + _s(16), _s(16)))

        completion = state.metrics.completion_rate() * 100.0
        lines = [
            f"Pending: {pending_n}",
            f"Free / busy: {free_n} / {busy_n}",
            f"Completed: {completed_n}",
            f"Mean wait: {mean_wait:.0f}s",
            f"Final completion: {completion:.1f}%",
            f"Assignments: {state.metrics.assignments}",
        ]
        for i, line in enumerate(lines):
            screen.blit(
                ui_font.render(line, True, MUTED if i else TEXT),
                (MAP_WIDTH + _s(16), _s(50) + i * _s(20)),
            )

        pygame.draw.rect(screen, (50, 60, 70), sort_btn, border_radius=_s(4))
        sort_lbl = small_font.render(
            "↓ Most busy" if sort_by_busy else "↑ Least busy", True, TEXT
        )
        screen.blit(sort_lbl, (sort_btn.x + _s(10), sort_btn.y + _s(4)))

        pygame.draw.rect(screen, (18, 22, 28), sidebar_list, border_radius=_s(5))
        start_idx = int(list_scroll_y // row_h)
        end_idx = min(len(state.sorted_vehicle_ids), start_idx + (sidebar_list.h // row_h) + 1)
        for i in range(start_idx, end_idx):
            vid = state.sorted_vehicle_ids[i]
            g = vehicle_state_at(state, vid, float_sec)
            y_pos = sidebar_list.y + (i * row_h) - list_scroll_y
            if not (sidebar_list.y <= y_pos <= sidebar_list.bottom - row_h):
                continue
            color = (90, 170, 255) if vid == selected_vehicle else (200, 205, 210)
            txt = wait_font.render(f"V{vid}  {g['status']}", True, color)
            screen.blit(txt, (sidebar_list.x + _s(10), y_pos + _s(6)))
        pygame.draw.rect(screen, (90, 100, 110), sidebar_list, 1, border_radius=_s(5))

        pygame.draw.line(
            screen,
            (80, 90, 100),
            (MAP_WIDTH + _s(16), _s(440)),
            (SCREEN_WIDTH - _s(16), _s(440)),
            1,
        )
        if selected_vehicle is not None:
            g = vehicle_state_at(state, selected_vehicle, float_sec)
            t1 = large_font.render(f"Vehicle {selected_vehicle}", True, ACCENT)
            screen.blit(t1, (MAP_WIDTH + _s(16), _s(455)))
            status = g["status"]
            t2 = ui_font.render(status.upper(), True, (120, 220, 140))
            screen.blit(t2, (MAP_WIDTH + _s(16), _s(490)))
            rid = int(g["request_id"])
            if rid >= 0:
                req = state.requests[rid] if rid < len(state.requests) else None
                detail = f"Request #{rid}"
                if req is not None:
                    detail += f"  O{int(req.origin)}→D{int(req.dest)}"
                screen.blit(
                    ui_font.render(detail, True, TEXT),
                    (MAP_WIDTH + _s(16), _s(520)),
                )
                screen.blit(
                    ui_font.render(f"ETA {g['eta']:.0f}s", True, MUTED),
                    (MAP_WIDTH + _s(16), _s(545)),
                )
            else:
                screen.blit(
                    ui_font.render("No assignment", True, MUTED),
                    (MAP_WIDTH + _s(16), _s(520)),
                )
            n_trips = len(state.trips_by_vehicle.get(selected_vehicle, ()))
            screen.blit(
                small_font.render(f"{n_trips} trips recorded", True, MUTED),
                (MAP_WIDTH + _s(16), _s(575)),
            )
        else:
            hint = ui_font.render("Click a vehicle to track", True, MUTED)
            screen.blit(hint, (MAP_WIDTH + _s(16), _s(455)))

        # Legend
        legend_y = SCREEN_HEIGHT - _s(70)
        for i, (label, col) in enumerate(
            [
                ("Idle", VEHICLE_IDLE),
                ("Pickup", VEHICLE_PICKUP),
                ("Dropoff", VEHICLE_DROPOFF),
                ("Unassigned", PENDING_DOT),
                ("Awaiting", ASSIGNED_DOT),
            ]
        ):
            lx = MAP_WIDTH + _s(16) + (i % 2) * _s(140)
            ly = legend_y + (i // 2) * _s(22)
            pygame.draw.circle(screen, col, (lx + _s(6), ly + _s(8)), _s(5))
            screen.blit(small_font.render(label, True, MUTED), (lx + _s(18), ly + _s(2)))

        pygame.display.flip()

    if screenshot_path:
        float_sec = min(float(screenshot_sec), day_end - 1)
        playing = False
        draw()
        pygame.image.save(screen, screenshot_path)
        print(f"Wrote screenshot {screenshot_path} at t={format_clock(float_sec)}")
        pygame.quit()
        return

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, MAX_FRAME_DT)
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    sim_speed = min(600.0, sim_speed * 1.5)
                elif event.key == pygame.K_MINUS:
                    sim_speed = max(1.0, sim_speed / 1.5)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if play_btn.collidepoint(mx, my):
                        playing = not playing
                    elif slider.collidepoint(mx, my) or (
                        abs(my - slider.centery) < _s(20) and slider.x <= mx <= slider.right
                    ):
                        dragging = True
                        playing = False
                        progress = max(0.0, min(1.0, (mx - slider.x) / slider.w))
                        float_sec = progress * (day_end - 1)
                    elif speed_minus.collidepoint(mx, my):
                        sim_speed = max(1.0, sim_speed / 1.5)
                    elif speed_plus.collidepoint(mx, my):
                        sim_speed = min(600.0, sim_speed * 1.5)
                    elif sort_btn.collidepoint(mx, my):
                        sort_by_busy = not sort_by_busy
                        list_scroll_y = 0
                        resort()
                    elif sidebar_list.collidepoint(mx, my):
                        click_y = my - sidebar_list.y + list_scroll_y
                        idx = int(click_y // row_h)
                        if 0 <= idx < len(state.sorted_vehicle_ids):
                            selected_vehicle = state.sorted_vehicle_ids[idx]
                elif event.button == 4 and mx > MAP_WIDTH:
                    list_scroll_y = max(0, list_scroll_y - row_h)
                elif event.button == 5 and mx > MAP_WIDTH:
                    max_scroll = max(0, len(state.sorted_vehicle_ids) * row_h - sidebar_list.h)
                    list_scroll_y = min(max_scroll, list_scroll_y + row_h)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEWHEEL and mx > MAP_WIDTH:
                list_scroll_y = max(0, list_scroll_y - event.y * row_h)
                max_scroll = max(0, len(state.sorted_vehicle_ids) * row_h - sidebar_list.h)
                list_scroll_y = min(max_scroll, list_scroll_y)

        if dragging:
            progress = max(0.0, min(1.0, (mx - slider.x) / slider.w))
            float_sec = progress * (day_end - 1)
        elif playing:
            float_sec += dt * sim_speed
            if float_sec >= day_end - 1:
                float_sec = day_end - 1
                playing = False

        draw()

    pygame.quit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OmniQueue fleet PPO visualizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ppo/ppo_final.pt",
        help="PPO checkpoint from ppo_train (contains model state_dict)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--speed", type=float, default=60.0, help="Simulated seconds per real second")
    parser.add_argument(
        "--sample-interval", type=int, default=60, help="Metric sample interval (sec)"
    )
    parser.add_argument(
        "--horizon-sec",
        type=int,
        default=3600,
        help="Episode length if not stored in checkpoint config",
    )
    parser.add_argument("--num-vehicles", type=int, default=30)
    parser.add_argument("--num-requests", type=int, default=120)
    parser.add_argument("--num-intersections", type=int, default=80)
    parser.add_argument("--city-width", type=int, default=None)
    parser.add_argument("--city-height", type=int, default=None)
    parser.add_argument("--vehicle-speed", type=float, default=2.0)
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Optional shorter replay window",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Write one frame to this path and exit (for CI / docs)",
    )
    parser.add_argument(
        "--screenshot-sec",
        type=float,
        default=1800,
        help="Simulated second for --screenshot",
    )
    args = parser.parse_args(argv)
    run_visualizer(
        seed=args.seed,
        checkpoint=args.checkpoint,
        device=args.device,
        speed=args.speed,
        sample_interval=args.sample_interval,
        horizon_sec=args.horizon_sec,
        num_vehicles=args.num_vehicles,
        num_requests=args.num_requests,
        num_intersections=args.num_intersections,
        city_width=args.city_width,
        city_height=args.city_height,
        vehicle_speed=args.vehicle_speed,
        max_seconds=args.max_seconds,
        screenshot_path=args.screenshot,
        screenshot_sec=args.screenshot_sec,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
