"""Pygame visualization of a recorded park day (Phase 4).

Usage:
    python visualize.py --seed 42
    python visualize.py --seed 42 --speed 120 --sample-interval 60
"""

from __future__ import annotations

import argparse
import bisect
import math
import sys
from dataclasses import dataclass, field

import config
from park_graph import get_park_graph
from simulator import native_backend_name, record_day

# Layout (park coords are ~0–1000; display scaled ~15% down)
UI_SCALE = 0.85
PARK_LOGICAL = 1000
PARK_WIDTH = int(PARK_LOGICAL * UI_SCALE)
PARK_HEIGHT = int(PARK_LOGICAL * UI_SCALE)
SIDEBAR_WIDTH = int(320 * UI_SCALE)
CONTROL_HEIGHT = int(70 * UI_SCALE)
SCREEN_WIDTH = PARK_WIDTH + SIDEBAR_WIDTH
SCREEN_HEIGHT = PARK_HEIGHT + CONTROL_HEIGHT
FPS = 60
MAX_WALK_DOTS = 2500


def _s(v: float) -> int:
    """Scale a layout length from unscaled pixels to the display size."""
    return max(1, int(round(v * UI_SCALE)))


def _xy(x: float, y: float) -> tuple[int, int]:
    """Map park / layout coordinates onto the scaled display."""
    return int(x * UI_SCALE), int(y * UI_SCALE)


@dataclass
class ReplayState:
    """Indexed recording for scrubbing and party tracking."""

    parties: list
    walks: list
    ride_samples: list
    ride_completions: list
    metrics: object
    node_coords: list[tuple[float, float]]
    walks_by_party: dict[int, list[int]] = field(default_factory=dict)
    completions_by_party: dict[int, list] = field(default_factory=dict)
    sample_secs: list[int] = field(default_factory=list)
    sorted_party_ids: list[int] = field(default_factory=list)
    # Minute bucket -> walk indices that overlap that minute
    walks_by_minute: dict[int, list[int]] = field(default_factory=dict)

    @classmethod
    def from_recording(cls, recording, node_coords: list[tuple[float, float]]) -> ReplayState:
        walks = list(recording.walks)
        parties = list(recording.parties)
        samples = list(recording.ride_samples)
        completions = list(recording.ride_completions)

        walks_by_party: dict[int, list[int]] = {}
        walks_by_minute: dict[int, list[int]] = {}
        for i, w in enumerate(walks):
            walks_by_party.setdefault(int(w.party_id), []).append(i)
            start_m = int(w.start_sec) // 60
            end_m = max(start_m, (int(w.end_sec) - 1) // 60)
            for m in range(start_m, end_m + 1):
                walks_by_minute.setdefault(m, []).append(i)

        completions_by_party: dict[int, list] = {}
        for ev in completions:
            completions_by_party.setdefault(int(ev.party_id), []).append(ev)

        rides_map = {int(p.party_id): int(p.rides_completed) for p in parties}
        sorted_ids = sorted(rides_map.keys(), key=lambda pid: rides_map[pid], reverse=True)

        return cls(
            parties=parties,
            walks=walks,
            ride_samples=samples,
            ride_completions=completions,
            metrics=recording.metrics,
            node_coords=node_coords,
            walks_by_party=walks_by_party,
            completions_by_party=completions_by_party,
            sample_secs=[int(s.sec) for s in samples],
            sorted_party_ids=sorted_ids,
            walks_by_minute=walks_by_minute,
        )


def _node_xy(state: ReplayState, idx: int) -> tuple[float, float]:
    return state.node_coords[idx]


def format_clock(sec: float) -> str:
    total = int(sec)
    hour = config.DAY_START_HOUR + total // 3600
    minute = (total % 3600) // 60
    second = total % 60
    am_pm = "AM" if hour < 12 else "PM"
    display_hour = hour % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour:02d}:{minute:02d}:{second:02d} {am_pm}"


def ride_state_at(state: ReplayState, sec: float) -> object | None:
    if not state.ride_samples:
        return None
    i = bisect.bisect_right(state.sample_secs, int(sec)) - 1
    if i < 0:
        i = 0
    return state.ride_samples[i]


def active_walks_at(state: ReplayState, sec: float) -> list:
    """Return walks active at ``sec`` (start <= sec < end)."""
    minute = int(sec) // 60
    candidates = state.walks_by_minute.get(minute, ())
    out = []
    for wi in candidates:
        w = state.walks[wi]
        if w.start_sec <= sec < w.end_sec:
            out.append(w)
    return out


def walk_position(state: ReplayState, walk, sec: float) -> tuple[float, float]:
    from pathways import interpolate_polyline

    planned = int(getattr(walk, "planned_end_sec", walk.end_sec))
    duration = max(1, planned - int(walk.start_sec))
    progress = max(0.0, min(1.0, (sec - float(walk.start_sec)) / duration))
    park = get_park_graph()
    variant = int(getattr(walk, "path_variant", 0) or 0)
    poly = park.path_polyline_for_idx(int(walk.from_idx), int(walk.to_idx), variant=variant)
    if len(poly) >= 2:
        return interpolate_polyline(poly, progress)
    sx, sy = _node_xy(state, int(walk.from_idx))
    ex, ey = _node_xy(state, int(walk.to_idx))
    return sx + (ex - sx) * progress, sy + (ey - sy) * progress


def party_state_at(state: ReplayState, party_id: int, sec: float) -> dict | None:
    party = next((p for p in state.parties if int(p.party_id) == party_id), None)
    if party is None:
        return None
    if sec < float(party.spawn_sec):
        return {"status": "Not arrived", "pos": None}

    idxs = state.walks_by_party.get(party_id, [])
    current = None
    for wi in idxs:
        w = state.walks[wi]
        if w.start_sec <= sec:
            current = w
        else:
            break

    if current is None:
        return {
            "status": "At entrance",
            "pos": _node_xy(state, 0),
        }

    if current.start_sec <= sec < current.end_sec:
        pos = walk_position(state, current, sec)
        target = int(current.target_ride)
        if target == -1:
            label = "Walking to exit"
        elif target == -2:
            label = "Idle walk"
        else:
            name = config.RIDES[target]["name"] if 0 <= target < len(config.RIDES) else f"ride {target}"
            label = f"Walking to {name}"
        dest = _node_xy(state, int(current.to_idx))
        return {"status": label, "pos": pos, "dest": dest}

    # After walk ended
    target = int(current.target_ride)
    pos = _node_xy(state, int(current.to_idx))
    if target == -1:
        return {"status": "Left park", "pos": None if not current.cancelled else pos}
    if target == -2:
        return {"status": "Idle at node", "pos": pos}
    if current.cancelled:
        return {"status": "Re-routing", "pos": walk_position(state, current, float(current.end_sec))}

    name = config.RIDES[target]["name"] if 0 <= target < len(config.RIDES) else f"ride {target}"
    return {"status": f"At {name}", "pos": pos}


def rides_so_far(state: ReplayState, party_id: int, sec: float) -> list[str]:
    events = state.completions_by_party.get(party_id, [])
    names = []
    for ev in events:
        if ev.sec <= sec:
            rid = int(ev.ride_id)
            if 0 <= rid < len(config.RIDES):
                names.append(config.RIDES[rid]["name"])
    return names


def build_node_coords() -> list[tuple[float, float]]:
    park = get_park_graph()
    coords = []
    for i in range(park.num_nodes):
        nid = park.idx_to_node(i)
        coords.append(park._graph.node_coords[nid])
    return coords


def run_visualizer(
    seed: int = 42,
    speed: float = 60.0,
    sample_interval: int = 60,
    max_seconds: int | None = None,
    screenshot_path: str | None = None,
    screenshot_sec: float = 3 * 3600,
) -> None:
    import pygame

    if native_backend_name() != "native":
        raise SystemExit("Native simulator required. Run: pip install -e .")

    print(f"Recording day (seed={seed})...")
    recording = record_day(seed=seed, sample_interval_sec=sample_interval)
    node_coords = build_node_coords()
    state = ReplayState.from_recording(recording, node_coords)
    day_end = float(max_seconds if max_seconds is not None else config.DAY_SECONDS)

    print(
        f"Ready: {len(state.parties)} parties, {len(state.walks)} walks, "
        f"{len(state.ride_samples)} ride samples, "
        f"{state.metrics.rides_completed} rides completed"
    )

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("OmniQueue — Park Day Visualizer")
    clock = pygame.time.Clock()

    small_font = pygame.font.SysFont("DejaVu Sans", _s(11))
    wait_font = pygame.font.SysFont("DejaVu Sans", _s(13), bold=True)
    large_font = pygame.font.SysFont("DejaVu Sans", _s(22), bold=True)
    ui_font = pygame.font.SysFont("DejaVu Sans", _s(15), bold=True)

    # Colors
    BG = (28, 36, 44)
    PATH = (55, 72, 88)
    HUB = (70, 95, 110)
    ENTRANCE = (46, 125, 70)
    RIDE_OPEN = (52, 120, 200)
    RIDE_BROKEN = (190, 55, 55)
    WALK_DOT = (160, 175, 190)
    PANEL = (22, 28, 34)
    ACCENT = (240, 190, 60)
    TEXT = (230, 235, 240)
    MUTED = (140, 150, 160)

    float_sec = 0.0
    playing = True
    dragging = False
    list_scroll_y = 0
    selected_party: int | None = None
    sort_descending = True
    sim_speed = speed

    play_btn = pygame.Rect(_s(24), PARK_HEIGHT + _s(20), _s(90), _s(32))
    slider = pygame.Rect(_s(140), PARK_HEIGHT + _s(28), PARK_WIDTH - _s(280), _s(14))
    speed_minus = pygame.Rect(PARK_WIDTH - _s(120), PARK_HEIGHT + _s(20), _s(36), _s(32))
    speed_plus = pygame.Rect(PARK_WIDTH - _s(50), PARK_HEIGHT + _s(20), _s(36), _s(32))
    sidebar_list = pygame.Rect(PARK_WIDTH + _s(16), _s(78), SIDEBAR_WIDTH - _s(32), _s(300))
    sort_btn = pygame.Rect(PARK_WIDTH + _s(16), _s(50), _s(180), _s(22))
    row_h = _s(28)

    party_by_id = {int(p.party_id): p for p in state.parties}

    def resort() -> None:
        nonlocal state
        state.sorted_party_ids = sorted(
            party_by_id.keys(),
            key=lambda pid: party_by_id[pid].rides_completed,
            reverse=sort_descending,
        )

    resort()
    if state.sorted_party_ids:
        selected_party = state.sorted_party_ids[0]

    def draw() -> None:
        screen.fill(BG)

        # Pathways (OSM walkway polylines when available)
        park = get_park_graph()
        from pathways import load_pathways

        pathways = load_pathways()
        if pathways is not None:
            for poly in pathways.all_edge_polylines():
                if len(poly) < 2:
                    continue
                pygame.draw.lines(
                    screen,
                    PATH,
                    False,
                    [_xy(x, y) for x, y in poly],
                    max(1, _s(2)),
                )
        else:
            for a, b in config.MACRO_EDGES:
                ax, ay = park._graph.node_coords[a]
                bx, by = park._graph.node_coords[b]
                pygame.draw.line(screen, PATH, _xy(ax, ay), _xy(bx, by), max(1, _s(2)))
            for ride_id, hub in enumerate(config.RIDE_HUB):
                rx, ry = config.RIDES[ride_id]["coords"]
                hx, hy = config.HUB_COORDS[hub]
                pygame.draw.line(screen, PATH, _xy(hx, hy), _xy(rx, ry), 1)

        # Hubs
        for nid, (hx, hy) in config.HUB_COORDS.items():
            if nid == config.NODE_ENTRANCE:
                continue
            pygame.draw.circle(screen, HUB, _xy(hx, hy), _s(6))

        # Entrance
        ex, ey = _xy(*config.ENTRANCE_COORDS)
        pygame.draw.rect(
            screen,
            ENTRANCE,
            (ex - _s(40), ey - _s(18), _s(80), _s(36)),
            border_radius=_s(4),
        )
        ent = wait_font.render("ENTRANCE", True, TEXT)
        screen.blit(ent, (ex - ent.get_width() // 2, ey - _s(8)))

        # Clock + metrics
        clock_txt = large_font.render(format_clock(float_sec), True, TEXT)
        screen.blit(clock_txt, (_s(16), _s(12)))
        meta = ui_font.render(
            f"seed {seed}  |  rides {state.metrics.rides_completed}  |  "
            f"var {state.metrics.avg_wait_variance():.0f}",
            True,
            MUTED,
        )
        screen.blit(meta, (_s(16), _s(42)))

        # Walking crowd (subsample if dense)
        walks_now = active_walks_at(state, float_sec)
        step = max(1, len(walks_now) // MAX_WALK_DOTS)
        for i in range(0, len(walks_now), step):
            w = walks_now[i]
            x, y = walk_position(state, w, float_sec)
            pygame.draw.circle(screen, WALK_DOT, _xy(x, y), max(1, _s(2)))

        # Rides
        sample = ride_state_at(state, float_sec)
        for ride_id, ride in enumerate(config.RIDES):
            x, y = _xy(*ride["coords"])
            broken = bool(sample.broken[ride_id]) if sample is not None else False
            wait = float(sample.wait[ride_id]) if sample is not None else 0.0
            color = RIDE_BROKEN if broken else RIDE_OPEN
            pygame.draw.circle(screen, color, (x, y), _s(15))
            pygame.draw.circle(screen, (10, 10, 10), (x, y), _s(15), 1)
            label = "X" if broken or wait >= 9000 else str(int(wait / 60))
            wt = wait_font.render(label, True, TEXT)
            screen.blit(wt, (x - wt.get_width() // 2, y - wt.get_height() // 2))
            words = ride["name"].split()
            short = " ".join(words[:2]) + ("…" if len(words) > 2 else "")
            nt = small_font.render(short, True, MUTED)
            screen.blit(nt, (x - nt.get_width() // 2, y + _s(17)))

        # Tracked party
        if selected_party is not None:
            g = party_state_at(state, selected_party, float_sec)
            if g and g.get("pos"):
                gx, gy = _xy(*g["pos"])
                if "dest" in g:
                    dx, dy = _xy(*g["dest"])
                    pygame.draw.line(screen, ACCENT, (gx, gy), (dx, dy), max(1, _s(2)))
                pulse = abs(math.sin(pygame.time.get_ticks() / 200.0)) * 5
                pygame.draw.circle(screen, ACCENT, (gx, gy), _s(10 + pulse))
                pygame.draw.circle(screen, (0, 0, 0), (gx, gy), _s(10 + pulse), max(1, _s(2)))
                lbl = wait_font.render(f"#{selected_party}", True, (20, 20, 20), ACCENT)
                screen.blit(lbl, (gx + _s(14), gy - _s(10)))

        # Control bar
        pygame.draw.rect(screen, PANEL, (0, PARK_HEIGHT, PARK_WIDTH, CONTROL_HEIGHT))
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
        screen.blit(spd, (PARK_WIDTH - _s(95), PARK_HEIGHT + _s(6)))

        # Sidebar
        pygame.draw.rect(screen, PANEL, (PARK_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))
        pygame.draw.line(screen, (80, 90, 100), (PARK_WIDTH, 0), (PARK_WIDTH, SCREEN_HEIGHT), max(1, _s(2)))
        title = large_font.render("PARTIES", True, TEXT)
        screen.blit(title, (PARK_WIDTH + _s(16), _s(16)))

        pygame.draw.rect(screen, (50, 60, 70), sort_btn, border_radius=_s(4))
        sort_lbl = small_font.render(
            "↓ Most rides" if sort_descending else "↑ Fewest rides", True, TEXT
        )
        screen.blit(sort_lbl, (sort_btn.x + _s(10), sort_btn.y + _s(4)))

        pygame.draw.rect(screen, (18, 22, 28), sidebar_list, border_radius=_s(5))
        start_idx = int(list_scroll_y // row_h)
        end_idx = min(len(state.sorted_party_ids), start_idx + (sidebar_list.h // row_h) + 1)
        for i in range(start_idx, end_idx):
            pid = state.sorted_party_ids[i]
            total = party_by_id[pid].rides_completed
            y_pos = sidebar_list.y + (i * row_h) - list_scroll_y
            if not (sidebar_list.y <= y_pos <= sidebar_list.bottom - row_h):
                continue
            color = (90, 170, 255) if pid == selected_party else (200, 205, 210)
            txt = wait_font.render(f"#{pid}   ({total} rides)", True, color)
            screen.blit(txt, (sidebar_list.x + _s(10), y_pos + _s(6)))
        pygame.draw.rect(screen, (90, 100, 110), sidebar_list, 1, border_radius=_s(5))

        pygame.draw.line(
            screen,
            (80, 90, 100),
            (PARK_WIDTH + _s(16), _s(400)),
            (SCREEN_WIDTH - _s(16), _s(400)),
            1,
        )
        if selected_party is not None:
            g = party_state_at(state, selected_party, float_sec)
            t1 = large_font.render(f"Tracking #{selected_party}", True, ACCENT)
            screen.blit(t1, (PARK_WIDTH + _s(16), _s(420)))
            status = g["status"] if g else "Unknown"
            if len(status) > 34:
                status = status[:31] + "…"
            t2 = ui_font.render(status, True, (120, 220, 140) if g and g.get("pos") else (220, 100, 100))
            screen.blit(t2, (PARK_WIDTH + _s(16), _s(455)))
            hist = rides_so_far(state, selected_party, float_sec)
            total = party_by_id[selected_party].rides_completed
            t3 = ui_font.render(f"Itinerary ({len(hist)}/{total})", True, MUTED)
            screen.blit(t3, (PARK_WIDTH + _s(16), _s(490)))
            for idx, name in enumerate(hist[-12:]):
                short = name if len(name) <= 32 else name[:29] + "…"
                rtxt = wait_font.render(f"✓ {short}", True, (200, 205, 210))
                screen.blit(rtxt, (PARK_WIDTH + _s(24), _s(520) + idx * _s(22)))
        else:
            hint = ui_font.render("Click a party to track", True, MUTED)
            screen.blit(hint, (PARK_WIDTH + _s(16), _s(420)))

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
        dt = clock.tick(FPS) / 1000.0
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
                        sort_descending = not sort_descending
                        list_scroll_y = 0
                        resort()
                    elif sidebar_list.collidepoint(mx, my):
                        click_y = my - sidebar_list.y + list_scroll_y
                        idx = int(click_y // row_h)
                        if 0 <= idx < len(state.sorted_party_ids):
                            selected_party = state.sorted_party_ids[idx]
                elif event.button == 4 and mx > PARK_WIDTH:
                    list_scroll_y = max(0, list_scroll_y - row_h)
                elif event.button == 5 and mx > PARK_WIDTH:
                    max_scroll = max(0, len(state.sorted_party_ids) * row_h - sidebar_list.h)
                    list_scroll_y = min(max_scroll, list_scroll_y + row_h)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEWHEEL and mx > PARK_WIDTH:
                list_scroll_y = max(0, list_scroll_y - event.y * row_h)
                max_scroll = max(0, len(state.sorted_party_ids) * row_h - sidebar_list.h)
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
    parser = argparse.ArgumentParser(description="OmniQueue park-day visualizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=60.0, help="Simulated seconds per real second")
    parser.add_argument("--sample-interval", type=int, default=60, help="Ride wait sample interval (sec)")
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Optional shorter day for quick demos",
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
        default=3 * 3600,
        help="Simulated second for --screenshot",
    )
    args = parser.parse_args(argv)
    run_visualizer(
        seed=args.seed,
        speed=args.speed,
        sample_interval=args.sample_interval,
        max_seconds=args.max_seconds,
        screenshot_path=args.screenshot,
        screenshot_sec=args.screenshot_sec,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
