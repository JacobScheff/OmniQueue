"""Pygame interactive play application."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import config
from park_graph import get_park_graph
from play.benchmark import run_ai_compare, run_park_benchmark
from play.driver import HybridDriver
from play.scoring import format_focal_line, format_park_line
from play.session import FocalProfile, SessionStore
from simulator import native_backend_name
from visualize import (
    CONTROL_HEIGHT,
    FPS,
    MAX_FRAME_DT,
    MAX_WALK_DOTS,
    PARK_HEIGHT,
    PARK_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SIDEBAR_WIDTH,
    ReplayState,
    _s,
    _xy,
    active_walks_at,
    build_node_coords,
    format_clock,
    party_state_at,
    ride_state_at,
    walk_position,
)


def _default_weights() -> np.ndarray:
    pops = np.array([float(r["popularity"]) for r in config.RIDES], dtype=np.float32)
    return pops


class PlayApp:
    """Setup → live play → session history / AI compare / benchmark."""

    def __init__(
        self,
        seed: int = 42,
        checkpoint: str | None = None,
        crowd_router: str = "heuristic",
        device: str = "cpu",
        speed: float = 120.0,
        sample_interval: int = 60,
    ) -> None:
        self.seed = int(seed)
        self.checkpoint = checkpoint
        self.crowd_router = crowd_router
        self.device = device
        self.sim_speed = float(speed)
        self.sample_interval = int(sample_interval)

        self.store = SessionStore()
        self.profile = FocalProfile(
            spawn_sec=10 * 60,
            leave_sec=12 * 3600,
            preference_weights=_default_weights(),
            must_dos=np.zeros(config.NUM_RIDES, dtype=np.uint8),
        )
        self.mode = "setup"  # setup | play | results
        self.status_msg = ""
        self.error_msg = ""

        self.driver: HybridDriver | None = None
        self.replay: ReplayState | None = None
        self.node_coords = build_node_coords()
        self.float_sec = 0.0
        self.segment_end = 0.0
        self.pending_decision = None
        self.playing_segment = False
        self.decision_scroll = 0
        self.pref_scroll = 0
        self.history_scroll = 0
        self.sorted_pref_ids = list(range(config.NUM_RIDES))

    def _validate_ppo(self) -> None:
        if not self.checkpoint:
            raise FileNotFoundError("PPO checkpoint path is required.")
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {self.checkpoint}")

    def sort_preferences(self) -> None:
        w = self.profile.preference_weights
        self.sorted_pref_ids = sorted(
            range(config.NUM_RIDES), key=lambda i: float(w[i]), reverse=True
        )

    def start_play(self) -> None:
        self.error_msg = ""
        try:
            if self.crowd_router == "ppo":
                self._validate_ppo()
            self.status_msg = "Starting day (full park)…"
            self.driver = HybridDriver(
                seed=self.seed,
                profile=self.profile,
                crowd_router=self.crowd_router,
                focal_router="human",
                checkpoint=self.checkpoint,
                device=self.device,
                enable_recording=True,
                sample_interval_sec=self.sample_interval,
                soft_human_leave=True,
            )
            self.mode = "play"
            self.float_sec = float(self.profile.spawn_sec)
            self._advance_to_decision_or_done()
        except Exception as exc:  # noqa: BLE001 — show in UI
            self.error_msg = str(exc)
            self.driver = None
            self.mode = "setup"

    def _rebuild_replay(self) -> None:
        assert self.driver is not None
        rec = self.driver.recording()
        if rec is None:
            return
        self.replay = ReplayState.from_recording(rec, self.node_coords)

    def _advance_to_decision_or_done(self) -> None:
        assert self.driver is not None
        self.status_msg = "Simulating until next decision…"
        decision = self.driver.advance()
        self._rebuild_replay()
        if self.replay is not None and decision is not None:
            # Warm polylines for the upcoming animation window only (full-day prefetch is too heavy).
            lo = max(0.0, self.float_sec - 60.0)
            hi = float(decision.now_sec) + 120.0
            keys_warmed = 0
            park = get_park_graph()
            for w in self.replay.walks:
                if float(w.end_sec) < lo or float(w.start_sec) > hi:
                    continue
                park.path_arc_for_idx(
                    int(w.from_idx),
                    int(w.to_idx),
                    variant=int(getattr(w, "path_variant", 0) or 0),
                )
                keys_warmed += 1
                if keys_warmed > 5000:
                    break
        if decision is None:
            self.pending_decision = None
            self.playing_segment = False
            run = self.driver.to_session_run(kind="human", label="human play")
            self.store.add(run)
            self.status_msg = "Day complete — " + format_focal_line(run.focal)
            self.mode = "results"
            return
        self.pending_decision = decision
        self.segment_end = float(decision.now_sec)
        self.playing_segment = self.float_sec < self.segment_end - 1e-3
        if not self.playing_segment:
            self.status_msg = "Your turn — pick a ride, Wander, or Exit"
        else:
            self.status_msg = f"Decision at {format_clock(decision.now_sec)}"

    def apply_action(self, action: int) -> None:
        if self.driver is None or self.pending_decision is None:
            return
        self.driver.apply_human_action(action)
        self.pending_decision = None
        self.playing_segment = False
        self.float_sec = self.segment_end
        self._advance_to_decision_or_done()

    def run_compare(self) -> None:
        self.error_msg = ""
        try:
            self._validate_ppo()
            self.status_msg = "Running 4 AI compare cells…"
            runs = run_ai_compare(
                seed=self.seed,
                profile=self.profile,
                checkpoint=self.checkpoint,
                store=self.store,
                device=self.device,
            )
            self.status_msg = f"AI compare done ({len(runs)} cells)."
            self.mode = "results"
        except Exception as exc:  # noqa: BLE001
            self.error_msg = str(exc)

    def run_benchmark(self, n_days: int = 3) -> None:
        self.error_msg = ""
        try:
            self._validate_ppo()
            self.status_msg = f"Benchmarking {n_days} days…"
            result = run_park_benchmark(
                seed_start=self.seed,
                n_days=n_days,
                checkpoint=self.checkpoint,  # type: ignore[arg-type]
                store=self.store,
                device=self.device,
            )
            self.status_msg = " | ".join(
                x.strip() for x in result.summary_lines() if x.strip()
            )
            self.mode = "results"
        except Exception as exc:  # noqa: BLE001
            self.error_msg = str(exc)

    def run(self) -> None:
        import pygame
        from park_graph import get_park_graph
        from pathways import load_pathways

        if native_backend_name() != "native":
            raise SystemExit("Native simulator required. Run: pip install -e .")

        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("OmniQueue — Interactive Play")
        clock = pygame.time.Clock()

        font = pygame.font.SysFont("DejaVu Sans", _s(14))
        small = pygame.font.SysFont("DejaVu Sans", _s(12))
        title_f = pygame.font.SysFont("DejaVu Sans", _s(22), bold=True)
        bold = pygame.font.SysFont("DejaVu Sans", _s(14), bold=True)

        BG = (28, 36, 44)
        PANEL = (22, 28, 34)
        PATH = (55, 72, 88)
        HUB = (70, 95, 110)
        ENTRANCE = (46, 125, 70)
        RIDE_OPEN = (52, 120, 200)
        RIDE_BROKEN = (190, 55, 55)
        WALK_DOT = (160, 175, 190)
        ACCENT = (240, 190, 60)
        TEXT = (230, 235, 240)
        MUTED = (140, 150, 160)
        BTN = (60, 110, 170)
        BTN2 = (70, 140, 90)
        ERR = (220, 90, 90)

        park_surface = pygame.Surface((PARK_WIDTH, PARK_HEIGHT))
        park_surface.fill(BG)
        park = get_park_graph()
        pathways = load_pathways()
        if pathways is not None:
            for poly in pathways.all_edge_polylines():
                if len(poly) < 2:
                    continue
                pygame.draw.lines(
                    park_surface, PATH, False, [_xy(x, y) for x, y in poly], max(1, _s(2))
                )
        for nid, (hx, hy) in config.HUB_COORDS.items():
            if nid == config.NODE_ENTRANCE:
                continue
            pygame.draw.circle(park_surface, HUB, _xy(hx, hy), _s(6))
        ex, ey = _xy(*config.ENTRANCE_COORDS)
        pygame.draw.rect(
            park_surface, ENTRANCE, (ex - _s(40), ey - _s(18), _s(80), _s(36)), border_radius=_s(4)
        )

        self.sort_preferences()

        # Buttons (setup)
        btn_play = pygame.Rect(_s(20), SCREEN_HEIGHT - _s(50), _s(140), _s(34))
        btn_compare = pygame.Rect(_s(180), SCREEN_HEIGHT - _s(50), _s(160), _s(34))
        btn_bench = pygame.Rect(_s(360), SCREEN_HEIGHT - _s(50), _s(160), _s(34))
        btn_sort = pygame.Rect(_s(20), _s(200), _s(180), _s(28))
        btn_crowd = pygame.Rect(_s(220), _s(200), _s(200), _s(28))
        btn_setup = pygame.Rect(PARK_WIDTH + _s(16), SCREEN_HEIGHT - _s(50), _s(120), _s(34))

        # Decision action buttons
        btn_exit = pygame.Rect(PARK_WIDTH + _s(16), SCREEN_HEIGHT - _s(100), _s(120), _s(32))
        btn_idle = pygame.Rect(PARK_WIDTH + _s(150), SCREEN_HEIGHT - _s(100), _s(120), _s(32))

        row_h = _s(26)
        pref_list = pygame.Rect(_s(20), _s(240), PARK_WIDTH - _s(40), SCREEN_HEIGHT - _s(310))
        ride_list = pygame.Rect(PARK_WIDTH + _s(12), _s(80), SIDEBAR_WIDTH - _s(24), _s(420))
        hist_list = pygame.Rect(_s(20), _s(80), SCREEN_WIDTH - _s(40), SCREEN_HEIGHT - _s(150))

        def draw_button(rect, label, color=BTN) -> None:
            pygame.draw.rect(screen, color, rect, border_radius=_s(5))
            t = bold.render(label, True, TEXT)
            screen.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))

        def draw_setup() -> None:
            screen.fill(PANEL)
            screen.blit(title_f.render("Interactive Play — Setup", True, TEXT), (_s(20), _s(16)))
            lines = [
                f"Seed: {self.seed}   (CLI --seed)",
                f"Enter: {format_clock(self.profile.spawn_sec)}   Leave (soft): {format_clock(self.profile.leave_sec)}",
                f"Checkpoint: {self.checkpoint or '(none — required for PPO)'}",
                "Keys: [ / ] spawn ±30m   ; / ' leave ±30m   1-9 weight focus   Space toggle must-do",
                "Click ride row to select; +/- change weight; M toggle must-do; S sort",
            ]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, MUTED), (_s(20), _s(52) + i * _s(22)))

            draw_button(btn_sort, "Sort by preference", BTN2)
            crowd_lbl = f"Crowd: {self.crowd_router.upper()}"
            draw_button(btn_crowd, crowd_lbl, BTN if self.crowd_router == "heuristic" else (140, 90, 160))

            pygame.draw.rect(screen, (18, 22, 28), pref_list, border_radius=_s(5))
            start = int(self.pref_scroll // row_h)
            visible = pref_list.h // row_h + 1
            for i in range(start, min(len(self.sorted_pref_ids), start + visible)):
                rid = self.sorted_pref_ids[i]
                y = pref_list.y + (i * row_h) - self.pref_scroll
                if not (pref_list.y <= y <= pref_list.bottom - row_h):
                    continue
                w = float(self.profile.preference_weights[rid])
                md = bool(self.profile.must_dos[rid])
                name = config.RIDES[rid]["name"]
                short = name if len(name) <= 36 else name[:33] + "…"
                check = "[x]" if md else "[ ]"
                color = ACCENT if md else TEXT
                screen.blit(
                    small.render(f"{check} #{rid:02d}  w={w:6.1f}  {short}", True, color),
                    (pref_list.x + _s(8), y + _s(4)),
                )
            pygame.draw.rect(screen, (90, 100, 110), pref_list, 1, border_radius=_s(5))

            draw_button(btn_play, "Play day", BTN2)
            draw_button(btn_compare, "AI compare (4)", BTN)
            draw_button(btn_bench, "Benchmark 3d", BTN)
            if self.status_msg:
                screen.blit(font.render(self.status_msg, True, ACCENT), (_s(540), SCREEN_HEIGHT - _s(42)))
            if self.error_msg:
                screen.blit(font.render(self.error_msg, True, ERR), (_s(20), SCREEN_HEIGHT - _s(80)))

        def draw_play() -> None:
            screen.blit(park_surface, (0, 0))
            if self.replay is not None:
                walks_now = active_walks_at(self.replay, self.float_sec)
                step = max(1, len(walks_now) // MAX_WALK_DOTS)
                for i in range(0, len(walks_now), step):
                    w = walks_now[i]
                    x, y = walk_position(self.replay, w, self.float_sec)
                    pygame.draw.circle(screen, WALK_DOT, _xy(x, y), max(1, _s(2)))

                sample = ride_state_at(self.replay, self.float_sec)
                for ride_id, ride in enumerate(config.RIDES):
                    x, y = _xy(*ride["coords"])
                    broken = bool(sample.broken[ride_id]) if sample is not None else False
                    wait = float(sample.wait[ride_id]) if sample is not None else 0.0
                    color = RIDE_BROKEN if broken else RIDE_OPEN
                    pygame.draw.circle(screen, color, (x, y), _s(15))
                    label = "X" if broken or wait >= 9000 else str(int(wait / 60))
                    wt = bold.render(label, True, TEXT)
                    screen.blit(wt, (x - wt.get_width() // 2, y - wt.get_height() // 2))

                # Focal party highlight (party 0)
                g = party_state_at(self.replay, 0, self.float_sec)
                if g and g.get("pos"):
                    gx, gy = _xy(*g["pos"])
                    pulse = abs(math.sin(pygame.time.get_ticks() / 200.0)) * 5
                    pygame.draw.circle(screen, ACCENT, (gx, gy), _s(10 + pulse))

            # HUD
            pygame.draw.rect(screen, PANEL, (0, PARK_HEIGHT, PARK_WIDTH, CONTROL_HEIGHT))
            screen.blit(
                title_f.render(format_clock(self.float_sec), True, TEXT), (_s(16), PARK_HEIGHT + _s(18))
            )
            screen.blit(
                font.render(
                    f"{self.sim_speed:.0f}x  crowd={self.crowd_router}  {self.status_msg}",
                    True,
                    MUTED,
                ),
                (_s(200), PARK_HEIGHT + _s(26)),
            )

            # Sidebar decision UI
            pygame.draw.rect(screen, PANEL, (PARK_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))
            screen.blit(title_f.render("YOU", True, ACCENT), (PARK_WIDTH + _s(16), _s(16)))
            screen.blit(
                font.render(f"seed {self.seed}  leave {format_clock(self.profile.leave_sec)}", True, MUTED),
                (PARK_WIDTH + _s(16), _s(48)),
            )

            if self.pending_decision is not None and not self.playing_segment:
                dec = self.pending_decision
                screen.blit(bold.render("Choose next action", True, TEXT), (PARK_WIDTH + _s(16), _s(72)))
                pygame.draw.rect(screen, (18, 22, 28), ride_list, border_radius=_s(5))
                order = sorted(
                    range(config.NUM_RIDES),
                    key=lambda i: float(dec.preferences[i]),
                    reverse=True,
                )
                start = int(self.decision_scroll // row_h)
                visible = ride_list.h // row_h
                for i in range(start, min(len(order), start + visible)):
                    rid = order[i]
                    y = ride_list.y + (i - start) * row_h
                    wait_m = dec.waits[rid] / 60.0
                    open_ok = bool(dec.open_mask[rid]) and wait_m < 150
                    md = bool(dec.must_do_remaining[rid])
                    pref = float(dec.preferences[rid])
                    name = config.RIDES[rid]["name"]
                    short = name if len(name) <= 22 else name[:19] + "…"
                    mark = "*" if md else " "
                    col = TEXT if open_ok else MUTED
                    if md:
                        col = ACCENT
                    screen.blit(
                        small.render(
                            f"{mark}{rid:02d} {wait_m:4.0f}m p={pref:.2f} {short}",
                            True,
                            col,
                        ),
                        (ride_list.x + _s(6), y + _s(4)),
                    )
                draw_button(btn_exit, "Exit", (160, 70, 70))
                draw_button(btn_idle, "Wander", BTN)
            elif self.playing_segment:
                screen.blit(font.render("Walking / watching crowd…", True, MUTED), (PARK_WIDTH + _s(16), _s(100)))
                screen.blit(font.render("Space pauses segment", True, MUTED), (PARK_WIDTH + _s(16), _s(130)))
            else:
                screen.blit(font.render(self.status_msg or "…", True, MUTED), (PARK_WIDTH + _s(16), _s(100)))

        def draw_results() -> None:
            screen.fill(PANEL)
            screen.blit(title_f.render("Session runs (this launch only)", True, TEXT), (_s(20), _s(16)))
            draw_button(btn_setup, "Back to setup", BTN)
            pygame.draw.rect(screen, (18, 22, 28), hist_list, border_radius=_s(5))
            y = hist_list.y + _s(8) - self.history_scroll
            for run in self.store.runs:
                block = [
                    run.settings.summary(),
                    "  park:  " + format_park_line(run.park),
                    "  focal: " + format_focal_line(run.focal),
                    f"  enter={format_clock(run.profile.spawn_sec)} leave={format_clock(run.profile.leave_sec)}",
                    "",
                ]
                for line in block:
                    if hist_list.y <= y <= hist_list.bottom - _s(18):
                        screen.blit(small.render(line, True, TEXT if line else MUTED), (hist_list.x + _s(10), y))
                    y += _s(18)
            if self.status_msg:
                screen.blit(font.render(self.status_msg[:120], True, ACCENT), (_s(20), SCREEN_HEIGHT - _s(80)))
            if self.error_msg:
                screen.blit(font.render(self.error_msg, True, ERR), (_s(20), SCREEN_HEIGHT - _s(110)))

        selected_pref_row = 0
        segment_paused = False
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000.0, MAX_FRAME_DT)
            mx, my = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.mode == "play":
                            self.mode = "setup"
                            self.driver = None
                        else:
                            running = False
                    elif self.mode == "setup":
                        if event.key == pygame.K_LEFTBRACKET:
                            self.profile.spawn_sec = max(0, self.profile.spawn_sec - 1800)
                        elif event.key == pygame.K_RIGHTBRACKET:
                            self.profile.spawn_sec = min(
                                config.DAY_SECONDS - config.MIN_DWELL_SEC,
                                self.profile.spawn_sec + 1800,
                            )
                        elif event.key == pygame.K_SEMICOLON:
                            self.profile.leave_sec = max(
                                self.profile.spawn_sec + config.MIN_DWELL_SEC,
                                self.profile.leave_sec - 1800,
                            )
                        elif event.key == pygame.K_QUOTE:
                            self.profile.leave_sec = min(
                                config.DAY_SECONDS, self.profile.leave_sec + 1800
                            )
                        elif event.key == pygame.K_s:
                            self.sort_preferences()
                        elif event.key == pygame.K_c:
                            self.crowd_router = (
                                "ppo" if self.crowd_router == "heuristic" else "heuristic"
                            )
                        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                            rid = self.sorted_pref_ids[min(selected_pref_row, len(self.sorted_pref_ids) - 1)]
                            self.profile.preference_weights[rid] += 1.0
                        elif event.key == pygame.K_MINUS:
                            rid = self.sorted_pref_ids[min(selected_pref_row, len(self.sorted_pref_ids) - 1)]
                            self.profile.preference_weights[rid] = max(
                                0.0, float(self.profile.preference_weights[rid]) - 1.0
                            )
                        elif event.key == pygame.K_m:
                            rid = self.sorted_pref_ids[min(selected_pref_row, len(self.sorted_pref_ids) - 1)]
                            self.profile.must_dos[rid] = 0 if self.profile.must_dos[rid] else 1
                        elif event.key == pygame.K_UP:
                            selected_pref_row = max(0, selected_pref_row - 1)
                        elif event.key == pygame.K_DOWN:
                            selected_pref_row = min(config.NUM_RIDES - 1, selected_pref_row + 1)
                        elif event.key == pygame.K_RETURN:
                            self.start_play()
                    elif self.mode == "play":
                        if event.key == pygame.K_SPACE:
                            segment_paused = not segment_paused
                        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                            self.sim_speed = min(600.0, self.sim_speed * 1.5)
                        elif event.key == pygame.K_MINUS:
                            self.sim_speed = max(1.0, self.sim_speed / 1.5)
                    elif self.mode == "results" and event.key == pygame.K_b:
                        self.mode = "setup"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.mode == "setup":
                        if btn_play.collidepoint(mx, my):
                            self.start_play()
                        elif btn_compare.collidepoint(mx, my):
                            self.run_compare()
                        elif btn_bench.collidepoint(mx, my):
                            self.run_benchmark(3)
                        elif btn_sort.collidepoint(mx, my):
                            self.sort_preferences()
                        elif btn_crowd.collidepoint(mx, my):
                            self.crowd_router = (
                                "ppo" if self.crowd_router == "heuristic" else "heuristic"
                            )
                        elif pref_list.collidepoint(mx, my):
                            idx = int((my - pref_list.y + self.pref_scroll) // row_h)
                            if 0 <= idx < len(self.sorted_pref_ids):
                                selected_pref_row = idx
                                rid = self.sorted_pref_ids[idx]
                                # click left checkbox zone toggles must-do
                                if mx < pref_list.x + _s(40):
                                    self.profile.must_dos[rid] = (
                                        0 if self.profile.must_dos[rid] else 1
                                    )
                    elif self.mode == "play" and self.pending_decision is not None and not self.playing_segment:
                        if btn_exit.collidepoint(mx, my):
                            self.apply_action(34)
                        elif btn_idle.collidepoint(mx, my):
                            self.apply_action(35)
                        elif ride_list.collidepoint(mx, my):
                            order = sorted(
                                range(config.NUM_RIDES),
                                key=lambda i: float(self.pending_decision.preferences[i]),
                                reverse=True,
                            )
                            start = int(self.decision_scroll // row_h)
                            idx = start + int((my - ride_list.y) // row_h)
                            if 0 <= idx < len(order):
                                self.apply_action(order[idx])
                    elif self.mode == "results" and btn_setup.collidepoint(mx, my):
                        self.mode = "setup"
                elif event.type == pygame.MOUSEWHEEL:
                    if self.mode == "setup" and mx < PARK_WIDTH:
                        self.pref_scroll = max(0, self.pref_scroll - event.y * row_h)
                    elif self.mode == "play" and mx > PARK_WIDTH:
                        self.decision_scroll = max(0, self.decision_scroll - event.y * row_h)
                    elif self.mode == "results":
                        self.history_scroll = max(0, self.history_scroll - event.y * _s(40))

            # Animate play segment
            if self.mode == "play" and self.playing_segment and not segment_paused:
                self.float_sec = min(self.segment_end, self.float_sec + dt * self.sim_speed)
                if self.float_sec >= self.segment_end - 1e-6:
                    self.playing_segment = False
                    self.status_msg = "Your turn — pick a ride, Wander, or Exit"

            if self.mode == "setup":
                draw_setup()
            elif self.mode == "play":
                draw_play()
            else:
                draw_results()
            pygame.display.flip()

        pygame.quit()


def run_play_app(**kwargs) -> None:
    PlayApp(**kwargs).run()
