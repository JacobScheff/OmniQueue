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

# Preference slider range (raw UI weights before L1-normalize / must-do boost in sim).
WEIGHT_SLIDER_MAX = 250.0


def _default_weights() -> np.ndarray:
    pops = np.array([float(r["popularity"]) for r in config.RIDES], dtype=np.float32)
    return np.clip(pops, 0.0, WEIGHT_SLIDER_MAX)


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
        self.checkpoint = checkpoint or ""
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
        self.model_input_focused = False
        self.dragging_slider_rid: int | None = None

    def _checkpoint_or_none(self) -> str | None:
        text = (self.checkpoint or "").strip()
        return text or None

    def _validate_ppo(self) -> None:
        path = self._checkpoint_or_none()
        if not path:
            raise FileNotFoundError(
                "PPO model path is required. Set it in Setup or pass --model."
            )
        if not Path(path).is_file():
            raise FileNotFoundError(f"PPO model not found: {path}")

    def sort_preferences(self) -> None:
        w = self.profile.preference_weights
        self.sorted_pref_ids = sorted(
            range(config.NUM_RIDES), key=lambda i: float(w[i]), reverse=True
        )
        self.pref_scroll = 0

    def start_play(self) -> None:
        self.error_msg = ""
        self.model_input_focused = False
        try:
            if self.crowd_router == "ppo":
                self._validate_ppo()
            self.status_msg = "Starting day (full park)…"
            self.driver = HybridDriver(
                seed=self.seed,
                profile=self.profile,
                crowd_router=self.crowd_router,
                focal_router="human",
                checkpoint=self._checkpoint_or_none(),
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
        self.model_input_focused = False
        try:
            self._validate_ppo()
            self.status_msg = "Running 4 AI compare cells…"
            runs = run_ai_compare(
                seed=self.seed,
                profile=self.profile,
                checkpoint=self._checkpoint_or_none(),
                store=self.store,
                device=self.device,
            )
            self.status_msg = f"AI compare done ({len(runs)} cells)."
            self.mode = "results"
        except Exception as exc:  # noqa: BLE001
            self.error_msg = str(exc)

    def run_benchmark(self, n_days: int = 3) -> None:
        self.error_msg = ""
        self.model_input_focused = False
        try:
            self._validate_ppo()
            self.status_msg = f"Benchmarking {n_days} days…"
            result = run_park_benchmark(
                seed_start=self.seed,
                n_days=n_days,
                checkpoint=self._checkpoint_or_none(),  # type: ignore[arg-type]
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
        from pathways import load_pathways

        if native_backend_name() != "native":
            raise SystemExit("Native simulator required. Run: pip install -e .")

        pygame.init()
        # Wider setup-friendly window (preference rows need horizontal room).
        win_w = max(SCREEN_WIDTH, _s(1280))
        win_h = max(SCREEN_HEIGHT, _s(900))
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("OmniQueue — Interactive Play")
        clock = pygame.time.Clock()

        font = pygame.font.SysFont("DejaVu Sans", _s(16))
        ride_name_f = pygame.font.SysFont("DejaVu Sans", _s(22), bold=True)
        title_f = pygame.font.SysFont("DejaVu Sans", _s(26), bold=True)
        bold = pygame.font.SysFont("DejaVu Sans", _s(18), bold=True)
        small = pygame.font.SysFont("DejaVu Sans", _s(14))
        decision_name_f = pygame.font.SysFont("DejaVu Sans", _s(18), bold=True)

        BG = (28, 36, 44)
        PANEL = (22, 28, 34)
        PATH = (55, 72, 88)
        HUB = (70, 95, 110)
        ENTRANCE = (46, 125, 70)
        RIDE_OPEN = (52, 120, 200)
        RIDE_BROKEN = (190, 55, 55)
        WALK_DOT = (160, 175, 190)
        ACCENT = (240, 190, 60)
        TEXT = (235, 240, 245)
        MUTED = (170, 180, 190)
        BTN = (60, 110, 170)
        BTN2 = (70, 140, 90)
        ERR = (220, 90, 90)
        TRACK = (55, 65, 78)
        FILL = (90, 160, 230)
        ROW_BG = (18, 22, 28)
        CHECK_ON = (70, 160, 100)

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

        header_h = _s(168)
        footer_h = _s(64)
        row_h = _s(48)
        name_w = _s(420)
        check_w = _s(120)
        pad = _s(20)

        pref_list = pygame.Rect(
            pad,
            header_h,
            win_w - 2 * pad,
            win_h - header_h - footer_h - _s(12),
        )
        slider_x0 = pref_list.x + name_w + _s(12)
        slider_x1 = pref_list.right - check_w - _s(24)
        slider_w = max(_s(160), slider_x1 - slider_x0)

        btn_play = pygame.Rect(pad, win_h - _s(50), _s(140), _s(36))
        btn_compare = pygame.Rect(pad + _s(160), win_h - _s(50), _s(170), _s(36))
        btn_bench = pygame.Rect(pad + _s(350), win_h - _s(50), _s(170), _s(36))
        btn_sort = pygame.Rect(pad, _s(118), _s(200), _s(34))
        btn_crowd = pygame.Rect(pad + _s(220), _s(118), _s(220), _s(34))
        model_box = pygame.Rect(pad + _s(460), _s(118), win_w - pad - _s(480), _s(34))
        btn_setup = pygame.Rect(PARK_WIDTH + _s(16), win_h - _s(50), _s(140), _s(36))

        btn_exit = pygame.Rect(PARK_WIDTH + _s(16), win_h - _s(110), _s(120), _s(36))
        btn_idle = pygame.Rect(PARK_WIDTH + _s(150), win_h - _s(110), _s(120), _s(36))
        ride_list = pygame.Rect(PARK_WIDTH + _s(12), _s(90), SIDEBAR_WIDTH - _s(24), win_h - _s(220))
        hist_list = pygame.Rect(pad, _s(80), win_w - 2 * pad, win_h - _s(160))
        decision_row_h = _s(36)

        def draw_button(rect, label, color=BTN) -> None:
            pygame.draw.rect(screen, color, rect, border_radius=_s(6))
            t = bold.render(label, True, TEXT)
            screen.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))

        def row_geometry(list_index: int) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, int]:
            """Return (row_rect, slider_rect, check_rect, ride_id) for a visible list index."""
            rid = self.sorted_pref_ids[list_index]
            y = pref_list.y + (list_index * row_h) - self.pref_scroll
            row = pygame.Rect(pref_list.x, y, pref_list.w, row_h - _s(4))
            slider = pygame.Rect(slider_x0, y + _s(16), slider_w, _s(14))
            check = pygame.Rect(pref_list.right - check_w + _s(8), y + _s(8), _s(28), _s(28))
            return row, slider, check, rid

        def set_weight_from_x(rid: int, mx: int) -> None:
            t = (mx - slider_x0) / max(1, slider_w)
            t = max(0.0, min(1.0, t))
            self.profile.preference_weights[rid] = float(t * WEIGHT_SLIDER_MAX)

        def draw_setup() -> None:
            screen.fill(PANEL)
            screen.blit(title_f.render("Interactive Play — Setup", True, TEXT), (pad, _s(14)))

            meta = [
                f"Seed {self.seed}   Enter {format_clock(self.profile.spawn_sec)}   "
                f"Leave (soft) {format_clock(self.profile.leave_sec)}",
                "[ / ] enter ±30m    ; / ' leave ±30m    Scroll the list    Drag sliders    Click must-do",
            ]
            for i, line in enumerate(meta):
                screen.blit(font.render(line, True, MUTED), (pad, _s(52) + i * _s(24)))

            draw_button(btn_sort, "Sort by preference", BTN2)
            crowd_color = BTN if self.crowd_router == "heuristic" else (130, 90, 170)
            draw_button(btn_crowd, f"Crowd AI: {self.crowd_router.upper()}", crowd_color)

            # PPO model path field
            screen.blit(small.render("PPO model", True, MUTED), (model_box.x, model_box.y - _s(18)))
            border = ACCENT if self.model_input_focused else (90, 100, 110)
            pygame.draw.rect(screen, (30, 36, 44), model_box, border_radius=_s(5))
            pygame.draw.rect(screen, border, model_box, 2, border_radius=_s(5))
            path_txt = self.checkpoint or "(click and type path…)"
            # Truncate from the left so the filename stays visible.
            shown = path_txt
            while small.size(shown)[0] > model_box.w - _s(16) and len(shown) > 4:
                shown = "…" + shown[2:]
            col = TEXT if self.checkpoint else MUTED
            screen.blit(small.render(shown, True, col), (model_box.x + _s(8), model_box.y + _s(8)))

            # Column headers
            hdr_y = pref_list.y - _s(28)
            screen.blit(bold.render("Ride", True, MUTED), (pref_list.x + _s(8), hdr_y))
            screen.blit(bold.render("Preference", True, MUTED), (slider_x0, hdr_y))
            screen.blit(bold.render("Must-do", True, MUTED), (pref_list.right - check_w + _s(4), hdr_y))

            pygame.draw.rect(screen, ROW_BG, pref_list, border_radius=_s(6))
            start = max(0, int(self.pref_scroll // row_h))
            visible = pref_list.h // row_h + 2
            for i in range(start, min(len(self.sorted_pref_ids), start + visible)):
                row, slider, check, rid = row_geometry(i)
                if row.bottom < pref_list.y or row.y > pref_list.bottom:
                    continue
                # Clip drawing to list
                if row.y < pref_list.y or row.bottom > pref_list.bottom:
                    continue

                w = float(self.profile.preference_weights[rid])
                md = bool(self.profile.must_dos[rid])
                name = config.RIDES[rid]["name"]

                # Alternating row tint
                if i % 2 == 0:
                    pygame.draw.rect(screen, (24, 30, 38), row, border_radius=_s(4))

                screen.blit(
                    ride_name_f.render(name, True, TEXT),
                    (row.x + _s(10), row.y + _s(10)),
                )

                # Slider track + fill + knob
                pygame.draw.rect(screen, TRACK, slider, border_radius=_s(7))
                frac = max(0.0, min(1.0, w / WEIGHT_SLIDER_MAX))
                fill_w = int(slider.w * frac)
                if fill_w > 0:
                    pygame.draw.rect(
                        screen,
                        FILL,
                        (slider.x, slider.y, fill_w, slider.h),
                        border_radius=_s(7),
                    )
                knob_x = slider.x + fill_w
                pygame.draw.circle(screen, TEXT, (knob_x, slider.centery), _s(9))
                val = small.render(f"{w:.0f}", True, MUTED)
                screen.blit(val, (slider.right + _s(8), slider.y - _s(2)))

                # Must-do checkbox
                pygame.draw.rect(
                    screen,
                    CHECK_ON if md else TRACK,
                    check,
                    border_radius=_s(4),
                )
                pygame.draw.rect(screen, TEXT if md else MUTED, check, 2, border_radius=_s(4))
                if md:
                    mark = bold.render("✓", True, TEXT)
                    screen.blit(
                        mark,
                        (check.centerx - mark.get_width() // 2, check.centery - mark.get_height() // 2),
                    )
                screen.blit(
                    small.render("must-do", True, ACCENT if md else MUTED),
                    (check.right + _s(8), check.y + _s(6)),
                )

            pygame.draw.rect(screen, (90, 100, 110), pref_list, 1, border_radius=_s(6))

            draw_button(btn_play, "Play day", BTN2)
            draw_button(btn_compare, "AI compare (4)", BTN)
            draw_button(btn_bench, "Benchmark 3d", BTN)
            if self.status_msg:
                screen.blit(
                    font.render(self.status_msg[:80], True, ACCENT),
                    (pad + _s(540), win_h - _s(42)),
                )
            if self.error_msg:
                screen.blit(font.render(self.error_msg, True, ERR), (pad, win_h - _s(88)))

        def draw_play() -> None:
            # Map uses original park layout; fill rest of taller window with panel.
            screen.fill(PANEL)
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

                g = party_state_at(self.replay, 0, self.float_sec)
                if g and g.get("pos"):
                    gx, gy = _xy(*g["pos"])
                    pulse = abs(math.sin(pygame.time.get_ticks() / 200.0)) * 5
                    pygame.draw.circle(screen, ACCENT, (gx, gy), _s(10 + pulse))

            pygame.draw.rect(screen, PANEL, (0, PARK_HEIGHT, PARK_WIDTH, CONTROL_HEIGHT))
            screen.blit(
                title_f.render(format_clock(self.float_sec), True, TEXT),
                (_s(16), PARK_HEIGHT + _s(16)),
            )
            screen.blit(
                font.render(
                    f"{self.sim_speed:.0f}x   crowd={self.crowd_router}   {self.status_msg}",
                    True,
                    MUTED,
                ),
                (_s(200), PARK_HEIGHT + _s(24)),
            )

            pygame.draw.rect(screen, PANEL, (PARK_WIDTH, 0, win_w - PARK_WIDTH, win_h))
            screen.blit(title_f.render("Your turn", True, ACCENT), (PARK_WIDTH + _s(16), _s(16)))
            screen.blit(
                font.render(
                    f"seed {self.seed}   leave {format_clock(self.profile.leave_sec)}",
                    True,
                    MUTED,
                ),
                (PARK_WIDTH + _s(16), _s(52)),
            )

            if self.pending_decision is not None and not self.playing_segment:
                dec = self.pending_decision
                screen.blit(
                    bold.render("Pick a ride", True, TEXT),
                    (PARK_WIDTH + _s(16), _s(78)),
                )
                list_rect = pygame.Rect(
                    PARK_WIDTH + _s(12),
                    _s(110),
                    win_w - PARK_WIDTH - _s(24),
                    win_h - _s(240),
                )
                pygame.draw.rect(screen, ROW_BG, list_rect, border_radius=_s(5))
                order = sorted(
                    range(config.NUM_RIDES),
                    key=lambda i: float(dec.preferences[i]),
                    reverse=True,
                )
                start = int(self.decision_scroll // decision_row_h)
                visible = list_rect.h // decision_row_h + 1
                for i in range(start, min(len(order), start + visible)):
                    rid = order[i]
                    y = list_rect.y + (i - start) * decision_row_h
                    wait_m = dec.waits[rid] / 60.0
                    open_ok = bool(dec.open_mask[rid]) and wait_m < 150
                    md = bool(dec.must_do_remaining[rid])
                    name = config.RIDES[rid]["name"]
                    col = ACCENT if md else (TEXT if open_ok else MUTED)
                    prefix = "★ " if md else "  "
                    screen.blit(
                        decision_name_f.render(
                            f"{prefix}{name}",
                            True,
                            col,
                        ),
                        (list_rect.x + _s(10), y + _s(6)),
                    )
                    meta = small.render(f"{wait_m:.0f} min wait", True, MUTED)
                    screen.blit(meta, (list_rect.right - meta.get_width() - _s(12), y + _s(10)))
                # Store for hit-testing
                draw_play.list_rect = list_rect  # type: ignore[attr-defined]
                draw_play.order = order  # type: ignore[attr-defined]
                draw_button(btn_exit, "Exit", (160, 70, 70))
                draw_button(btn_idle, "Wander", BTN)
            elif self.playing_segment:
                screen.blit(
                    font.render("Watching the crowd… (Space to pause)", True, MUTED),
                    (PARK_WIDTH + _s(16), _s(120)),
                )
            else:
                screen.blit(
                    font.render(self.status_msg or "…", True, MUTED),
                    (PARK_WIDTH + _s(16), _s(120)),
                )

        draw_play.list_rect = ride_list  # type: ignore[attr-defined]
        draw_play.order = list(range(config.NUM_RIDES))  # type: ignore[attr-defined]

        def draw_results() -> None:
            screen.fill(PANEL)
            screen.blit(title_f.render("Session runs (this launch only)", True, TEXT), (pad, _s(16)))
            draw_button(btn_setup, "Back to setup", BTN)
            pygame.draw.rect(screen, ROW_BG, hist_list, border_radius=_s(5))
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
                    if hist_list.y <= y <= hist_list.bottom - _s(20):
                        screen.blit(
                            font.render(line, True, TEXT if line else MUTED),
                            (hist_list.x + _s(10), y),
                        )
                    y += _s(22)
            if self.status_msg:
                screen.blit(font.render(self.status_msg[:120], True, ACCENT), (pad, win_h - _s(80)))
            if self.error_msg:
                screen.blit(font.render(self.error_msg, True, ERR), (pad, win_h - _s(110)))

        segment_paused = False
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000.0, MAX_FRAME_DT)
            mx, my = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.mode == "setup" and self.model_input_focused:
                        if event.key == pygame.K_RETURN:
                            self.model_input_focused = False
                        elif event.key == pygame.K_ESCAPE:
                            self.model_input_focused = False
                        elif event.key == pygame.K_BACKSPACE:
                            self.checkpoint = self.checkpoint[:-1]
                        else:
                            ch = event.unicode
                            if ch and ch.isprintable():
                                self.checkpoint += ch
                        continue

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
                        if model_box.collidepoint(mx, my):
                            self.model_input_focused = True
                        else:
                            self.model_input_focused = False

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
                                row, slider, check, rid = row_geometry(idx)
                                # Expand slider hit target vertically.
                                slider_hit = slider.inflate(0, _s(16))
                                if check.collidepoint(mx, my) or (
                                    mx >= check.x and row.collidepoint(mx, my) and mx > slider.right + _s(40)
                                ):
                                    self.profile.must_dos[rid] = (
                                        0 if self.profile.must_dos[rid] else 1
                                    )
                                elif slider_hit.collidepoint(mx, my) or (
                                    row.collidepoint(mx, my) and slider_x0 <= mx <= slider_x1
                                ):
                                    self.dragging_slider_rid = rid
                                    set_weight_from_x(rid, mx)
                    elif (
                        self.mode == "play"
                        and self.pending_decision is not None
                        and not self.playing_segment
                    ):
                        if btn_exit.collidepoint(mx, my):
                            self.apply_action(34)
                        elif btn_idle.collidepoint(mx, my):
                            self.apply_action(35)
                        else:
                            list_rect = draw_play.list_rect  # type: ignore[attr-defined]
                            order = draw_play.order  # type: ignore[attr-defined]
                            if list_rect.collidepoint(mx, my):
                                start = int(self.decision_scroll // decision_row_h)
                                idx = start + int((my - list_rect.y) // decision_row_h)
                                if 0 <= idx < len(order):
                                    self.apply_action(order[idx])
                    elif self.mode == "results" and btn_setup.collidepoint(mx, my):
                        self.mode = "setup"
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_slider_rid = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider_rid is not None and self.mode == "setup":
                        set_weight_from_x(self.dragging_slider_rid, mx)
                elif event.type == pygame.MOUSEWHEEL:
                    if self.mode == "setup":
                        max_scroll = max(0, len(self.sorted_pref_ids) * row_h - pref_list.h)
                        self.pref_scroll = max(
                            0, min(max_scroll, self.pref_scroll - event.y * row_h)
                        )
                    elif self.mode == "play" and mx > PARK_WIDTH:
                        self.decision_scroll = max(0, self.decision_scroll - event.y * decision_row_h)
                    elif self.mode == "results":
                        self.history_scroll = max(0, self.history_scroll - event.y * _s(40))

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
