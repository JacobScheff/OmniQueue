"""Pygame PPO-focal day watcher."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import config
from park_graph import get_park_graph
from play.scoring import format_focal_line, format_park_line
from play.session import FocalProfile
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
from watch.driver import STATE_IN_QUEUE, WatchDriver
from watch.session import WatchStore
from watch.timeline import (
    TimelineState,
    completion_counts_at,
    mark_index_for_click,
    scrub_to_frac,
)

WEIGHT_SLIDER_MAX = 250.0
# Large preference sliders (setup + mid-day edit) for precise dragging.
SETUP_PREF_ROW_H = 56
SETUP_SLIDER_W = 360
SETUP_SLIDER_H = 24
SETUP_MUST_W = 64
SETUP_MUST_H = 28
WATCH_EDIT_ROW_H = 34
WATCH_SLIDER_W = 120
WATCH_SLIDER_H = 18
WATCH_MUST_W = 36
WATCH_MUST_H = 22

# Colors
BG = (24, 28, 34)
PANEL = (32, 38, 46)
TEXT = (230, 230, 230)
MUTED = (140, 150, 160)
ACCENT = (255, 170, 40)  # golden/orange focal
FOCAL_GLOW = (255, 140, 20)
DONE_GREEN = (80, 190, 110)
MUST_AMBER = (255, 190, 70)
BTN = (60, 90, 140)
BTN2 = (70, 110, 90)
ERR = (220, 90, 90)
ROW_BG = (40, 46, 56)
TRACK = (50, 56, 66)
FILL = (90, 150, 220)
RIDE_OPEN = (70, 130, 180)
RIDE_BROKEN = (160, 60, 60)
WALK_DOT = (180, 190, 200)
MARK_FOCAL = (255, 170, 40)
MARK_CROWD = (120, 160, 255)
CHECK_ON = (80, 180, 100)


def _default_weights() -> np.ndarray:
    pops = np.array([float(r["popularity"]) for r in config.RIDES], dtype=np.float32)
    return np.clip(pops, 0.0, WEIGHT_SLIDER_MAX)


class WatchApp:
    """Setup → watch (PPO focal) → session history."""

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

        self.store = WatchStore()
        self.profile = FocalProfile(
            spawn_sec=0,
            leave_sec=config.DAY_SECONDS,
            preference_weights=_default_weights(),
            must_dos=np.zeros(config.NUM_RIDES, dtype=np.uint8),
        )
        self.mode = "setup"  # setup | watch | results
        self.status_msg = ""
        self.error_msg = ""

        self.driver: WatchDriver | None = None
        self.replay: ReplayState | None = None
        self.timeline = TimelineState()
        self.node_coords = build_node_coords()
        self.pref_scroll = 0
        self.history_scroll = 0
        self.ride_scroll = 0
        self.sorted_pref_ids = list(range(config.NUM_RIDES))
        self.model_input_focused = False
        self.dragging_slider_rid: int | None = None
        self.dragging_timeline = False
        self.queue_pause_latched = False

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

    def start_watch(self) -> None:
        self.error_msg = ""
        self.model_input_focused = False
        try:
            self._validate_ppo()
            self.status_msg = "Starting PPO-focal day…"
            self.profile.spawn_sec = 0
            self.profile.leave_sec = config.DAY_SECONDS
            self.driver = WatchDriver(
                seed=self.seed,
                profile=self.profile,
                crowd_router=self.crowd_router,
                checkpoint=self._checkpoint_or_none() or "",
                device=self.device,
                sample_interval_sec=self.sample_interval,
            )
            self.timeline = TimelineState(
                playhead_sec=0.0,
                frontier_sec=0.0,
                paused=False,
                marks_scope="focal",
            )
            self.queue_pause_latched = False
            self.mode = "watch"
            self._grow_frontier(initial=True)
        except Exception as exc:  # noqa: BLE001
            self.error_msg = str(exc)
            self.driver = None
            self.mode = "setup"

    def _rebuild_replay(self) -> None:
        assert self.driver is not None
        rec = self.driver.recording()
        if rec is None:
            return
        self.replay = ReplayState.from_recording(rec, self.node_coords)
        self.timeline.decisions = list(self.driver.decisions)
        self.timeline.frontier_sec = float(self.driver.now_sec())

    def _warm_paths(self, lo: float, hi: float) -> None:
        if self.replay is None:
            return
        park = get_park_graph()
        keys_warmed = 0
        for w in self.replay.walks:
            if float(w.end_sec) < lo or float(w.start_sec) > hi:
                continue
            park.path_arc_for_idx(
                int(w.from_idx),
                int(w.to_idx),
                variant=int(getattr(w, "path_variant", 0) or 0),
            )
            keys_warmed += 1
            if keys_warmed > 4000:
                break

    def _grow_frontier(self, *, initial: bool = False, skip: bool = False) -> None:
        assert self.driver is not None
        if self.driver.done:
            return
        if skip:
            result = self.driver.skip_to_next_focal_decision()
        elif initial:
            result = self.driver.advance_until(
                stop_on_queue=True,
                stop_on_focal_decision=True,
                min_time_advance=30,
            )
        else:
            # Grow enough recording for smooth playback at current speed.
            chunk = max(20, int(self.sim_speed * 0.35))
            result = self.driver.advance_until(
                stop_on_queue=True,
                stop_on_focal_decision=False,
                min_time_advance=chunk,
            )
        self._rebuild_replay()
        lo = max(0.0, self.timeline.playhead_sec - 60.0)
        hi = self.timeline.frontier_sec + 120.0
        self._warm_paths(lo, hi)

        if result.done:
            self._finish_day()
            return

        if result.entered_queue and not self.queue_pause_latched:
            self.timeline.paused = True
            self.queue_pause_latched = True
            self.timeline.playhead_sec = self.timeline.frontier_sec
            self.status_msg = "Paused in queue — edit prefs if you want, then Resume"
        elif skip:
            self.timeline.playhead_sec = self.timeline.frontier_sec
            if result.focal_decisions:
                self.timeline.selected_mark_idx = self.timeline.decisions.index(
                    result.focal_decisions[-1]
                )
            self.status_msg = f"Skipped to {format_clock(self.timeline.frontier_sec)}"
        else:
            if self.driver.focal_state() != STATE_IN_QUEUE:
                self.queue_pause_latched = False
            self.status_msg = (
                f"Frontier {format_clock(self.timeline.frontier_sec)}   "
                f"crowd={self.crowd_router}   focal=ppo"
            )

    def _finish_day(self) -> None:
        assert self.driver is not None
        self._rebuild_replay()
        self.timeline.frontier_sec = float(config.DAY_SECONDS)
        self.timeline.playhead_sec = min(
            self.timeline.playhead_sec, self.timeline.frontier_sec
        )
        self.timeline.paused = True
        run = self.driver.to_watch_run(label="watch")
        self.store.add(run)
        self.status_msg = "Day complete — " + format_focal_line(run.focal)
        self.mode = "results"

    def apply_pref_edits(self) -> None:
        if self.driver is None or not self.timeline.can_edit_prefs():
            return
        self.driver.update_preferences(
            self.profile.preference_weights, self.profile.must_dos
        )
        self.status_msg = "Preferences applied for future decisions"

    def toggle_pause(self) -> None:
        if self.mode != "watch":
            return
        self.timeline.paused = not self.timeline.paused
        if not self.timeline.paused and self.timeline.at_frontier():
            # Leaving a queue pause — allow re-latch later.
            if self.driver and self.driver.focal_state() != STATE_IN_QUEUE:
                self.queue_pause_latched = False
        self.status_msg = "Paused" if self.timeline.paused else "Playing"

    def skip(self) -> None:
        if self.mode != "watch" or self.driver is None or self.driver.done:
            return
        # Skip only advances the live frontier (jump playhead to end afterward).
        self.timeline.playhead_sec = self.timeline.frontier_sec
        self.timeline.paused = False
        self._grow_frontier(skip=True)
        self.timeline.paused = True

    def run(self) -> None:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("OmniQueue Watch — PPO focal")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", _s(16))
        small = pygame.font.SysFont("consolas", _s(13))
        bold = pygame.font.SysFont("consolas", _s(16), bold=True)
        title_f = pygame.font.SysFont("consolas", _s(22), bold=True)
        map_name_f = pygame.font.SysFont("consolas", _s(11))

        win_w, win_h = SCREEN_WIDTH, SCREEN_HEIGHT
        pad = _s(16)
        sidebar_x = PARK_WIDTH
        sidebar_w = SIDEBAR_WIDTH

        # Setup layout
        btn_start = pygame.Rect(pad, win_h - _s(56), _s(180), _s(40))
        btn_crowd = pygame.Rect(pad + _s(200), win_h - _s(56), _s(200), _s(40))
        btn_sort = pygame.Rect(pad + _s(420), win_h - _s(56), _s(160), _s(40))
        btn_history = pygame.Rect(pad + _s(600), win_h - _s(56), _s(160), _s(40))
        model_box = pygame.Rect(pad + _s(120), _s(56), win_w - pad * 2 - _s(120), _s(32))
        pref_list = pygame.Rect(pad, _s(110), win_w - pad * 2, win_h - _s(190))
        pref_row_h = _s(SETUP_PREF_ROW_H)
        setup_slider_w = _s(SETUP_SLIDER_W)
        setup_slider_h = _s(SETUP_SLIDER_H)
        setup_must_w = _s(SETUP_MUST_W)
        setup_must_h = _s(SETUP_MUST_H)
        setup_ctrl_w = setup_slider_w + _s(16) + setup_must_w

        # Watch controls
        btn_pause = pygame.Rect(_s(16), PARK_HEIGHT + _s(18), _s(90), _s(34))
        btn_skip = pygame.Rect(_s(116), PARK_HEIGHT + _s(18), _s(90), _s(34))
        btn_marks = pygame.Rect(_s(216), PARK_HEIGHT + _s(18), _s(160), _s(34))
        btn_apply = pygame.Rect(_s(390), PARK_HEIGHT + _s(18), _s(120), _s(34))
        btn_setup_w = pygame.Rect(_s(520), PARK_HEIGHT + _s(18), _s(100), _s(34))
        slider = pygame.Rect(_s(16), PARK_HEIGHT + _s(54), PARK_WIDTH - _s(32), _s(16))

        # Results
        btn_back_setup = pygame.Rect(pad, win_h - _s(50), _s(160), _s(36))
        hist_list = pygame.Rect(pad, _s(60), win_w - pad * 2, win_h - _s(130))

        park_surface = pygame.Surface((PARK_WIDTH, PARK_HEIGHT))
        park_surface.fill((18, 22, 28))

        def setup_track_rect(row_y: int) -> pygame.Rect:
            return pygame.Rect(
                pref_list.right - setup_ctrl_w,
                row_y + (pref_row_h - setup_slider_h) // 2,
                setup_slider_w,
                setup_slider_h,
            )

        def setup_must_rect(row_y: int) -> pygame.Rect:
            return pygame.Rect(
                pref_list.right - setup_must_w - _s(4),
                row_y + (pref_row_h - setup_must_h) // 2,
                setup_must_w,
                setup_must_h,
            )

        def draw_weight_slider(track: pygame.Rect, weight: float) -> None:
            pygame.draw.rect(screen, TRACK, track, border_radius=_s(6))
            fill_w = int(track.w * (weight / WEIGHT_SLIDER_MAX))
            if fill_w > 0:
                pygame.draw.rect(
                    screen,
                    FILL,
                    (track.x, track.y, fill_w, track.h),
                    border_radius=_s(6),
                )
            pygame.draw.rect(screen, (90, 100, 110), track, 1, border_radius=_s(6))
            knob_x = track.x + max(0, min(track.w, fill_w))
            pygame.draw.circle(screen, ACCENT, (knob_x, track.centery), track.h // 2 + _s(2))
            pygame.draw.circle(
                screen, (20, 20, 20), (knob_x, track.centery), track.h // 2 + _s(2), 1
            )

        def draw_button(rect: pygame.Rect, label: str, color) -> None:
            pygame.draw.rect(screen, color, rect, border_radius=_s(6))
            pygame.draw.rect(screen, (20, 20, 20), rect, 1, border_radius=_s(6))
            t = bold.render(label, True, TEXT)
            screen.blit(
                t,
                (rect.x + (rect.w - t.get_width()) // 2, rect.y + (rect.h - t.get_height()) // 2),
            )

        def draw_setup() -> None:
            screen.fill(PANEL)
            screen.blit(title_f.render("Watch — PPO focal guest", True, ACCENT), (pad, _s(16)))
            screen.blit(font.render("Model:", True, MUTED), (pad, _s(62)))
            pygame.draw.rect(screen, ROW_BG, model_box, border_radius=_s(4))
            if self.model_input_focused:
                pygame.draw.rect(screen, ACCENT, model_box, 2, border_radius=_s(4))
            ck = self.checkpoint or "(click to type path)"
            screen.blit(font.render(ck[-70:], True, TEXT), (model_box.x + _s(8), model_box.y + _s(6)))
            screen.blit(
                font.render(
                    f"Seed {self.seed}   day always {format_clock(0)} → {format_clock(config.DAY_SECONDS)}   "
                    f"crowd={self.crowd_router}",
                    True,
                    MUTED,
                ),
                (pad, _s(92)),
            )

            pygame.draw.rect(screen, ROW_BG, pref_list, border_radius=_s(5))
            start = int(self.pref_scroll // pref_row_h)
            visible = pref_list.h // pref_row_h + 1
            name_max_w = pref_list.w - setup_ctrl_w - _s(24)
            for i in range(start, min(len(self.sorted_pref_ids), start + visible)):
                rid = self.sorted_pref_ids[i]
                y = pref_list.y + (i - start) * pref_row_h
                name = config.RIDES[rid]["name"]
                w = float(self.profile.preference_weights[rid])
                md = bool(self.profile.must_dos[rid])
                col = MUST_AMBER if md else TEXT
                label = ("★ " if md else "  ") + name
                # Truncate by pixel width so the large slider stays fully visible.
                while label and font.size(label)[0] > name_max_w:
                    label = label[:-1]
                if label != (("★ " if md else "  ") + name) and len(label) > 1:
                    label = label[:-1] + "…"
                screen.blit(
                    font.render(label, True, col),
                    (pref_list.x + _s(10), y + (pref_row_h - font.get_height()) // 2),
                )
                track = setup_track_rect(y)
                draw_weight_slider(track, w)
                # Numeric readout for precise tuning
                val = small.render(f"{w:.0f}", True, MUTED)
                screen.blit(val, (track.x - val.get_width() - _s(8), track.centery - val.get_height() // 2))
                md_box = setup_must_rect(y)
                pygame.draw.rect(
                    screen, CHECK_ON if md else TRACK, md_box, border_radius=_s(6)
                )
                screen.blit(
                    bold.render("MUST" if md else "must", True, TEXT),
                    (
                        md_box.x + (md_box.w - bold.size("MUST" if md else "must")[0]) // 2,
                        md_box.y + (md_box.h - bold.get_height()) // 2,
                    ),
                )

            draw_button(btn_start, "Start day", BTN2)
            draw_button(
                btn_crowd,
                f"Crowd: {self.crowd_router}",
                BTN if self.crowd_router == "heuristic" else BTN2,
            )
            draw_button(btn_sort, "Sort prefs", BTN)
            draw_button(btn_history, "History", BTN)
            if self.error_msg:
                screen.blit(font.render(self.error_msg, True, ERR), (pad, win_h - _s(100)))
            if self.status_msg:
                screen.blit(font.render(self.status_msg[:90], True, MUTED), (pad, win_h - _s(88)))

        def draw_watch() -> None:
            screen.fill(PANEL)
            screen.blit(park_surface, (0, 0))
            sample = None
            if self.replay is not None:
                walks_now = active_walks_at(self.replay, self.timeline.playhead_sec)
                step = max(1, len(walks_now) // MAX_WALK_DOTS)
                for i in range(0, len(walks_now), step):
                    w = walks_now[i]
                    x, y = walk_position(self.replay, w, self.timeline.playhead_sec)
                    pygame.draw.circle(screen, WALK_DOT, _xy(x, y), max(1, _s(2)))
                sample = ride_state_at(self.replay, self.timeline.playhead_sec)

            for ride_id, ride in enumerate(config.RIDES):
                x, y = _xy(*ride["coords"])
                broken = bool(sample.broken[ride_id]) if sample is not None else False
                wait = float(sample.wait[ride_id]) if sample is not None else 0.0
                color = RIDE_BROKEN if broken else RIDE_OPEN
                pygame.draw.circle(screen, color, (x, y), _s(15))
                pygame.draw.circle(screen, (10, 10, 10), (x, y), _s(15), 1)
                label = "X" if broken or wait >= 9000 else str(int(wait / 60))
                wt = bold.render(label, True, TEXT)
                screen.blit(wt, (x - wt.get_width() // 2, y - wt.get_height() // 2))
                words = ride["name"].split()
                short = " ".join(words[:2]) + ("…" if len(words) > 2 else "")
                nt = map_name_f.render(short, True, MUTED)
                screen.blit(nt, (x - nt.get_width() // 2, y + _s(17)))

            # Golden/orange focal guest
            if self.replay is not None:
                g = party_state_at(self.replay, 0, self.timeline.playhead_sec)
                if g and g.get("pos"):
                    gx, gy = _xy(*g["pos"])
                    if "dest" in g and g["dest"] is not None:
                        dx, dy = _xy(*g["dest"])
                        pygame.draw.line(screen, FOCAL_GLOW, (gx, gy), (dx, dy), max(1, _s(2)))
                    pulse = abs(math.sin(pygame.time.get_ticks() / 220.0)) * 6
                    pygame.draw.circle(screen, FOCAL_GLOW, (gx, gy), _s(13 + pulse))
                    pygame.draw.circle(screen, ACCENT, (gx, gy), _s(9))
                    pygame.draw.circle(screen, (20, 20, 20), (gx, gy), _s(13 + pulse), max(1, _s(2)))
                    pygame.draw.circle(screen, (255, 255, 255), (gx, gy), _s(3))
                    you = bold.render("FOCAL", True, (20, 20, 20), ACCENT)
                    screen.blit(you, (gx + _s(14), gy - _s(12)))

            # Control bar
            pygame.draw.rect(screen, PANEL, (0, PARK_HEIGHT, PARK_WIDTH, CONTROL_HEIGHT))
            draw_button(btn_pause, "Resume" if self.timeline.paused else "Pause", BTN)
            draw_button(btn_skip, "Skip", BTN2)
            marks_label = "Marks: Focal" if self.timeline.marks_scope == "focal" else "Marks: All"
            draw_button(btn_marks, marks_label, BTN)
            can_edit = self.timeline.can_edit_prefs()
            draw_button(btn_apply, "Apply prefs", BTN2 if can_edit else (70, 70, 75))
            draw_button(btn_setup_w, "Setup", BTN)

            screen.blit(
                title_f.render(format_clock(self.timeline.playhead_sec), True, TEXT),
                (_s(640), PARK_HEIGHT + _s(20)),
            )
            screen.blit(
                small.render(
                    f"{self.sim_speed:.0f}x  frontier {format_clock(self.timeline.frontier_sec)}  "
                    f"{self.status_msg}",
                    True,
                    MUTED,
                ),
                (_s(640), PARK_HEIGHT + _s(48)),
            )

            # Timeline slider + marks
            pygame.draw.rect(screen, TRACK, slider, border_radius=_s(4))
            frac = 0.0
            if config.DAY_SECONDS > 0:
                frac = min(1.0, self.timeline.frontier_sec / config.DAY_SECONDS)
            if frac > 0:
                pygame.draw.rect(
                    screen,
                    (70, 80, 95),
                    (slider.x, slider.y, int(slider.w * frac), slider.h),
                    border_radius=_s(4),
                )
            play_frac = (
                self.timeline.playhead_sec / config.DAY_SECONDS if config.DAY_SECONDS else 0.0
            )
            hx = slider.x + int(slider.w * play_frac)
            pygame.draw.circle(screen, ACCENT, (hx, slider.centery), _s(7))

            for idx, mark in self.timeline.visible_marks():
                if mark.sec > self.timeline.frontier_sec:
                    continue
                mx = slider.x + int(round((mark.sec / config.DAY_SECONDS) * slider.w))
                col = MARK_FOCAL if mark.scope == "focal" else MARK_CROWD
                if idx == self.timeline.selected_mark_idx:
                    pygame.draw.line(
                        screen, col, (mx, slider.y - _s(8)), (mx, slider.bottom + _s(8)), _s(3)
                    )
                else:
                    pygame.draw.line(
                        screen, col, (mx, slider.y - _s(4)), (mx, slider.bottom + _s(4)), 1
                    )

            # Sidebar: ride list with counts + colors, prefs when editable, probs
            pygame.draw.rect(screen, PANEL, (sidebar_x, 0, sidebar_w, win_h))
            screen.blit(title_f.render("Focal itinerary", True, ACCENT), (sidebar_x + _s(12), _s(12)))
            screen.blit(
                small.render(
                    f"seed {self.seed}  crowd={self.crowd_router}",
                    True,
                    MUTED,
                ),
                (sidebar_x + _s(12), _s(42)),
            )

            completions = []
            if self.replay is not None:
                completions = list(self.replay.ride_completions)
            counts = completion_counts_at(
                completions, 0, self.timeline.playhead_sec, config.NUM_RIDES
            )
            order = sorted(
                range(config.NUM_RIDES),
                key=lambda i: float(self.profile.preference_weights[i]),
                reverse=True,
            )
            list_top = _s(64)
            list_h = _s(220)
            row_h = _s(22)
            list_rect = pygame.Rect(sidebar_x + _s(8), list_top, sidebar_w - _s(16), list_h)
            pygame.draw.rect(screen, ROW_BG, list_rect, border_radius=_s(4))
            start = int(self.ride_scroll // row_h)
            visible = list_rect.h // row_h + 1
            for i in range(start, min(len(order), start + visible)):
                rid = order[i]
                y = list_rect.y + (i - start) * row_h
                n = counts[rid]
                md = bool(self.profile.must_dos[rid])
                if n > 0:
                    col = DONE_GREEN
                elif md:
                    col = MUST_AMBER
                else:
                    col = TEXT
                name = config.RIDES[rid]["name"]
                short = name if len(name) <= 28 else name[:27] + "…"
                prefix = "★ " if md else "  "
                screen.blit(
                    small.render(f"{prefix}{short}", True, col),
                    (list_rect.x + _s(6), y + _s(3)),
                )
                screen.blit(
                    small.render(f"×{n}", True, DONE_GREEN if n else MUTED),
                    (list_rect.right - _s(36), y + _s(3)),
                )

            y_cursor = list_rect.bottom + _s(10)
            if can_edit:
                screen.blit(
                    bold.render("Edit prefs (at frontier)", True, ACCENT),
                    (sidebar_x + _s(12), y_cursor),
                )
                y_cursor += _s(24)
                edit_h = _s(220)
                edit_rect = pygame.Rect(
                    sidebar_x + _s(8), y_cursor, sidebar_w - _s(16), edit_h
                )
                pygame.draw.rect(screen, ROW_BG, edit_rect, border_radius=_s(4))
                erow = _s(WATCH_EDIT_ROW_H)
                e_slider_w = _s(WATCH_SLIDER_W)
                e_slider_h = _s(WATCH_SLIDER_H)
                e_must_w = _s(WATCH_MUST_W)
                e_must_h = _s(WATCH_MUST_H)
                estart = int(self.pref_scroll // erow)
                evis = edit_rect.h // erow
                for i in range(estart, min(len(self.sorted_pref_ids), estart + evis)):
                    rid = self.sorted_pref_ids[i]
                    ey = edit_rect.y + (i - estart) * erow
                    md = bool(self.profile.must_dos[rid])
                    w = float(self.profile.preference_weights[rid])
                    screen.blit(
                        small.render(config.RIDES[rid]["name"][:16], True, MUST_AMBER if md else TEXT),
                        (edit_rect.x + _s(4), ey + (erow - small.get_height()) // 2),
                    )
                    track = pygame.Rect(
                        edit_rect.right - e_must_w - _s(10) - e_slider_w,
                        ey + (erow - e_slider_h) // 2,
                        e_slider_w,
                        e_slider_h,
                    )
                    draw_weight_slider(track, w)
                    mb = pygame.Rect(
                        edit_rect.right - e_must_w - _s(4),
                        ey + (erow - e_must_h) // 2,
                        e_must_w,
                        e_must_h,
                    )
                    pygame.draw.rect(screen, CHECK_ON if md else TRACK, mb, border_radius=_s(4))
                y_cursor = edit_rect.bottom + _s(8)
                draw_watch.edit_rect = edit_rect  # type: ignore[attr-defined]
                draw_watch.edit_row_h = erow  # type: ignore[attr-defined]
            else:
                screen.blit(
                    small.render("Scrubbing past — prefs locked", True, MUTED),
                    (sidebar_x + _s(12), y_cursor),
                )
                y_cursor += _s(28)
                draw_watch.edit_rect = None  # type: ignore[attr-defined]
                draw_watch.edit_row_h = _s(WATCH_EDIT_ROW_H)  # type: ignore[attr-defined]

            # Probability panel for selected mark
            screen.blit(bold.render("Decision probs", True, TEXT), (sidebar_x + _s(12), y_cursor))
            y_cursor += _s(22)
            mark = None
            if self.timeline.selected_mark_idx is not None and self.timeline.selected_mark_idx < len(
                self.timeline.decisions
            ):
                mark = self.timeline.decisions[self.timeline.selected_mark_idx]
            if mark is None:
                screen.blit(
                    small.render("Click a timeline mark", True, MUTED),
                    (sidebar_x + _s(12), y_cursor),
                )
            else:
                screen.blit(
                    small.render(
                        f"{mark.scope} @{format_clock(mark.sec)} act={mark.action}",
                        True,
                        MUTED,
                    ),
                    (sidebar_x + _s(12), y_cursor),
                )
                y_cursor += _s(18)
                # Top-8 actions by probability
                order_p = np.argsort(-mark.probs)[:8]
                for a in order_p:
                    p = float(mark.probs[a])
                    if p < 1e-4:
                        continue
                    if a < config.NUM_RIDES:
                        label = config.RIDES[int(a)]["name"][:20]
                    elif a == config.NUM_RIDES:
                        label = "Exit"
                    else:
                        label = "Wander"
                    bar = pygame.Rect(sidebar_x + _s(12), y_cursor + _s(2), sidebar_w - _s(24), _s(14))
                    pygame.draw.rect(screen, TRACK, bar, border_radius=_s(3))
                    bw = int(bar.w * min(1.0, p))
                    col = ACCENT if int(a) == int(mark.action) else FILL
                    if bw:
                        pygame.draw.rect(screen, col, (bar.x, bar.y, bw, bar.h), border_radius=_s(3))
                    screen.blit(
                        small.render(f"{label}  {p * 100:.1f}%", True, TEXT),
                        (bar.x + _s(4), bar.y - _s(1)),
                    )
                    y_cursor += _s(18)

        def draw_results() -> None:
            screen.fill(PANEL)
            screen.blit(title_f.render("Watch runs (this launch)", True, TEXT), (pad, _s(16)))
            draw_button(btn_back_setup, "Back to setup", BTN)
            pygame.draw.rect(screen, ROW_BG, hist_list, border_radius=_s(5))
            y = hist_list.y + _s(8) - self.history_scroll
            for run in self.store.runs:
                lines = [
                    run.settings.summary(),
                    "  park:  " + format_park_line(run.park),
                    "  focal: " + format_focal_line(run.focal),
                    f"  decisions={len(run.decisions)}",
                    "",
                ]
                for line in lines:
                    if hist_list.y <= y <= hist_list.bottom - _s(18):
                        screen.blit(font.render(line, True, TEXT), (hist_list.x + _s(10), y))
                    y += _s(20)
            if self.status_msg:
                screen.blit(font.render(self.status_msg[:110], True, ACCENT), (pad, win_h - _s(90)))

        def hit_pref_controls(mx: int, my: int, in_setup: bool) -> None:
            if in_setup:
                if not pref_list.collidepoint(mx, my):
                    return
                start = int(self.pref_scroll // pref_row_h)
                i = start + (my - pref_list.y) // pref_row_h
                if i < 0 or i >= len(self.sorted_pref_ids):
                    return
                rid = self.sorted_pref_ids[i]
                y = pref_list.y + (i - start) * pref_row_h
                md_box = setup_must_rect(y)
                track = setup_track_rect(y)
                # Expand vertical hit box slightly around the track/knob.
                hit_track = track.inflate(0, _s(10))
                if md_box.collidepoint(mx, my):
                    self.profile.must_dos[rid] = 0 if self.profile.must_dos[rid] else 1
                elif hit_track.collidepoint(mx, my):
                    self.dragging_slider_rid = rid
                    t = (mx - track.x) / max(1, track.w)
                    self.profile.preference_weights[rid] = float(
                        np.clip(t * WEIGHT_SLIDER_MAX, 0.0, WEIGHT_SLIDER_MAX)
                    )
            else:
                if not self.timeline.can_edit_prefs():
                    return
                edit_rect = getattr(draw_watch, "edit_rect", None)
                erow = int(getattr(draw_watch, "edit_row_h", _s(WATCH_EDIT_ROW_H)))
                if edit_rect is None or not edit_rect.collidepoint(mx, my):
                    return
                e_slider_w = _s(WATCH_SLIDER_W)
                e_slider_h = _s(WATCH_SLIDER_H)
                e_must_w = _s(WATCH_MUST_W)
                e_must_h = _s(WATCH_MUST_H)
                estart = int(self.pref_scroll // erow)
                i = estart + (my - edit_rect.y) // erow
                if i < 0 or i >= len(self.sorted_pref_ids):
                    return
                rid = self.sorted_pref_ids[i]
                ey = edit_rect.y + (i - estart) * erow
                track = pygame.Rect(
                    edit_rect.right - e_must_w - _s(10) - e_slider_w,
                    ey + (erow - e_slider_h) // 2,
                    e_slider_w,
                    e_slider_h,
                )
                mb = pygame.Rect(
                    edit_rect.right - e_must_w - _s(4),
                    ey + (erow - e_must_h) // 2,
                    e_must_w,
                    e_must_h,
                )
                if mb.collidepoint(mx, my):
                    self.profile.must_dos[rid] = 0 if self.profile.must_dos[rid] else 1
                elif track.inflate(0, _s(8)).collidepoint(mx, my):
                    self.dragging_slider_rid = rid
                    t = (mx - track.x) / max(1, track.w)
                    self.profile.preference_weights[rid] = float(
                        np.clip(t * WEIGHT_SLIDER_MAX, 0.0, WEIGHT_SLIDER_MAX)
                    )

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
                        elif event.unicode and event.unicode.isprintable():
                            self.checkpoint += event.unicode
                    elif event.key == pygame.K_SPACE and self.mode == "watch":
                        self.toggle_pause()
                    elif event.key == pygame.K_s and self.mode == "watch":
                        self.skip()
                    elif event.key == pygame.K_ESCAPE:
                        if self.mode == "watch":
                            self.mode = "setup"
                            self.driver = None
                        elif self.mode == "results":
                            self.mode = "setup"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.mode == "setup":
                        if model_box.collidepoint(mx, my):
                            self.model_input_focused = True
                        else:
                            self.model_input_focused = False
                        if btn_start.collidepoint(mx, my):
                            self.start_watch()
                        elif btn_crowd.collidepoint(mx, my):
                            self.crowd_router = (
                                "ppo" if self.crowd_router == "heuristic" else "heuristic"
                            )
                        elif btn_sort.collidepoint(mx, my):
                            self.sort_preferences()
                        elif btn_history.collidepoint(mx, my):
                            self.mode = "results"
                        else:
                            hit_pref_controls(mx, my, in_setup=True)
                    elif self.mode == "watch":
                        if btn_pause.collidepoint(mx, my):
                            self.toggle_pause()
                        elif btn_skip.collidepoint(mx, my):
                            self.skip()
                        elif btn_marks.collidepoint(mx, my):
                            self.timeline.marks_scope = (
                                "all" if self.timeline.marks_scope == "focal" else "focal"
                            )
                        elif btn_apply.collidepoint(mx, my):
                            self.apply_pref_edits()
                        elif btn_setup_w.collidepoint(mx, my):
                            self.mode = "setup"
                            self.driver = None
                        elif slider.collidepoint(mx, my) or (
                            abs(my - slider.centery) <= _s(14)
                            and slider.x <= mx <= slider.right
                        ):
                            # Prefer mark hit, else scrub
                            hit = mark_index_for_click(
                                self.timeline.visible_marks(),
                                mx,
                                slider.x,
                                slider.w,
                                float(config.DAY_SECONDS),
                                hit_px=_s(10),
                            )
                            if hit is not None:
                                self.timeline.selected_mark_idx = hit
                                self.timeline.paused = True
                                self.timeline.playhead_sec = float(
                                    self.timeline.decisions[hit].sec
                                )
                            else:
                                self.dragging_timeline = True
                                frac = (mx - slider.x) / max(1, slider.w)
                                self.timeline.playhead_sec = scrub_to_frac(
                                    frac, float(config.DAY_SECONDS), self.timeline.frontier_sec
                                )
                                self.timeline.paused = True
                        else:
                            hit_pref_controls(mx, my, in_setup=False)
                    elif self.mode == "results":
                        if btn_back_setup.collidepoint(mx, my):
                            self.mode = "setup"
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_slider_rid = None
                    self.dragging_timeline = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider_rid is not None:
                        rid = self.dragging_slider_rid
                        if self.mode == "setup":
                            track = setup_track_rect(0)
                            t = (mx - track.x) / max(1, track.w)
                        else:
                            edit_rect = getattr(draw_watch, "edit_rect", None)
                            e_slider_w = _s(WATCH_SLIDER_W)
                            e_must_w = _s(WATCH_MUST_W)
                            if edit_rect is not None:
                                track_x = edit_rect.right - e_must_w - _s(10) - e_slider_w
                                t = (mx - track_x) / max(1, e_slider_w)
                            else:
                                t = 0.0
                        self.profile.preference_weights[rid] = float(
                            np.clip(t * WEIGHT_SLIDER_MAX, 0.0, WEIGHT_SLIDER_MAX)
                        )
                    if self.dragging_timeline and self.mode == "watch":
                        frac = (mx - slider.x) / max(1, slider.w)
                        self.timeline.playhead_sec = scrub_to_frac(
                            frac, float(config.DAY_SECONDS), self.timeline.frontier_sec
                        )
                elif event.type == pygame.MOUSEWHEEL:
                    if self.mode == "setup" and pref_list.collidepoint(mx, my):
                        self.pref_scroll = max(0, self.pref_scroll - event.y * pref_row_h)
                    elif self.mode == "watch" and mx >= sidebar_x:
                        self.ride_scroll = max(0, self.ride_scroll - event.y * _s(22))
                        self.pref_scroll = max(0, self.pref_scroll - event.y * _s(26))
                    elif self.mode == "results" and hist_list.collidepoint(mx, my):
                        self.history_scroll = max(0, self.history_scroll - event.y * _s(20))

            # Playback / frontier growth
            if self.mode == "watch" and self.driver is not None and not self.driver.done:
                if not self.timeline.paused:
                    if self.timeline.at_frontier():
                        self._grow_frontier()
                    else:
                        self.timeline.playhead_sec = min(
                            self.timeline.frontier_sec,
                            self.timeline.playhead_sec + self.sim_speed * dt,
                        )

            if self.mode == "setup":
                draw_setup()
            elif self.mode == "watch":
                draw_watch()
            else:
                draw_results()

            pygame.display.flip()

        pygame.quit()


def run_watch_app(**kwargs) -> None:
    WatchApp(**kwargs).run()
