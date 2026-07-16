"""Watch driver: PPO focal + heuristic/PPO crowd with decision logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from play.driver import make_focal_config
from play.scoring import focal_from_native, park_from_native
from play.session import FocalProfile
from watch.session import DecisionMark, WatchRun, WatchSettings
from watch.timeline import insert_sorted_by_sec

# PartyState ints from native/include/park_sim.hpp
STATE_WALKING = 1
STATE_IN_QUEUE = 2
STATE_ON_RIDE = 4
STATE_EVACUATING = 8
STATE_EXITED = 16


def _require_native():
    try:
        import _park_sim  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "C++ extension _park_sim is not built. Run: pip install -e ."
        ) from exc
    return _park_sim


def _require_watch_apis(park_sim) -> None:
    """Fail fast if the installed extension predates watch-mode bindings."""
    required = (
        "play_focal_state",
        "play_focal_ride_history",
        "play_update_focal_preferences",
    )
    missing = [name for name in required if not hasattr(park_sim.ParkEnv, name)]
    if missing:
        raise RuntimeError(
            "Installed _park_sim is missing watch APIs "
            f"({', '.join(missing)}). Rebuild the extension: pip install -e ."
        )


@dataclass
class WatchStepResult:
    done: bool
    now_sec: int
    focal_state: int
    entered_queue: bool
    focal_decisions: list[DecisionMark] = field(default_factory=list)
    all_decisions: list[DecisionMark] = field(default_factory=list)


class WatchDriver:
    """Drive a PPO-focal hybrid day one PPO batch at a time."""

    FOCAL_PPO = 2

    def __init__(
        self,
        seed: int,
        profile: FocalProfile,
        crowd_router: str,
        checkpoint: str | Path,
        device: str = "cpu",
        sample_interval_sec: int = 60,
    ) -> None:
        if crowd_router not in ("heuristic", "ppo"):
            raise ValueError(f"Unknown crowd router: {crowd_router}")
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {path}")

        self.seed = int(seed)
        self.profile = profile.copy()
        # Always open → close for watch mode.
        self.profile.spawn_sec = 0
        self.profile.leave_sec = int(config.DAY_SECONDS)
        self.crowd_router = crowd_router
        self.checkpoint = str(path)
        self.device = device
        self.sample_interval_sec = int(sample_interval_sec)

        self._park_sim = _require_native()
        _require_watch_apis(self._park_sim)
        from router.ppo import PPOPolicy

        self._policy = PPOPolicy(self.checkpoint, device=device)
        self.env = self._park_sim.ParkEnv(self.seed)
        self.env.reset_play(
            self.seed,
            make_focal_config(self.profile),
            crowd_auto_heuristic=(crowd_router == "heuristic"),
            focal_policy=self.FOCAL_PPO,
            soft_human_leave=False,
            enable_recording=True,
            sample_interval_sec=self.sample_interval_sec,
        )
        self.done = False
        self.last_metrics = None
        self.last_focal = None
        self.decisions: list[DecisionMark] = []
        self._prev_focal_state = int(self.env.play_focal_state())
        self.focal_party_id = int(self.env.play_focal_party_id())

    def recording(self):
        return self.env.play_recording()

    def now_sec(self) -> int:
        return int(self.env.play_now_sec())

    def focal_state(self) -> int:
        return int(self.env.play_focal_state())

    def focal_ride_history(self) -> np.ndarray:
        return np.asarray(self.env.play_focal_ride_history(), dtype=np.int16)

    def update_preferences(self, preference_weights: np.ndarray, must_dos: np.ndarray) -> None:
        """Apply mid-day pref edits to the live focal guest."""
        self.profile.preference_weights = np.asarray(preference_weights, dtype=np.float32).copy()
        self.profile.must_dos = np.asarray(must_dos, dtype=np.uint8).copy()
        self.profile.spawn_sec = 0
        self.profile.leave_sec = int(config.DAY_SECONDS)
        self.env.play_update_focal_preferences(make_focal_config(self.profile))

    def _waits_from_obs_row(self, obs_row: np.ndarray) -> np.ndarray:
        from training.features import GUEST_FEAT_DIM, RIDE_DYNAMIC_FEAT_DIM

        ride = obs_row[
            GUEST_FEAT_DIM : GUEST_FEAT_DIM + config.NUM_RIDES * RIDE_DYNAMIC_FEAT_DIM
        ].reshape(config.NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM)
        return ride[:, 0] * 3600.0

    def advance_batch(self) -> WatchStepResult:
        """Resolve one play_advance PPO batch (or finish)."""
        if self.done:
            return WatchStepResult(
                done=True,
                now_sec=self.now_sec(),
                focal_state=self.focal_state(),
                entered_queue=False,
            )

        step = self.env.play_advance()
        if step.done:
            self.done = True
            self.last_metrics = step.metrics
            self.last_focal = step.focal
            state = self.focal_state()
            return WatchStepResult(
                done=True,
                now_sec=int(step.now_sec),
                focal_state=state,
                entered_queue=False,
            )

        if not step.needs_ppo_batch:
            raise RuntimeError("watch expected PPO batch (focal is always PPO)")

        obs = np.asarray(step.ppo_obs, dtype=np.float32)
        party_ids = np.asarray(step.ppo_party_ids, dtype=np.int32)
        actions, probs = self._policy.act_batch_with_probs(obs)
        if actions.ndim == 0:
            actions = np.asarray([int(actions)], dtype=np.int64)
            probs = probs.reshape(1, -1)

        focal_marks: list[DecisionMark] = []
        all_marks: list[DecisionMark] = []
        now = int(step.now_sec)
        for i, pid in enumerate(party_ids.tolist()):
            row = obs[i] if obs.ndim == 2 else obs
            mark = DecisionMark(
                sec=now,
                scope="focal" if int(pid) == self.focal_party_id else "crowd",
                party_id=int(pid),
                action=int(actions[i]),
                probs=np.asarray(probs[i], dtype=np.float32).copy(),
                wait_snapshot=self._waits_from_obs_row(np.asarray(row, dtype=np.float32)),
            )
            insert_sorted_by_sec(self.decisions, mark)
            all_marks.append(mark)
            if mark.scope == "focal":
                focal_marks.append(mark)

        self.env.play_apply_ppo_actions([int(a) for a in actions.tolist()])
        state = self.focal_state()
        entered = self._prev_focal_state != STATE_IN_QUEUE and state == STATE_IN_QUEUE
        self._prev_focal_state = state
        return WatchStepResult(
            done=False,
            now_sec=self.now_sec(),
            focal_state=state,
            entered_queue=entered,
            focal_decisions=focal_marks,
            all_decisions=all_marks,
        )

    def advance_until(
        self,
        *,
        stop_on_queue: bool = True,
        stop_on_focal_decision: bool = False,
        max_batches: int = 10_000,
        min_time_advance: int = 0,
    ) -> WatchStepResult:
        """Advance multiple PPO batches until a stop condition."""
        start = self.now_sec()
        last = WatchStepResult(
            done=self.done,
            now_sec=start,
            focal_state=self.focal_state(),
            entered_queue=False,
        )
        for _ in range(max_batches):
            last = self.advance_batch()
            if last.done:
                return last
            if stop_on_queue and last.entered_queue:
                return last
            if stop_on_focal_decision and last.focal_decisions:
                return last
            if min_time_advance > 0 and last.now_sec - start >= min_time_advance:
                return last
        return last

    def skip_to_next_focal_decision(self) -> WatchStepResult:
        """Advance until the next focal PPO decision (or day end / queue entry)."""
        # If already needing to decide at frontier with no pending, pull batches
        # until a focal mark appears. Also stop if we enter a queue so prefs can edit.
        return self.advance_until(stop_on_queue=True, stop_on_focal_decision=True)

    def to_watch_run(self, label: str = "") -> WatchRun:
        from play.scoring import ParkScore

        if self.last_metrics is not None:
            park = park_from_native(self.last_metrics)
        else:
            park = ParkScore()
        focal_stats = (
            self.last_focal if self.last_focal is not None else self.env.play_focal_stats()
        )
        itinerary = [(int(ev.sec), int(ev.ride_id)) for ev in (focal_stats.completions or [])]
        return WatchRun(
            settings=WatchSettings(
                seed=self.seed,
                crowd_router=self.crowd_router,
                checkpoint=self.checkpoint,
                label=label,
            ),
            profile=self.profile.copy(),
            park=park,
            focal=focal_from_native(focal_stats),
            itinerary=itinerary,
            decisions=list(self.decisions),
            recording=self.recording(),
        )
