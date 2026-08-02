"""Hybrid day driver: human / heuristic / PPO focal + heuristic / PPO crowd."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Park.play.scoring import FocalScore, ParkScore, focal_from_native, park_from_native
from Park.play.session import FocalProfile, RunSettings, SessionRun


def _require_native():
    try:
        import _park_sim  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "C++ extension _park_sim is not built. Run: pip install -e ."
        ) from exc
    return _park_sim


def make_focal_config(profile: FocalProfile):
    _park_sim = _require_native()
    cfg = _park_sim.FocalPartyConfig()
    cfg.spawn_sec = int(profile.spawn_sec)
    cfg.leave_sec = int(profile.leave_sec)
    cfg.preference_weights = np.asarray(profile.preference_weights, dtype=np.float32)
    cfg.must_dos = np.asarray(profile.must_dos, dtype=np.uint8)
    return cfg


def _itinerary_from_focal(stats) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for ev in getattr(stats, "completions", []) or []:
        out.append((int(ev.sec), int(ev.ride_id)))
    return out


@dataclass
class DecisionRequest:
    now_sec: int
    obs_flat: np.ndarray
    waits: np.ndarray
    open_mask: np.ndarray
    must_do_remaining: np.ndarray
    preferences: np.ndarray
    rides_completed: int


class HybridDriver:
    """Drive a hybrid park day; pauses only when the focal guest needs a human action."""

    FOCAL_HUMAN = 0
    FOCAL_HEURISTIC = 1
    FOCAL_PPO = 2

    def __init__(
        self,
        seed: int,
        profile: FocalProfile,
        crowd_router: str,
        focal_router: str,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        enable_recording: bool = True,
        sample_interval_sec: int = 60,
        soft_human_leave: bool = True,
    ) -> None:
        if crowd_router not in ("heuristic", "ppo"):
            raise ValueError(f"Unknown crowd router: {crowd_router}")
        if focal_router not in ("human", "heuristic", "ppo"):
            raise ValueError(f"Unknown focal router: {focal_router}")
        if crowd_router == "ppo" or focal_router == "ppo":
            if not checkpoint:
                raise FileNotFoundError(
                    "PPO checkpoint path is required when crowd or focal router is 'ppo'."
                )
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"PPO checkpoint not found: {path}")

        self.seed = int(seed)
        self.profile = profile.copy()
        self.crowd_router = crowd_router
        self.focal_router = focal_router
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.device = device
        self.enable_recording = enable_recording
        self.sample_interval_sec = sample_interval_sec
        self.soft_human_leave = soft_human_leave

        self._park_sim = _require_native()
        self._policy = None
        if self.checkpoint:
            from Park.router.ppo import PPOPolicy

            self._policy = PPOPolicy(self.checkpoint, device=device)

        self.env = self._park_sim.ParkEnv(self.seed)
        focal_policy = {
            "human": self.FOCAL_HUMAN,
            "heuristic": self.FOCAL_HEURISTIC,
            "ppo": self.FOCAL_PPO,
        }[focal_router]
        self.env.reset_play(
            self.seed,
            make_focal_config(self.profile),
            crowd_auto_heuristic=(crowd_router == "heuristic"),
            focal_policy=focal_policy,
            soft_human_leave=soft_human_leave and focal_router == "human",
            enable_recording=enable_recording,
            sample_interval_sec=sample_interval_sec,
        )
        self.done = False
        self.last_metrics = None
        self.last_focal = None
        self._pending_decision: DecisionRequest | None = None

    def recording(self):
        if not self.enable_recording:
            return None
        return self.env.play_recording()

    def _decision_from_obs(self, now_sec: int, obs) -> DecisionRequest:
        flat = np.asarray(obs.flat(), dtype=np.float32)
        guest = np.asarray(obs.guest, dtype=np.float32)
        from Park.training.features import NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM

        ride = np.asarray(obs.ride, dtype=np.float32).reshape(
            NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM
        )
        return DecisionRequest(
            now_sec=int(now_sec),
            obs_flat=flat,
            waits=ride[:, 0] * 3600.0,
            open_mask=ride[:, 2] > 0.5,
            must_do_remaining=ride[:, 7] > 0.5,
            preferences=guest[:34].copy(),
            rides_completed=int(round(float(guest[39]) * 20.0)),
        )

    def advance(self) -> DecisionRequest | None:
        """Advance until human decision needed, or day completes. Returns decision or None if done."""
        if self.done:
            return None
        while True:
            step = self.env.play_advance()
            if step.done:
                self.done = True
                self.last_metrics = step.metrics
                self.last_focal = step.focal
                self._pending_decision = None
                return None
            if step.needs_ppo_batch:
                if self._policy is None:
                    raise RuntimeError("PPO batch requested but no policy loaded")
                obs = np.asarray(step.ppo_obs, dtype=np.float32)
                actions = self._policy.act_batch(obs)
                self.env.play_apply_ppo_actions([int(a) for a in actions.tolist()])
                continue
            if step.needs_human:
                req = self._decision_from_obs(step.now_sec, step.human_obs)
                self._pending_decision = req
                return req
            raise RuntimeError("play_advance returned without done/human/ppo")

    def apply_human_action(self, action: int) -> None:
        if self._pending_decision is None:
            raise RuntimeError("No pending human decision")
        self.env.play_apply_human_action(int(action))
        self._pending_decision = None

    def run_headless(self) -> SessionRun:
        """Run a non-human focal day to completion."""
        if self.focal_router == "human":
            raise ValueError("run_headless does not support human focal")
        while not self.done:
            decision = self.advance()
            if decision is not None:
                raise RuntimeError("unexpected human decision in headless run")
        return self.to_session_run(kind="ai_compare")

    def to_session_run(self, kind: str = "human", label: str = "") -> SessionRun:
        if self.last_metrics is not None:
            park = park_from_native(self.last_metrics)
        else:
            park = ParkScore()
        focal_stats = (
            self.last_focal if self.last_focal is not None else self.env.play_focal_stats()
        )
        focal = focal_from_native(focal_stats)
        return SessionRun(
            settings=RunSettings(
                seed=self.seed,
                kind=kind,
                crowd_router=self.crowd_router,
                focal_router=self.focal_router,
                checkpoint=self.checkpoint,
                label=label,
            ),
            profile=self.profile.copy(),
            park=park,
            focal=focal,
            itinerary=_itinerary_from_focal(focal_stats),
            recording=self.recording() if self.enable_recording else None,
        )


def run_heuristic_focal_day(
    seed: int,
    profile: FocalProfile,
    *,
    record: bool = False,
    sample_interval_sec: int = 60,
    label: str = "H-crowd / H-guest",
    kind: str = "ai_compare",
) -> SessionRun:
    """Fast path: all-heuristic day with custom focal guest."""
    _park_sim = _require_native()
    result = _park_sim.run_play_day(
        int(seed),
        make_focal_config(profile),
        sample_interval_sec=sample_interval_sec,
        record=record,
    )
    return SessionRun(
        settings=RunSettings(
            seed=int(seed),
            kind=kind,
            crowd_router="heuristic",
            focal_router="heuristic",
            label=label,
        ),
        profile=profile.copy(),
        park=park_from_native(result.metrics),
        focal=focal_from_native(result.focal),
        itinerary=_itinerary_from_focal(result.focal),
        recording=result.recording if record else None,
    )


def run_shadow_day(
    seed: int,
    profile: FocalProfile,
    crowd_router: str,
    focal_router: str,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    record: bool = False,
    label: str = "",
) -> SessionRun:
    """Run one AI comparison cell (no human)."""
    if focal_router == "human":
        raise ValueError("run_shadow_day does not support human focal")
    if crowd_router == "heuristic" and focal_router == "heuristic":
        return run_heuristic_focal_day(
            seed,
            profile,
            record=record,
            label=label or "H-crowd / H-guest",
        )
    cell_label = label or f"{crowd_router[0].upper()}-crowd / {focal_router[0].upper()}-guest"
    driver = HybridDriver(
        seed=seed,
        profile=profile,
        crowd_router=crowd_router,
        focal_router=focal_router,
        checkpoint=checkpoint,
        device=device,
        enable_recording=record,
        soft_human_leave=False,
    )
    run = driver.run_headless()
    run.settings.label = cell_label
    run.settings.kind = "ai_compare"
    return run
