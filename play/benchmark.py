"""Four-cell AI comparison and multi-day heuristic-vs-PPO benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from play.driver import run_shadow_day
from play.scoring import FocalScore, ParkScore, format_focal_line, format_park_line, park_from_native
from play.session import FocalProfile, RunSettings, SessionRun, SessionStore
from router.ppo import PPOPolicy
from simulator import run_day


COMPARE_CELLS: list[tuple[str, str, str]] = [
    ("heuristic", "heuristic", "H-crowd / H-guest"),
    ("heuristic", "ppo", "H-crowd / P-guest"),
    ("ppo", "heuristic", "P-crowd / H-guest"),
    ("ppo", "ppo", "P-crowd / P-guest"),
]


def run_ai_compare(
    seed: int,
    profile: FocalProfile,
    checkpoint: str | Path | None,
    store: SessionStore,
    device: str = "cpu",
) -> list[SessionRun]:
    """Run all 4 crowd×focal AI cells for the same seed/profile; store in session."""
    needs_ppo = any(c == "ppo" or f == "ppo" for c, f, _ in COMPARE_CELLS)
    if needs_ppo and not checkpoint:
        raise FileNotFoundError(
            "PPO checkpoint is required for AI compare cells that use PPO."
        )

    results: list[SessionRun] = []
    for crowd, focal, label in COMPARE_CELLS:
        ckpt = checkpoint if (crowd == "ppo" or focal == "ppo") else None
        run = run_shadow_day(
            seed=seed,
            profile=profile,
            crowd_router=crowd,
            focal_router=focal,
            checkpoint=ckpt,
            device=device,
            record=False,
            label=label,
        )
        store.add(run)
        results.append(run)
    return results


def _run_all_ppo_day(seed: int, checkpoint: str | Path, device: str = "cpu") -> ParkScore:
    """Full-park PPO day (every party) via ParkEnv batched inference."""
    import _park_sim  # type: ignore[import-not-found]

    policy = PPOPolicy(checkpoint, device=device)
    env = _park_sim.ParkEnv(int(seed))
    env.reset(int(seed))
    pending: list[int] = []
    batch_size = 256
    while True:
        batch = env.exchange_batch(pending, batch_size)
        pending = []
        if batch.episode_done:
            return park_from_native(batch.metrics)
        if batch.n_obs <= 0:
            continue
        obs = np.asarray(batch.obs, dtype=np.float32)
        actions = policy.act_batch(obs)
        pending = [int(a) for a in actions.tolist()]


@dataclass
class BenchmarkArmStats:
    router: str
    n_days: int = 0
    rides_per_party: list[float] = field(default_factory=list)
    mean_wait: list[float] = field(default_factory=list)
    wait_variance: list[float] = field(default_factory=list)

    def add(self, park: ParkScore) -> None:
        self.n_days += 1
        self.rides_per_party.append(park.rides_per_party)
        self.mean_wait.append(park.mean_wait)
        self.wait_variance.append(park.wait_variance)

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def summary_lines(self) -> list[str]:
        return [
            f"{self.router}: n={self.n_days}",
            f"  rides/party avg={self._mean(self.rides_per_party):.3f}",
            f"  mean wait avg={self._mean(self.mean_wait) / 60.0:.2f} min",
            f"  wait var avg={self._mean(self.wait_variance) / 1e6:.4f} e6",
        ]


@dataclass
class BenchmarkResult:
    seed_start: int
    n_days: int
    checkpoint: str | None
    heuristic: BenchmarkArmStats = field(default_factory=lambda: BenchmarkArmStats("heuristic"))
    ppo: BenchmarkArmStats = field(default_factory=lambda: BenchmarkArmStats("ppo"))
    runs: list[SessionRun] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Benchmark seeds {self.seed_start}..{self.seed_start + self.n_days - 1} "
            f"(checkpoint={self.checkpoint})",
            "",
        ]
        lines.extend(self.heuristic.summary_lines())
        lines.append("")
        lines.extend(self.ppo.summary_lines())
        return lines


def run_park_benchmark(
    seed_start: int,
    n_days: int,
    checkpoint: str | Path,
    store: SessionStore,
    device: str = "cpu",
) -> BenchmarkResult:
    """Compare all-heuristic vs all-PPO across N full park days (no human)."""
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {ckpt}")

    profile = FocalProfile(
        spawn_sec=0,
        leave_sec=14 * 3600,
        preference_weights=np.ones(34, dtype=np.float32),
        must_dos=np.zeros(34, dtype=np.uint8),
    )
    result = BenchmarkResult(
        seed_start=int(seed_start),
        n_days=int(n_days),
        checkpoint=str(ckpt),
    )

    for i in range(int(n_days)):
        seed = int(seed_start) + i

        h_metrics = run_day(seed=seed, router="heuristic")
        h_park = ParkScore(
            total_parties=h_metrics.total_parties,
            total_guests=h_metrics.total_guests,
            rides_completed=h_metrics.rides_completed,
            rides_per_party=h_metrics.rides_per_party,
            mean_wait=h_metrics.avg_mean_wait,
            wait_variance=h_metrics.avg_wait_variance,
            breakdown_count=h_metrics.breakdown_count,
            wall_time_sec=h_metrics.wall_time_sec,
        )
        h_run = SessionRun(
            settings=RunSettings(
                seed=seed,
                kind="benchmark",
                crowd_router="heuristic",
                focal_router="heuristic",
                label=f"benchmark H seed={seed}",
            ),
            profile=profile.copy(),
            park=h_park,
            focal=FocalScore(),
        )
        store.add(h_run)
        result.runs.append(h_run)
        result.heuristic.add(h_park)

        p_park = _run_all_ppo_day(seed, ckpt, device=device)
        p_run = SessionRun(
            settings=RunSettings(
                seed=seed,
                kind="benchmark",
                crowd_router="ppo",
                focal_router="ppo",
                checkpoint=str(ckpt),
                label=f"benchmark P seed={seed}",
            ),
            profile=profile.copy(),
            park=p_park,
            focal=FocalScore(),
        )
        store.add(p_run)
        result.runs.append(p_run)
        result.ppo.add(p_park)

    return result


def format_compare_table(runs: list[SessionRun]) -> list[str]:
    lines = ["AI compare (same seed / prefs / times):", ""]
    for run in runs:
        lines.append(run.settings.summary())
        lines.append("  park:  " + format_park_line(run.park))
        lines.append("  focal: " + format_focal_line(run.focal))
        lines.append("")
    return lines
