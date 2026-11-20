#!/usr/bin/env python3
"""Compare OmniQueue RL and replanning DP on a complete recorded park day.

    python Park/compare.py
    python Park/compare.py --day 2026-11-18 --headless
    python Park/compare.py --replay Park/comparisons/2026-11-18-seed42.json
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Park import config


def make_profile(args):
    rng = random.Random(args.seed)
    payload = json.loads(args.profile.read_text(encoding="utf-8")) if args.profile else {}
    prefs = payload.get("preferences", [r["popularity"]*rng.uniform(0.75, 1.25) for r in config.RIDES])
    must_dos = sorted(set(args.must_do if args.must_do is not None else payload.get("must_dos", [])))
    speed = args.walking_speed if args.walking_speed is not None else payload.get("speed", config.BASE_WALKING_SPEED)
    if (len(prefs) != config.NUM_RIDES or any(not isinstance(p, (float, int)) or not math.isfinite(p) or p < 0 for p in prefs)
            or any(type(j) is not int or not 0 <= j < config.NUM_RIDES for j in must_dos)
            or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0):
        raise ValueError("Profile needs 34 finite nonnegative preferences, valid must-do IDs and positive walking speed")
    # Match Play's popularity/noise and explicit must-do preference boost.
    prefs = [p*(config.MUST_DO_PREF_BOOST if j in must_dos else 1) for j, p in enumerate(prefs)]
    if any(not math.isfinite(p) for p in prefs) or not math.isfinite(sum(prefs)):
        raise ValueError("Preference weights are too large")
    if not sum(prefs):
        prefs = [1.0]*config.NUM_RIDES
    mass = sum(prefs)
    return dict(preferences=[p/mass for p in prefs], must_dos=must_dos, speed=speed)


def save_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", help="Park-local date; default yesterday, falling back to latest complete cached day")
    parser.add_argument("--dataset", type=Path, default=ROOT/"Wait Time Predictions/data/waits.json.gz")
    parser.add_argument("--model", "--checkpoint", type=Path,
                        help="RL checkpoint; default first compatible ppo_final.pt or deployed companion v2.pt")
    parser.add_argument("--device", default="cpu", help="RL torch device")
    venv = ROOT/"Wait Time Predictions/.venv/Scripts/python.exe"
    if not venv.exists():
        venv = ROOT/"Wait Time Predictions/.venv/bin/python"
    parser.add_argument("--forecast-python", type=Path, default=venv if venv.exists() else Path(sys.executable))
    parser.add_argument("--forecast-checkpoint", type=Path, default=ROOT/"Wait Time Predictions/checkpoints/waits.pt")
    parser.add_argument("--forecast-device", default="cpu")
    parser.add_argument("--samples", type=int, default=100, help="Wait-model forecast samples per origin")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads per model process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", type=Path, help="JSON: preferences (34 raw weights), must_dos (IDs), speed (m/s)")
    parser.add_argument("--must-do", type=int, action="append", help="Required ride ID, repeatable; overrides profile must_dos")
    parser.add_argument("--walking-speed", type=float, help="Shared party speed in meters/second")
    parser.add_argument("--dp-seconds", type=float, default=30, help="Search budget at EACH decision (seconds)")
    parser.add_argument("--dp-beam", type=int, default=1000)
    parser.add_argument("--dp-step", type=int, choices=(60, 300), default=60)
    parser.add_argument("--dp-buffer", type=int, default=300, help="DP scheduling margin per ride; not added to actual visit times")
    parser.add_argument("--quantile", choices=("p10", "p50", "p90"), default="p90")
    parser.add_argument("--walk-cost", type=float, default=2.0, help="DP utility cost per kilometer")
    parser.add_argument("--output", type=Path, help="Saved result JSON; default Park/comparisons/<day>-seed<seed>.json")
    parser.add_argument("--headless", action="store_true", help="Calculate and save without opening Pygame")
    parser.add_argument("--replay", type=Path, help="Visualize a saved comparison without loading ML models")
    parser.add_argument("--speed", type=float, default=300, help="Replay seconds per real second")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        from Park.comparison.history import ZONE, load_day

        if not math.isfinite(args.speed) or args.speed <= 0:
            raise ValueError("Replay speed must be finite and positive")
        if not args.headless and importlib.util.find_spec("pygame") is None:
            raise ValueError("Pygame is required for replay. Install pygame, or use --headless to calculate first.")
        if args.replay:
            payload = json.loads(args.replay.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or not payload.get("complete"):
                raise ValueError("Replay requires a complete version-1 comparison")
        else:
            if (not math.isfinite(args.dp_seconds) or args.dp_seconds <= 0 or args.dp_beam < 1
                    or args.dp_buffer < 0 or not math.isfinite(args.walk_cost) or args.walk_cost < 0
                    or args.samples < 1 or args.threads < 1):
                raise ValueError("Invalid DP search, sample, thread, or walking cost settings")
            for label, path in (("Dataset", args.dataset),
                                ("Wait model", args.forecast_checkpoint), ("Forecast Python", args.forecast_python)):
                if not path.is_file():
                    raise ValueError(f"{label} not found: {path}")
            profile = make_profile(args)
            day = load_day(args.dataset, args.day)
            import torch
            from Park.comparison.policies import DPPolicy, ForecastClient, RLPolicy
            from Park.comparison.runner import run_policy
            from Park.planner.data import WalkingMap

            torch.set_num_threads(args.threads)
            torch.manual_seed(args.seed)
            if args.model is None:
                from Park.training.features import GUEST_FEAT_DIM, RIDE_DYNAMIC_FEAT_DIM
                for candidate in (ROOT/"Park/checkpoints/ppo/ppo_final.pt", ROOT/"Park/companion/model/v2.pt"):
                    if not candidate.is_file():
                        continue
                    model_config = torch.load(candidate, map_location="cpu", weights_only=False).get("config", {})
                    if (model_config.get("guest_feat_dim") == GUEST_FEAT_DIM
                            and model_config.get("ride_dynamic_feat_dim") == RIDE_DYNAMIC_FEAT_DIM
                            and model_config.get("num_rides") == config.NUM_RIDES
                            and model_config.get("environment_dynamic_feat_dim") == 3):
                        args.model = candidate
                        break
                    print(f"Skipping incompatible RL checkpoint: {candidate} (observation dimensions)", flush=True)
                if args.model is None:
                    raise ValueError("No compatible RL checkpoint found; pass --model with a checkpoint for the current observation schema")
            print(f"RL checkpoint: {args.model.resolve()}", flush=True)
            walks = WalkingMap()
            clock = lambda t: datetime.fromtimestamp(t, ZONE).strftime("%H:%M:%S")
            print(f"Historical day {day.key}: {clock(day.opening)} to {clock(day.closing)} "
                  f"({len(day.timestamps)} recorded boards)", flush=True)
            print("Same profile and walks; no repeat visits; exit/idle choices retry until close.", flush=True)
            print("Actual visits use posted entry waits and archive interruptions, then walk out by close.", flush=True)
            output = args.output or ROOT/f"Park/comparisons/{day.key}-seed{args.seed}.json"
            payload = dict(version=1, complete=False, day=day.key, opening=day.opening, closing=day.closing,
                           profile=profile, seed=args.seed, source=day.source,
                           rl_checkpoint=str(args.model.resolve()),
                           settings=dict(dp_seconds=args.dp_seconds, dp_beam=args.dp_beam,
                                         dp_step=args.dp_step, dp_buffer=args.dp_buffer, quantile=args.quantile,
                                         walk_cost=args.walk_cost, samples=args.samples, threads=args.threads),
                           assumptions=["Recorded five-minute posted waits, not measured guest queue outcomes.",
                                        "Shared OSM paths, ride-node exit locations, fixed ride durations, no repeats.",
                                        "Exit/idle decisions wait for the next board until the departure walk is due.",
                                        "Unavailable archive buckets interrupt queues/rides without completion credit.",
                                        "Interpolated history is hidden until its right endpoint has been observed."],
                           timestamps=day.timestamps, waits=day.waits,
                           interpolated_ticks={str(t): right for t, right in day.interpolated.items()}, runs=[])

            def progress(name, now, count, detail):
                status = detail.get("planner", {}).get("status", "")
                print(f"{name:2} {clock(now)} | {count:2} completed | {status}", flush=True)

            rl = RLPolicy(args.model, args.device, walks, profile, day.opening, day.closing)
            payload["runs"].append(run_policy(day, walks, rl, profile, progress))
            save_result(output, payload)
            forecasts = ForecastClient(args.forecast_python, dict(
                dataset=str(args.dataset.resolve()), checkpoint=str(args.forecast_checkpoint.resolve()),
                day=day.key, device=args.forecast_device, samples=args.samples,
                seed=args.seed, threads=args.threads))
            try:
                dp = DPPolicy(walks, profile, day.closing, forecasts, seconds=args.dp_seconds,
                              beam=args.dp_beam, step=args.dp_step, buffer=args.dp_buffer,
                              quantile=args.quantile, walk_cost=args.walk_cost)
                payload["runs"].append(run_policy(day, walks, dp, profile, progress))
                payload["wait_model"] = forecasts.metadata
            finally:
                forecasts.close()
            payload["complete"] = True
            save_result(output, payload)
            print(f"Saved comparison: {output.resolve()}", flush=True)
            print(f"Wait-model day split: {payload['wait_model']['day_split']}", flush=True)
            for run in payload["runs"]:
                m = run["metrics"]
                print(f"{run['name']}: {m['rides_completed']} rides | pref {m['preference_score']:.3f} | "
                      f"utility {m['preference_utility']:.1f}% | must-do {m['must_dos_completed']}/{m['must_dos_assigned']} | "
                      f"queue {m['queue_sec']/60:.0f} min | walk {m['walk_m']/1000:.2f} km", flush=True)
        if not args.headless:
            from Park.comparison.app import run_replay
            run_replay(payload, speed=args.speed)
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Comparison interrupted. Any completed agent run remains in the output JSON.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
