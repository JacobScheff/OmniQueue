#!/usr/bin/env python3
"""Benchmark harness for the discrete event simulator."""

from __future__ import annotations

import argparse
import time

from Park.simulator import native_backend_name, run_day


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark park day simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--router", type=str, default="heuristic")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "native"],
        help="Simulation backend (requires built C++ extension)",
    )
    args = parser.parse_args()

    times: list[float] = []
    last = None

    for i in range(args.runs):
        seed = args.seed + i
        print(f"Run {i + 1}/{args.runs}...", flush=True)
        t0 = time.perf_counter()
        metrics = run_day(seed=seed, router=args.router, backend=args.backend)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last = metrics
        print(f"  {elapsed:.3f}s", flush=True)

    avg = sum(times) / len(times)
    backend_label = args.backend
    if args.backend == "auto":
        backend_label = native_backend_name()

    print(f"Runs: {args.runs}")
    print(f"Backend: {backend_label}")
    print(f"Wall time avg: {avg:.4f}s (min={min(times):.4f}s, max={max(times):.4f}s)")
    if last:
        print(f"Parties: {last.total_parties}, Guests: {last.total_guests}")
        print(f"Rides completed: {last.rides_completed}")
        print(f"Rides/party: {last.rides_per_party:.2f}")
        print(f"Avg wait variance: {last.avg_wait_variance:.1f}")
        print(f"Breakdowns: {last.breakdown_count}")


if __name__ == "__main__":
    main()
