#!/usr/bin/env python3
"""Benchmark harness for the discrete event simulator."""

from __future__ import annotations

import argparse
import time

from simulator import run_day


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark park day simulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--router", type=str, default="heuristic")
    args = parser.parse_args()

    times: list[float] = []
    last = None

    for i in range(args.runs):
        seed = args.seed + i
        t0 = time.perf_counter()
        metrics = run_day(seed=seed, router=args.router)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last = metrics

    avg = sum(times) / len(times)
    print(f"Runs: {args.runs}")
    print(f"Wall time avg: {avg:.4f}s (min={min(times):.4f}s, max={max(times):.4f}s)")
    if last:
        print(f"Parties: {last.total_parties}, Guests: {last.total_guests}")
        print(f"Rides completed: {last.rides_completed}")
        print(f"Rides/party: {last.rides_per_party:.2f}")
        print(f"Avg wait variance: {last.avg_wait_variance:.1f}")
        print(f"Breakdowns: {last.breakdown_count}")


if __name__ == "__main__":
    main()
