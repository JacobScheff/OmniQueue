# Benchmark

**Module:** `benchmark.py`

## Running

```bash
python3 benchmark.py --seed 0 --runs 5 --router heuristic
```

## Output

Reports average/min/max wall time, party count, guest count, rides completed, rides per party, average wait variance, and breakdown count.

## Notes

- Full-scale runs (~50k guests) target fast throughput for RL training.
- Tier 1 optimizations: struct-of-arrays parties, bucket timing wheel, precomputed walk rows, Numba routing.
- Compare before/after with the same seed:

```bash
python3 benchmark.py --seed 42 --runs 5
python3 -m cProfile -s cumulative benchmark.py --runs 1
```
