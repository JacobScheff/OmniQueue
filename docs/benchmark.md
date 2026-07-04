# Benchmark

**Module:** `benchmark.py`

## Running

```bash
python3 benchmark.py --seed 0 --runs 5 --router heuristic
```

## Output

Reports average/min/max wall time, party count, guest count, rides completed, rides per party, average wait variance, and breakdown count.

## Notes

- Full-scale runs (~50k guests) are intended for local profiling.
- Phase 1 targets fast DES throughput for future RL training; optimize hot paths (routing, walk lookups) as needed.
- Use `cProfile` for detailed profiling:

```bash
python3 -m cProfile -s cumulative benchmark.py --runs 1
```
