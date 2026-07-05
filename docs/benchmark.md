# Benchmark

**Module:** `benchmark.py`

## Running

```bash
python3 benchmark.py --seed 0 --runs 5
python3 benchmark.py --seed 42 --runs 5 --backend native
```

Requires the C++ extension (`pip install -e .`).

## Output

Reports average/min/max wall time, party count, guest count, rides completed, rides per party, average wait variance, and breakdown count.

## Notes

- Full-scale runs (~50k guests) target ~0.2s/day on modern hardware for RL training throughput.
- Compare runs with the same seed via `--seed`.

```bash
python3 benchmark.py --seed 42 --runs 5
```
