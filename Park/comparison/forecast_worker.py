"""JSON-lines forecast worker, run inside the wait model's Python environment.

The model is loaded once. Every request slices history at its observation origin
before model inference; future archive waits never enter the forecast tensor.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def visible_count(day, moment):
    """Longest contiguous history prefix whose repairs are knowable now."""
    count = bisect_right(day["timestamps"], moment)
    for tick, (_, right) in day.get("interpolated_ticks", {}).items():
        if int(tick) <= moment < right:
            count = min(count, bisect_right(day["timestamps"], int(tick)-1))
    return count


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "Wait Time Predictions"))
    import torch
    from ml.attractions import modeling_day, validate_checkpoint_rides
    from ml.data import TOKEN_LABELS, align_day, load_dataset, time_features, validate_day
    from ml.model import build_model, select_device
    from ml.run import summarize

    settings = json.loads(sys.stdin.readline())
    # Small per-tick tensors are much slower with a large CPU thread pool.
    torch.set_num_threads(settings["threads"])
    device = select_device(settings["device"])
    checkpoint = torch.load(settings["checkpoint"], map_location=device, weights_only=True)
    dataset = load_dataset(Path(settings["dataset"]))
    if (checkpoint.get("version") != 1 or checkpoint["token_labels"] != TOKEN_LABELS
            or checkpoint["park_id"] != dataset["park_id"]):
        raise ValueError("Wait checkpoint vocabulary/park does not match the historical dataset")
    validate_checkpoint_rides(checkpoint["rides"])
    day = modeling_day(dataset["days"][settings["day"]])
    validate_day(day)
    model = build_model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    closing = int(day.get("closing_at", day["timestamps"][-1]+300))
    split = next((name for name in ("train", "validation", "test")
                  if settings["day"] in checkpoint.get(name+"_dates", [])), "outside_recorded_splits")
    metadata = dict(checkpoint=settings["checkpoint"], day_split=split,
                    samples=settings["samples"], seed=settings["seed"])
    cached_origin, cached = None, None
    for line in sys.stdin:
        try:
            moment = int(json.loads(line)["now"])
            count = visible_count(day, moment)
            if not count:
                raise ValueError("No validated observations at the forecast origin")
            origin = day["timestamps"][count-1]
            if origin != cached_origin:
                validate_day(day, before=origin+300)
                history = {**day, "timestamps": day["timestamps"][:count], "tokens": day["tokens"][:count]}
                tokens = align_day(history, checkpoint["rides"])
                future = list(range(origin+300, closing, 300))
                torch.manual_seed(settings["seed"]+count)
                snapshots = []
                with torch.inference_mode(), redirect_stdout(sys.stderr):
                    if future:
                        features = torch.tensor(time_features(history["timestamps"][1:]+future),
                                                dtype=torch.float32, device=device)
                        paths = model.forecast(tokens.to(device), features, steps=len(future),
                                               samples=settings["samples"], temperature=1.0,
                                               history_timestamps=history["timestamps"])
                        snapshots = summarize(paths, checkpoint["rides"], future)
                    observed = summarize(tokens[-1:].unsqueeze(0), checkpoint["rides"], [origin])[0]
                cached = dict(as_of=datetime.fromtimestamp(origin, timezone.utc).isoformat(),
                              origin_snapshot=observed, snapshots=snapshots,
                              closing_timestamp=closing, history_ticks=count, metadata=metadata)
                cached_origin = origin
            print(json.dumps(cached), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)


if __name__ == "__main__":
    main()
