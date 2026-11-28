"""Publish iOS forecasting models and import wait history from the local cache."""

import argparse
import io
import re
import tempfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from ml import config
from ml.forecast_store import ForecastStore, digest, encode
from ml.parks import PARKS, get_park, settings_for


def verify_model(path, park, *, samples=config.SAMPLES):
    import torch

    from ml.attractions import modeling_day, validate_checkpoint_rides
    from ml.data import TICK_SECONDS, TOKEN_LABELS, interpolate_day, load_dataset, validate_day
    from ml.run import predict

    checkpoint = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
    if (checkpoint.get("version"), checkpoint.get("park_id"), checkpoint.get("token_labels")) != (
            1, park.id, TOKEN_LABELS):
        raise ValueError("This checkpoint has the wrong park or an unsupported model format.")
    rides = checkpoint["rides"]
    ride_ids = {ride["id"] for ride in rides}
    if not rides or len(ride_ids) != len(rides):
        raise ValueError("This checkpoint must contain distinct ride IDs.")
    validate_checkpoint_rides(rides)
    settings = settings_for(park, migrate=False, CHECKPOINT=path, DEVICE="cpu", STEPS=1, SAMPLES=samples)
    dataset = load_dataset(settings.DATASET)
    if dataset["park_id"] != park.id:
        raise ValueError("This dataset belongs to a different park.")
    for key, day in reversed(sorted(dataset["days"].items())):
        prepared = interpolate_day(modeling_day(day, ride_ids))
        if {ride["id"] for ride in prepared["rides"]} != ride_ids or len(prepared["timestamps"]) < 2:
            continue
        cutoff = prepared["timestamps"][0] + TICK_SECONDS
        try:
            validate_day(prepared, before=cutoff)
            if prepared["timestamps"][1] != cutoff:
                continue
        except ValueError:
            continue
        settings.PREDICT_DAY = key
        result = predict(settings, print_output=False, day_override=day, start_timestamp=cutoff)
        if len(result["snapshots"]) != 1 or len(result["snapshots"][0]["rides"]) != len(rides):
            raise ValueError("Sample forecast returned unexpected dimensions.")
        return {
            "park_id": park.id, "name": checkpoint.get("run_name") or "Park wait model",
            "architecture": checkpoint["model_config"].get("architecture", "park_gru_v1"),
            "samples": samples, "seed": settings.SEED, "temperature": settings.TEMPERATURE,
            "ride_ids": sorted(ride_ids),
            "train_dates": checkpoint.get("train_dates", []),
            "validation_dates": checkpoint.get("val_dates", checkpoint.get("validation_dates", [])),
            "test_dates": checkpoint.get("test_dates", []),
        }
    raise ValueError("No complete local day matches this checkpoint's ride catalog for a sample forecast.")


def publish_model(store, park, checkpoint_path, *, samples=config.SAMPLES):
    # Verify and publish the same bytes even if the lab replaces the source checkpoint.
    data = Path(checkpoint_path).read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "model.pt"
        copy.write_bytes(data)
        manifest = verify_model(copy, park, samples=samples)
    manifest["sha256"] = digest(data)
    version = digest(encode(manifest))
    manifest.update(version=version, path=f"models/{park.slug}/{manifest['sha256']}.pt")
    if not store.write_once(manifest["path"], data, "application/octet-stream"):
        existing = store.read(manifest["path"])
        if existing is None or digest(existing) != manifest["sha256"]:
            raise ValueError("The existing model artifact does not match its checksum.")
    store.write_json(f"models/{park.slug}/{version}.json.gz", manifest)
    store.activate(park.slug, manifest)
    return manifest


def activate_model(store, park, version):
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Provide the full version from the publish command.")
    manifest = store.read_json(f"models/{park.slug}/{version}.json.gz")
    if not manifest or manifest["park_id"] != park.id or manifest["version"] != version:
        raise ValueError("This version has not been published for this park.")
    data = store.read(manifest["path"])
    if data is None or digest(data) != manifest["sha256"]:
        raise ValueError("The stored model artifact does not match its manifest.")
    store.activate(park.slug, manifest)


def import_local(parks, root, *, since=None, until=None, store=None):
    """Read only local gzip datasets. Cloud writes happen only with a supplied store."""
    import gzip
    import json

    for park in parks:
        path = root / "data" / park.slug / "waits.json.gz"
        if not path.is_file():
            print(f"{park.slug}: missing {path}")
            continue
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            dataset = json.load(stream)
        if dataset["park_id"] != park.id:
            raise ValueError(f"{path} belongs to a different park.")
        counts = Counter()
        days = {**dataset.get("incomplete_days", {}), **dataset["days"]}
        for key, day in sorted(days.items()):
            if (since and key < since) or (until and key > until):
                continue
            complete = key in dataset["days"]
            counts[day.get("source", "unlabeled"), "complete" if complete else "partial"] += 1
            if store:
                store.archive_day(park.slug, key, day, complete)
        if days:
            first = date.fromisoformat(since or min(days))
            last = date.fromisoformat(until or max(days))
            gaps = []
            for offset in range((last - first).days + 1):
                key = (first + timedelta(days=offset)).isoformat()
                if key not in days:
                    gaps.append(key)
        else:
            gaps = []
        print(f"{park.slug}: {', '.join(f'{source} {state}={count}' for (source, state), count in sorted(counts.items())) or 'no cached days'}")
        if gaps:
            print(f"  {len(gaps)} dates without cached observations (may include closed days): "
                  f"{', '.join(gaps[:12])}{' …' if len(gaps) > 12 else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket")
    parser.add_argument("--project")
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("import-local")
    local.add_argument("--park", choices=PARKS)
    local.add_argument("--root", type=Path, default=config.ROOT)
    local.add_argument("--since", type=lambda value: date.fromisoformat(value).isoformat())
    local.add_argument("--until", type=lambda value: date.fromisoformat(value).isoformat())
    local.add_argument("--apply", action="store_true", help="Upload artifacts and index entries. Default is a dry run.")
    publish = commands.add_parser("publish")
    publish.add_argument("--park", choices=PARKS, required=True)
    publish.add_argument("--checkpoint", type=Path, required=True)
    rollback = commands.add_parser("activate")
    rollback.add_argument("--park", choices=PARKS, required=True)
    rollback.add_argument("--version", required=True)
    live = commands.add_parser("archive-live")
    live.add_argument("--park", choices=PARKS)
    args = parser.parse_args()
    if args.command == "import-local":
        if args.since and args.until and args.since > args.until:
            parser.error("--since must be on or before --until")
        parks = [get_park(args.park)] if args.park else list(PARKS.values())
        store = ForecastStore(args.bucket, args.project) if args.apply else None
        import_local(parks, args.root, since=args.since, until=args.until, store=store)
        return
    store = ForecastStore(args.bucket, args.project)
    if args.command == "publish":
        model = publish_model(store, get_park(args.park), args.checkpoint)
        print(f"Published {model['name']} ({model['version']}) for {args.park}.")
    elif args.command == "activate":
        activate_model(store, get_park(args.park), args.version)
        print(f"Activated {args.version} for {args.park}.")
    else:
        from ml.forecast_service import ForecastService

        service = ForecastService(store)
        service.archive_recent([get_park(args.park)] if args.park else PARKS.values())


if __name__ == "__main__":
    main()
