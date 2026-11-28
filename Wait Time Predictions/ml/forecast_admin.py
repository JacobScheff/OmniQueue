"""Publish iOS forecasting models and reactivate published versions."""

import argparse
import io
import re
import tempfile
from pathlib import Path

from ml import config
from ml.forecast_store import ForecastStore, digest, encode
from ml.model_library import is_best_checkpoint
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
    if not is_best_checkpoint(checkpoint):
        raise ValueError("Select the run's best checkpoint to publish to iOS.")
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
            "epoch": checkpoint.get("epoch"),
            "architecture": checkpoint["model_config"].get("architecture", "park_gru_v1"),
            "samples": samples, "seed": settings.SEED, "temperature": settings.TEMPERATURE,
            "ride_ids": sorted(ride_ids),
            "train_dates": checkpoint.get("train_dates", []),
            "validation_dates": checkpoint.get("val_dates", checkpoint.get("validation_dates", [])),
            "test_dates": checkpoint.get("test_dates", []),
        }
    raise ValueError("No complete local day matches this checkpoint's ride catalog for a sample forecast.")


def publish_model(store, park, checkpoint_path, *, samples=config.SAMPLES, model_id=None):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name.lower().endswith(".latest.pt"):
        checkpoint_path = checkpoint_path.with_name(checkpoint_path.name[:-len(".latest.pt")] + ".pt")
        if not checkpoint_path.is_file():
            raise ValueError("This run has no best checkpoint to publish.")
    # Verify and publish the same bytes even if the lab replaces the source checkpoint.
    data = checkpoint_path.read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "model.pt"
        copy.write_bytes(data)
        manifest = verify_model(copy, park, samples=samples)
    manifest.update(sha256=digest(data), checkpoint_bytes=len(data))
    if model_id is not None:
        manifest["source_model_id"] = model_id
    manifest["version"] = digest(encode(manifest))
    store.save_checkpoint(park.slug, manifest, data)
    store.activate(park.slug, manifest)
    return manifest


def activate_model(store, park, version):
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Provide the full version from the publish command.")
    manifest = store.version(park.slug, version)
    if not manifest or manifest["park_id"] != park.id or manifest["version"] != version:
        raise ValueError("This version has not been published for this park.")
    if store.read_checkpoint(park.slug, manifest) is None:
        raise ValueError("The stored model artifact does not match its manifest.")
    store.activate(park.slug, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project")
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--park", choices=PARKS, required=True)
    publish.add_argument("--checkpoint", type=Path, required=True)
    rollback = commands.add_parser("activate")
    rollback.add_argument("--park", choices=PARKS, required=True)
    rollback.add_argument("--version", required=True)
    args = parser.parse_args()
    store = ForecastStore(project=args.project)
    if args.command == "publish":
        model = publish_model(store, get_park(args.park), args.checkpoint)
        print(f"Published {model['name']} ({model['version']}) for {args.park}.")
    else:
        activate_model(store, get_park(args.park), args.version)
        print(f"Activated {args.version} for {args.park}.")


if __name__ == "__main__":
    main()
