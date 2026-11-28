"""One-time import of local wait history into the private forecast archive."""

import argparse
import gzip
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from ml import config
from ml.forecast_store import ForecastStore, digest, encode
from ml.parks import PARKS, get_park


def archive_day(store, slug, key, day, complete):
    from google.cloud.firestore import transactional

    if date.fromisoformat(key).isoformat() != key:
        raise ValueError("Archive day must use YYYY-MM-DD.")
    source = day.get("source", "unlabeled")
    collection = "days" if source == "firestore" else "private_days"
    reference = store.park(slug).collection(collection).document(key)
    data = encode(day)
    revision = digest(data)
    path = f"days/{slug}/{collection}/{key}/{revision}.json.gz"
    store.write_once(path, gzip.compress(data, mtime=0), "application/gzip")
    record = {"path": path, "revision": revision, "source": source, "complete": complete,
              "snapshots": len(day["timestamps"]),
              "expected": len(day.get("expected_timestamps", day["timestamps"]))}

    @transactional
    def index(transaction):
        previous = reference.get(transaction=transaction).to_dict()
        if previous == record:
            return
        if previous and collection == "days" and (previous["complete"], previous["snapshots"]) >= (
                complete, record["snapshots"]):
            return
        transaction.set(reference, record)

    index(store.db.transaction())


def import_local(parks, root, *, since=None, until=None, store=None):
    for park in parks:
        path = root / "data" / park.slug / "waits.json.gz"
        if not path.is_file():
            print(f"{park.slug}: missing {path}")
            if store is not None:
                raise FileNotFoundError(path)
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
            if store is not None:
                archive_day(store, park.slug, key, day, complete)
        summary = ", ".join(f"{source} {state}={count}" for (source, state), count in sorted(counts.items()))
        print(f"{park.slug}: {summary or 'no cached days'}")
        if days:
            first = date.fromisoformat(since or min(days))
            last = date.fromisoformat(until or max(days))
            gaps = []
            for offset in range((last - first).days + 1):
                key = (first + timedelta(days=offset)).isoformat()
                if key not in days:
                    gaps.append(key)
            if gaps:
                print(f"  {len(gaps)} dates without cached observations (may include closed days): "
                      f"{', '.join(gaps[:12])}{' …' if len(gaps) > 12 else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket")
    parser.add_argument("--project")
    parser.add_argument("--park", choices=PARKS)
    parser.add_argument("--root", type=Path, default=config.ROOT)
    parser.add_argument("--since", type=lambda value: date.fromisoformat(value).isoformat())
    parser.add_argument("--until", type=lambda value: date.fromisoformat(value).isoformat())
    parser.add_argument("--apply", action="store_true", help="Upload artifacts and index entries. Default is a dry run.")
    args = parser.parse_args()
    if args.since and args.until and args.since > args.until:
        parser.error("--since must be on or before --until")
    parks = [get_park(args.park)] if args.park else PARKS.values()
    store = ForecastStore(args.bucket, args.project) if args.apply else None
    import_local(parks, args.root, since=args.since, until=args.until, store=store)


if __name__ == "__main__":
    main()
