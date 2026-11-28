"""Private Cloud Storage artifacts and Firestore indexes for the iOS forecast service."""

import gzip
import hashlib
import json
import os
from datetime import date


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ForecastStore:
    def __init__(self, bucket=None, project=None):
        from google.cloud import firestore, storage

        bucket = bucket or os.environ.get("OMNIQUEUE_FORECAST_BUCKET")
        if not bucket:
            raise ValueError("Set OMNIQUEUE_FORECAST_BUCKET to the private forecast artifact bucket.")
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.db = firestore.Client(project=project)
        self.bucket = storage.Client(project=project).bucket(bucket)

    def read(self, path):
        from google.api_core.exceptions import NotFound

        try:
            return self.bucket.blob(path).download_as_bytes(timeout=10)
        except NotFound:
            return None

    def write_once(self, path, data, content_type):
        from google.api_core.exceptions import PreconditionFailed

        try:
            self.bucket.blob(path).upload_from_string(data, content_type=content_type,
                                                      if_generation_match=0, timeout=25)
            return True
        except PreconditionFailed:
            return False

    def read_json(self, path):
        data = self.read(path)
        return json.loads(gzip.decompress(data)) if data is not None else None

    def write_json(self, path, value):
        return self.write_once(path, gzip.compress(encode(value), mtime=0), "application/gzip")

    def first_issued(self, park, key):
        prefix = f"issued/{park}/{key}/"
        first = next(iter(self.bucket.list_blobs(prefix=prefix, max_results=1, timeout=5)), None)
        return self.read_json(first.name) if first is not None else None

    def park(self, slug):
        return self.db.collection("forecast_parks").document(slug)

    def active_model(self, slug):
        value = self.park(slug).get(timeout=5).to_dict()
        return value.get("active_model") if value else None

    def activate(self, slug, manifest):
        from google.cloud.firestore import transactional

        reference = self.park(slug)

        @transactional
        def update(transaction):
            previous = reference.get(transaction=transaction).to_dict() or {}
            if previous.get("active_model") != manifest:
                transaction.set(reference, {"active_model": manifest,
                                            "previous_model": previous.get("active_model")}, merge=True)

        update(self.db.transaction())

    def archive_day(self, slug, key, day, complete):
        if date.fromisoformat(key).isoformat() != key:
            raise ValueError("Archive day must use YYYY-MM-DD.")
        source = day.get("source", "unlabeled")
        collection = "days" if source == "firestore" else "private_days"
        reference = self.park(slug).collection(collection).document(key)
        data = encode(day)
        revision = digest(data)
        path = f"days/{slug}/{collection}/{key}/{revision}.json.gz"
        # Collector archives can improve when late snapshots arrive. A local seed
        # never replaces an already published collector day with fewer observations.
        self.write_once(path, gzip.compress(data, mtime=0), "application/gzip")
        self._index_day(reference, {"path": path, "revision": revision, "source": source,
                                    "complete": complete, "snapshots": len(day["timestamps"]),
                                    "expected": len(day.get("expected_timestamps", day["timestamps"]))},
                        prefer_coverage=collection == "days")

    def _index_day(self, reference, record, *, prefer_coverage):
        from google.cloud.firestore import transactional

        @transactional
        def update(transaction):
            previous = reference.get(transaction=transaction).to_dict()
            if previous == record:
                return
            if previous and prefer_coverage and (previous["complete"], previous["snapshots"]) >= (
                    record["complete"], record["snapshots"]):
                return
            transaction.set(reference, record)

        update(self.db.transaction())

    def public_day(self, slug, key):
        return self.park(slug).collection("days").document(key).get(timeout=5).to_dict()

    def public_dates(self, slug, before, limit):
        days = self.park(slug).collection("days")
        query = days.order_by("__name__", direction="DESCENDING")
        if before:
            query = query.start_after({"__name__": days.document(before)})
        return [{"date": row.id, "complete": row.get("complete")} for row in query.limit(limit).stream(timeout=8)]
