"""Private Cloud Storage artifacts and Firestore indexes for the iOS forecast service."""

import gzip
import hashlib
import json
import os

from ml.config import PROJECT


FORECAST_PARKS = "forecast_parks"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def forecast_project(override=None):
    return override or os.environ.get("GOOGLE_CLOUD_PROJECT") or PROJECT


def published_models(slugs):
    """Read all selected park activations in one Firestore request."""
    from google.api_core.exceptions import GoogleAPIError
    from google.auth.exceptions import GoogleAuthError
    from google.cloud import firestore

    status = {"project": forecast_project(),
              "bucket": os.environ.get("OMNIQUEUE_FORECAST_BUCKET"), "error": None, "parks": {}}
    db = None
    try:
        db = firestore.Client(project=status["project"])
        status["project"] = db.project
        collection = db.collection(FORECAST_PARKS)
        records = db.get_all([collection.document(slug) for slug in slugs],
                             field_paths=["active_model", "previous_model", "activated_at"],
                             retry=None, timeout=5)
        status["parks"] = {record.id: publication_summary(record.to_dict() or {})
                           for record in records}
    except (GoogleAuthError, OSError, ValueError):
        status["error"] = "Configure Google Application Default Credentials to check published models."
    except GoogleAPIError:
        status["error"] = "Could not read Firestore. Check the project, connection, and Firestore read access."
    finally:
        if db is not None:
            db.close()
    return status


def publication_summary(record):
    model = record.get("active_model")
    if not model:
        return None
    activated = record.get("activated_at")
    previous = record.get("previous_model")
    return {**{key: model.get(key) for key in ("name", "version", "sha256", "epoch", "source_model_id")},
            "activated_at": activated.isoformat() if activated else None,
            "previous_version": previous.get("version") if previous else None}


class ForecastStore:
    def __init__(self, bucket=None, project=None):
        from google.cloud import firestore, storage

        bucket = bucket or os.environ.get("OMNIQUEUE_FORECAST_BUCKET")
        if not bucket:
            raise ValueError("Set OMNIQUEUE_FORECAST_BUCKET to the private forecast artifact bucket.")
        project = forecast_project(project)
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
        return self.db.collection(FORECAST_PARKS).document(slug)

    def active_model(self, slug):
        value = self.park(slug).get(timeout=5).to_dict()
        return value.get("active_model") if value else None

    def activate(self, slug, manifest):
        from google.cloud.firestore import SERVER_TIMESTAMP, transactional

        reference = self.park(slug)

        @transactional
        def update(transaction):
            previous = reference.get(transaction=transaction).to_dict() or {}
            if previous.get("active_model") != manifest:
                transaction.set(reference, {"active_model": manifest,
                                            "previous_model": previous.get("active_model"),
                                            "activated_at": SERVER_TIMESTAMP},
                                merge=["active_model", "previous_model", "activated_at"])

        update(self.db.transaction())

    def public_day(self, slug, key):
        return self.park(slug).collection("days").document(key).get(timeout=5).to_dict()

    def public_dates(self, slug, before, limit):
        from google.cloud.firestore_v1.base_query import FieldFilter

        days = self.park(slug).collection("days")
        query = days.order_by("date", direction="DESCENDING")
        if before:
            query = query.where(filter=FieldFilter("date", "<", before))
        rows = query.select(["date", "complete"]).limit(limit).stream(timeout=8)
        return [{"date": row.id, "complete": row.get("complete")} for row in rows]
