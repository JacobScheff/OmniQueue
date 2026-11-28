"""Private checkpoint storage, activations, and archive reads on Firestore."""

import gzip
import hashlib
import json
import os

from ml.config import PROJECT


FORECAST_PARKS = "forecast_parks"
CHECKPOINT_CHUNK_SIZE = 768 * 1024


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def forecast_project(override=None):
    return override or os.environ.get("GOOGLE_CLOUD_PROJECT") or PROJECT


def forecast_firestore(project=None):
    from google.cloud import firestore
    from google.oauth2 import service_account

    key = os.environ.get("FIRESTORE_SERVICE_ACCOUNT_JSON")
    credentials = service_account.Credentials.from_service_account_info(json.loads(key)) if key else None
    return firestore.Client(project=forecast_project(project), credentials=credentials)


def published_models(slugs):
    """Read all selected park activations in one Firestore request."""
    from google.api_core.exceptions import GoogleAPIError
    from google.auth.exceptions import GoogleAuthError

    status = {"project": forecast_project(), "error": None, "parks": {}}
    db = None
    try:
        db = forecast_firestore(status["project"])
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
    def __init__(self, project=None):
        self.db = forecast_firestore(project)

    def park(self, slug):
        return self.db.collection(FORECAST_PARKS).document(slug)

    def checkpoint_chunks(self, slug, checksum, size):
        chunks = self.park(slug).collection("checkpoints").document(checksum).collection("forecast_model_chunks")
        count = (size + CHECKPOINT_CHUNK_SIZE - 1) // CHECKPOINT_CHUNK_SIZE
        return [chunks.document(str(index)) for index in range(count)]

    def save_checkpoint(self, slug, manifest, data):
        from google.api_core.exceptions import AlreadyExists

        size, checksum = manifest["checkpoint_bytes"], manifest["sha256"]
        if size != len(data) or digest(data) != checksum:
            raise ValueError("The checkpoint bytes do not match the model manifest.")
        references = self.checkpoint_chunks(slug, checksum, size)
        pieces = [data[offset:offset + CHECKPOINT_CHUNK_SIZE] for offset in range(0, size, CHECKPOINT_CHUNK_SIZE)]
        # Content-addressed chunks can be retried after a partial write. Publish
        # the version only after every chunk has been written and checked.
        for ref, piece in zip(references, pieces):
            try:
                ref.create({"payload": piece}, timeout=15)
            except AlreadyExists:
                if ref.get(timeout=8).get("payload") != piece:
                    raise ValueError("A stored checkpoint chunk failed verification.") from None
        if self.read_checkpoint(slug, manifest) != data:
            raise ValueError("The uploaded checkpoint failed verification.")
        reference = self.park(slug).collection("versions").document(manifest["version"])
        try:
            reference.create(manifest, timeout=10)
        except AlreadyExists:
            if reference.get(timeout=8).to_dict() != manifest:
                raise ValueError("The stored model manifest differs from this version.") from None

    def read_checkpoint(self, slug, manifest):
        size, checksum = manifest["checkpoint_bytes"], manifest["sha256"]
        if type(size) is not int or size < 1:
            return None
        references = self.checkpoint_chunks(slug, checksum, size)
        rows = {row.id: row.to_dict().get("payload") for row in self.db.get_all(references, timeout=15) if row.exists}
        pieces = [rows.get(reference.id) for reference in references]
        if any(not isinstance(piece, bytes) for piece in pieces):
            return None
        data = b"".join(pieces)
        return data if len(data) == size and digest(data) == checksum else None

    def version(self, slug, version):
        return self.park(slug).collection("versions").document(version).get(timeout=8).to_dict()

    def issued(self, park, key):
        record = self.park(park).collection("issued_forecasts").document(str(key)).get(timeout=5).to_dict()
        return json.loads(gzip.decompress(record["forecast_gzip"])) if record else None

    def save_issued(self, park, key, value):
        from google.api_core.exceptions import AlreadyExists

        reference = self.park(park).collection("issued_forecasts").document(str(key))
        try:
            reference.create({"forecast_gzip": gzip.compress(encode(value), compresslevel=3)}, timeout=8)
        except AlreadyExists:
            pass

    def active_model(self, slug):
        value = self.park(slug).get(timeout=5).to_dict()
        model = value.get("active_model") if value else None
        return model if model and "checkpoint_bytes" in model else None

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
