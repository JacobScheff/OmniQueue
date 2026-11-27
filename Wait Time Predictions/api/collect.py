"""Collect park wait-time snapshots while the park is scheduled to open."""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from struct import Struct
from zoneinfo import ZoneInfo
from urllib.parse import parse_qs, urlsplit
from concurrent.futures import ThreadPoolExecutor
from ml.parks import PARKS, get_park

import httpx
from google.api_core.exceptions import AlreadyExists, GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud import firestore
from google.oauth2 import service_account

PARK_ID = "7340550b-c14d-4def-80bb-acdb51d49a66"
PARK_TIMEZONE = ZoneInfo("America/Los_Angeles")
PARK_API = f"https://api.themeparks.wiki/v1/entity/{PARK_ID}"
OPERATING_HOURS = {"OPERATING", "EXTRA_HOURS", "TICKETED_EVENT", "PRIVATE_EVENT"}
STATUS_CODES = {"UNKNOWN": 0, "OPERATING": 1, "DOWN": 2, "CLOSED": 3, "REFURBISHMENT": 4}
MISSING_WAIT = 0xFFFF
RIDE_RECORD = Struct("<HB")
FIRESTORE_TIMEOUT = 4
logger = logging.getLogger(__name__)


def collector_parks(requested=None):
    """Use the same park selection for collection and public readiness checks."""
    slugs = requested or os.environ.get("COLLECTOR_PARKS", ",".join(PARKS)).split(",")
    return list(dict.fromkeys(get_park(slug.strip()) for slug in slugs))


def fetch_park_data(client, endpoint, field, park=None):
    park_id = get_park(park).id if park else PARK_ID
    response = client.get(f"https://api.themeparks.wiki/v1/entity/{park_id}/{endpoint}")
    response.raise_for_status()
    payload = response.json()
    if payload.get("id") != park_id or not isinstance(payload.get(field), list):
        raise ValueError(f"Invalid park {endpoint} response")
    return payload[field]


def park_is_open(client, sampled_at, park_date, park=None):
    # A session that opens on the last day of a month can close in the next month.
    months = [park_date.strftime("%Y/%m")]
    if park_date.day == 1:
        months.append((park_date - timedelta(days=1)).strftime("%Y/%m"))

    for month in months:
        for session in fetch_park_data(client, f"schedule/{month}", "schedule", park):
            if session["type"] in OPERATING_HOURS:
                opens = datetime.fromisoformat(session["openingTime"])
                closes = datetime.fromisoformat(session["closingTime"])
                if opens <= sampled_at < closes:
                    return True
    return False


def pack_snapshot(live_data, park=None):
    park_id = get_park(park).id if park else PARK_ID
    attractions = sorted(
        (ride for ride in live_data if ride["entityType"] == "ATTRACTION"),
        key=lambda ride: ride["id"],
    )
    if not attractions:
        raise ValueError("Live feed contains no attractions")

    catalog_entries = []
    packed_rides = bytearray()
    seen_ids = set()
    for ride in attractions:
        ride_id, name = ride["id"], ride["name"]
        if not isinstance(ride_id, str) or not ride_id or not isinstance(name, str):
            raise ValueError("Invalid attraction identity in live feed")
        if ride_id in seen_ids:
            raise ValueError("Duplicate attraction ID in live feed")
        seen_ids.add(ride_id)
        catalog_entries.append({"id": ride_id, "name": name})

        status = ride.get("status") or "UNKNOWN"
        standby = (ride.get("queue") or {}).get("STANDBY") or {}
        wait = standby.get("waitTime") if status == "OPERATING" else None
        if wait is None:
            wait = MISSING_WAIT
        elif type(wait) is not int or not 0 <= wait < MISSING_WAIT:
            raise ValueError("Invalid standby wait in live feed")
        packed_rides.extend(RIDE_RECORD.pack(wait, STATUS_CODES.get(status, 0)))

    catalog = {"park_id": park_id, "encoding_version": 1, "attractions": catalog_entries}
    catalog_json = json.dumps(catalog, sort_keys=True, separators=(",", ":"))
    catalog_version = hashlib.sha256(catalog_json.encode()).hexdigest()
    return catalog_version, catalog, bytes(packed_rides)


def firestore_client(service_account_json):
    info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(info)
    return firestore.Client(project=info["project_id"], credentials=credentials)


def store_snapshot(db, sampled_at, park_date, catalog_version, catalog, rides):
    park_id = catalog["park_id"]
    bucket = sampled_at.replace(minute=sampled_at.minute // 5 * 5, second=0, microsecond=0)
    catalog_ref = db.collection("wait_time_catalogs").document(catalog_version)
    rpc_options = {"retry": None, "timeout": FIRESTORE_TIMEOUT}
    if not catalog_ref.get(field_paths=("park_id",), **rpc_options).exists:
        # Publish the immutable catalog before any snapshot can reference it.
        try:
            catalog_ref.create(catalog, **rpc_options)
        except AlreadyExists:
            pass  # Another invocation published this catalog version.

    snapshot_ref = db.collection("wait_time_snapshots").document(f"{park_id}_{bucket:%Y%m%dT%H%MZ}")
    try:
        snapshot_ref.create({
            "sampled_at": sampled_at,
            "park_date": park_date.isoformat(),
            "catalog_version": catalog_version,
            "rides": rides,
        }, **rpc_options)
    except AlreadyExists:
        return 0
    return 1


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # A side-effect-free readiness endpoint lets deployment checks verify
        # the active park set without exposing the cron secret or doing writes.
        try:
            parks = collector_parks()
        except ValueError:
            return self.respond(500, status="not_ready", error="unknown_park")
        configured = bool(os.environ.get("CRON_SECRET") and os.environ.get("FIRESTORE_SERVICE_ACCOUNT_JSON"))
        return self.respond(200 if configured else 503, status="ready" if configured else "not_ready",
                            collector_version=3, configured=configured,
                            snapshot_bucket_minutes=5, parks=[park.slug for park in parks],
                            source_commit=os.environ.get("VERCEL_GIT_COMMIT_SHA"))

    def do_POST(self):
        secret = os.environ.get("CRON_SECRET")
        service_account_json = os.environ.get("FIRESTORE_SERVICE_ACCOUNT_JSON")
        if not secret or not service_account_json:
            return self.respond(500, error="collector_not_configured")
        authorization = self.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization.encode(), f"Bearer {secret}".encode()):
            return self.respond(401, error="unauthorized")

        requested = parse_qs(urlsplit(self.path).query).get("park")
        try:
            parks = collector_parks(requested)
        except ValueError:
            return self.respond(400, error="unknown_park")
        sampled_at = datetime.now(timezone.utc)
        try:
            db = firestore_client(service_account_json)
        except (GoogleAuthError, ValueError, KeyError, TypeError, AttributeError):
            return self.respond(500, error="collector_not_configured")

        def collect_one(park):
            park_date = sampled_at.astimezone(park.tz).date()
            try:
                with httpx.Client(timeout=5, headers={"User-Agent": "OmniQueueWaitCollector/2.0"}) as client:
                    if not park_is_open(client, sampled_at, park_date, park):
                        return {"park": park.slug, "status": "park_closed", "inserted": 0}
                    version, catalog, rides = pack_snapshot(fetch_park_data(client, "live", "liveData", park), park)
                inserted = store_snapshot(db, sampled_at, park_date, version, catalog, rides)
                return {"park": park.slug, "status": "collected", "inserted": inserted}
            except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
                logger.exception("Park data unavailable: %s", park.slug)
                return {"park": park.slug, "error": "park_data_unavailable", "inserted": 0}
            except (GoogleAPIError, GoogleAuthError):
                logger.exception("Firestore write failed: %s", park.slug)
                return {"park": park.slug, "error": "database_unavailable", "inserted": 0}

        try:
            with ThreadPoolExecutor(max_workers=min(10, len(parks))) as workers:
                results = list(workers.map(collect_one, parks))
        finally:
            db.close()
        failed = any("error" in result for result in results)
        return self.respond(502 if failed else 200, status="partial_failure" if failed else "collected",
                            inserted=sum(result["inserted"] for result in results), parks=results)

    def respond(self, status_code, **data):
        body = json.dumps(data).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
