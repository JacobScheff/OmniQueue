"""Collect a Disneyland wait-time snapshot while the park is scheduled to open."""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from struct import Struct
from zoneinfo import ZoneInfo

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


def fetch_park_data(client, endpoint, field):
    response = client.get(f"{PARK_API}/{endpoint}")
    response.raise_for_status()
    payload = response.json()
    if payload.get("id") != PARK_ID or not isinstance(payload.get(field), list):
        raise ValueError(f"Invalid park {endpoint} response")
    return payload[field]


def park_is_open(client, sampled_at, park_date):
    # A session that opens on the last day of a month can close in the next month.
    months = [park_date.strftime("%Y/%m")]
    if park_date.day == 1:
        months.append((park_date - timedelta(days=1)).strftime("%Y/%m"))

    for month in months:
        for session in fetch_park_data(client, f"schedule/{month}", "schedule"):
            if session["type"] in OPERATING_HOURS:
                opens = datetime.fromisoformat(session["openingTime"])
                closes = datetime.fromisoformat(session["closingTime"])
                if opens <= sampled_at < closes:
                    return True
    return False


def pack_snapshot(live_data):
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

    catalog = {"park_id": PARK_ID, "encoding_version": 1, "attractions": catalog_entries}
    catalog_json = json.dumps(catalog, sort_keys=True, separators=(",", ":"))
    catalog_version = hashlib.sha256(catalog_json.encode()).hexdigest()
    return catalog_version, catalog, bytes(packed_rides)


def firestore_client(service_account_json):
    info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(info)
    return firestore.Client(project=info["project_id"], credentials=credentials)


def store_snapshot(db, sampled_at, park_date, catalog_version, catalog, rides):
    bucket = sampled_at.replace(minute=sampled_at.minute // 5 * 5, second=0, microsecond=0)
    catalog_ref = db.collection("wait_time_catalogs").document(catalog_version)
    rpc_options = {"retry": None, "timeout": FIRESTORE_TIMEOUT}
    if not catalog_ref.get(field_paths=("park_id",), **rpc_options).exists:
        # Publish the immutable catalog before any snapshot can reference it.
        try:
            catalog_ref.create(catalog, **rpc_options)
        except AlreadyExists:
            pass  # Another invocation published this catalog version.

    snapshot_ref = db.collection("wait_time_snapshots").document(f"{PARK_ID}_{bucket:%Y%m%dT%H%MZ}")
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
    def do_POST(self):
        secret = os.environ.get("CRON_SECRET")
        service_account_json = os.environ.get("FIRESTORE_SERVICE_ACCOUNT_JSON")
        if not secret or not service_account_json:
            return self.respond(500, error="collector_not_configured")
        authorization = self.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization.encode(), f"Bearer {secret}".encode()):
            return self.respond(401, error="unauthorized")

        sampled_at = datetime.now(timezone.utc)
        park_date = sampled_at.astimezone(PARK_TIMEZONE).date()
        try:
            with httpx.Client(timeout=5, headers={"User-Agent": "OmniQueueWaitCollector/1.0"}) as client:
                if not park_is_open(client, sampled_at, park_date):
                    return self.respond(200, status="park_closed", inserted=0)
                catalog_version, catalog, rides = pack_snapshot(fetch_park_data(client, "live", "liveData"))
        except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
            logger.exception("Disneyland schedule or live feed unavailable")
            return self.respond(502, error="park_data_unavailable")

        try:
            db = firestore_client(service_account_json)
        except (GoogleAuthError, ValueError, KeyError, TypeError, AttributeError) as error:
            logger.error("Firestore credentials are invalid (%s)", type(error).__name__)
            return self.respond(500, error="collector_not_configured")

        try:
            inserted = store_snapshot(db, sampled_at, park_date, catalog_version, catalog, rides)
        except (GoogleAPIError, GoogleAuthError) as error:
            logger.error("Firestore write failed (%s)", type(error).__name__)
            return self.respond(503, error="database_unavailable")
        return self.respond(200, status="collected", inserted=inserted)

    def respond(self, status_code, **data):
        body = json.dumps(data).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
