"""Run a local, read-only browser for collected Firestore wait times."""

import argparse
import json
import logging
import os
import sys
import webbrowser
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

try:
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter

    from api.collect import MISSING_WAIT, PARK_ID, PARK_TIMEZONE, RIDE_RECORD, STATUS_CODES, firestore_client
except ImportError as error:
    requirements = Path(__file__).with_name("requirements.txt")
    raise SystemExit(
        f"A viewer dependency is missing or could not load ({error}).\n"
        f'Install dependencies for this interpreter with\n"{sys.executable}" -m pip install -r "{requirements}"'
    ) from error


def decode(snapshot, catalog):
    if catalog.get("encoding_version") != 1:
        raise ValueError("Unsupported catalog encoding version")
    attractions = catalog["attractions"]
    payload = snapshot["rides"]
    if len(payload) != len(attractions) * RIDE_RECORD.size:
        raise ValueError("Snapshot length does not match its ride catalog")
    statuses = {value: name for name, value in STATUS_CODES.items()}
    return [
        {**ride, "wait": None if wait == MISSING_WAIT else wait, "status": statuses.get(status, "UNKNOWN")}
        for ride, (wait, status) in zip(attractions, RIDE_RECORD.iter_unpack(payload))
    ]


class Archive:
    def __init__(self, db):
        self.db = db
        self.snapshots = db.collection("wait_time_snapshots")

    @lru_cache(maxsize=64)
    def catalog(self, version):
        doc = self.db.collection("wait_time_catalogs").document(version).get(timeout=15)
        if not doc.exists:
            raise ValueError("The snapshot's ride catalog is missing")
        catalog = doc.to_dict()
        if catalog["park_id"] != PARK_ID:
            raise ValueError("The catalog belongs to a different park")
        return catalog

    def between(self, start, end):
        # Document IDs are chronological UTC buckets, so no additional index is needed.
        return self.snapshots.where(filter=FieldFilter("__name__", ">=", self.snapshots.document(start))).where(
            filter=FieldFilter("__name__", "<", self.snapshots.document(end))
        )

    def load(self, requested_date):
        if not requested_date:
            latest = list(self.snapshots.order_by(
                "sampled_at", direction=firestore.Query.DESCENDING
            ).limit(1).stream(timeout=15))
            if not latest:
                return {"date": datetime.now(PARK_TIMEZONE).date().isoformat(), "snapshots": []}
            requested_date = latest[0].to_dict()["park_date"]
        day = date.fromisoformat(requested_date)
        start = datetime.combine(day, time(), PARK_TIMEZONE).astimezone(timezone.utc)
        end = datetime.combine(day + timedelta(days=1), time(), PARK_TIMEZONE).astimezone(timezone.utc)
        documents = self.between(f"{PARK_ID}_{start:%Y%m%dT%H%MZ}", f"{PARK_ID}_{end:%Y%m%dT%H%MZ}")
        snapshots = []
        for doc in documents.order_by("__name__").limit(400).stream(timeout=15):
            row = doc.to_dict()
            snapshots.append({
                "time": row["sampled_at"].isoformat(),
                "rides": decode(row, self.catalog(row["catalog_version"])),
            })
        return {"date": day.isoformat(), "snapshots": snapshots}


class Viewer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send_error(403)
            return
        path = urlsplit(self.path)
        if path.path == "/":
            self.reply(200, Path(__file__).with_name("viewer.html").read_bytes(), "text/html; charset=utf-8")
        elif path.path == "/api/snapshots":
            requested_date = parse_qs(path.query).get("date", [""])[0]
            try:
                if requested_date:
                    date.fromisoformat(requested_date)
            except ValueError:
                self.reply(400, b'{"error":"Choose a valid date."}')
                return
            try:
                data = self.server.archive.load(requested_date)
                self.reply(200, json.dumps(data).encode())
            except Exception:
                logging.exception("Could not load Firestore snapshots")
                self.reply(502, b'{"error":"Could not load the archive. Check credentials, project access, and the terminal for details."}')
        else:
            self.send_error(404)

    def reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, help="Path to your service account JSON key")
    parser.add_argument("--project", default="omniqueue-9ce1b", help="Project for application default credentials")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.credentials:
        db = firestore_client(args.credentials.read_text(encoding="utf-8-sig"))
    elif os.environ.get("FIRESTORE_SERVICE_ACCOUNT_JSON"):
        db = firestore_client(os.environ["FIRESTORE_SERVICE_ACCOUNT_JSON"])
    else:
        db = firestore.Client(project=args.project)
    with HTTPServer(("127.0.0.1", args.port), Viewer) as server:
        server.archive = Archive(db)
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"OmniQueue archive at {url}\nPress Ctrl+C to stop.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            db.close()


if __name__ == "__main__":
    main()
