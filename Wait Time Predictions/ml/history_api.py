"""Paid REST/history budgets shared by local processes, with bounded retries."""

import json
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import closing
from email.utils import parsedate_to_datetime
from ml import config

API = "https://api.themeparks.wiki/v1"


class RateBudget:
    def __init__(self, path=None):
        self.path = path or config.ROOT / "data/themeparks-rate.sqlite3"

    def acquire(self, history=False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            now = time.time()
            with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
                db.execute("CREATE TABLE IF NOT EXISTS requests (time REAL, history INTEGER)")
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM requests WHERE time <= ?", (now - 3600,))
                rest = [r[0] for r in db.execute("SELECT time FROM requests WHERE time > ? ORDER BY time", (now - 60,))]
                past = [r[0] for r in db.execute("SELECT time FROM requests WHERE history=1 ORDER BY time")]
                delay = max(rest[-600] + 60 - now if len(rest) >= 600 else 0,
                            past[-1200] + 3600 - now if history and len(past) >= 1200 else 0)
                if delay <= 0:
                    db.execute("INSERT INTO requests VALUES (?, ?)", (now, int(history)))
                    return
            print(f"API budget: waiting {int(delay) + 1}s before the next request (cached work is saved).", flush=True)
            time.sleep(min(30, delay + .1))


def retry_delay(value, default):
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return default


def request_json(path, key):
    for attempt in range(5):
        RateBudget().acquire(history="/history" in path)
        request = urllib.request.Request(API + path, headers={
            "X-API-Key": key, "User-Agent": "OmniQueueBackfill/2.0",
        })
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f"ThemeParks.wiki returned HTTP {error.code} for {path}; rerun to resume the saved cache") from None
            delay = retry_delay(error.headers.get("Retry-After"), 2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise RuntimeError("ThemeParks.wiki is unreachable; rerun to resume the saved cache") from None
            delay = 2 ** attempt
        print(f"API retry in {int(delay) + 1}s", flush=True)
        # Short sleeps keep interruption responsive even for long Retry-After values.
        while delay > 0:
            pause = min(30, delay)
            time.sleep(pause)
            delay -= pause
