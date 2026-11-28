"""Regression checks for mobile replay and the read API."""

import gzip
import importlib.util
import json
import unittest
from datetime import date, datetime, timezone
from unittest.mock import Mock

from ml.data import MISSING
from ml.forecast_service import ForecastBusy, ForecastService, legal_starts
from ml.parks import get_park


class ForecastServiceTests(unittest.TestCase):
    def test_start_excludes_interpolation_using_an_observation_at_the_cutoff(self):
        day = {"rides": [{"id": "ride", "name": "Ride"}], "timestamps": [300, 900, 1200],
               "expected_timestamps": [300, 600, 900, 1200, 1500],
               "tokens": [[4], [6], [MISSING]]}
        self.assertEqual(legal_starts(day, ["ride"]), [600, 1200])
        self.assertEqual(legal_starts(day, ["ride", "absent"]), [])

    def test_public_day_rejects_mixed_source_even_if_indexed(self):
        store = Mock()
        store.public_day.return_value = {"source": "firestore", "path": "mixed"}
        store.read_json.return_value = {"source": "firestore_with_history_prefix"}
        service = ForecastService(store)
        service.today = Mock(return_value=date(2026, 11, 27))
        service.recorded_day = Mock(return_value={"source": "firestore", "timestamps": []})
        try:
            with self.assertRaisesRegex(ValueError, "No recorded waits"):
                service.day(get_park("disneyland"), date(2026, 11, 21), None)
        finally:
            service.client.close()
            service.workspace.cleanup()

    def test_past_day_reads_directly_from_firestore_when_seed_is_absent_or_partial(self):
        park = get_park("disneyland")
        store = Mock()
        store.public_day.side_effect = [None, {"source": "firestore", "path": "seed", "complete": False}]
        store.read_json.return_value = {"source": "firestore", "timestamps": [300]}
        service = ForecastService(store)
        service.today = Mock(return_value=date(2026, 11, 27))
        collected = {"source": "firestore", "timestamps": [300, 600]}
        service.recorded_day = Mock(return_value=collected)
        try:
            for key in (date(2026, 11, 25), date(2026, 11, 26)):
                self.assertEqual(service.day(park, key, None)[2], collected)
        finally:
            service.client.close()
            service.workspace.cleanup()

    def test_history_calendar_includes_live_collector_dates_and_preserves_overnight_opening_date(self):
        park = get_park("disneyland")
        store = Mock()
        store.public_dates.return_value = [{"date": "2026-11-23", "complete": True}]
        service = ForecastService(store)
        archive = Mock()
        service.archives[park.slug] = archive
        night = date(2026, 11, 26)
        closed = datetime(2026, 11, 27, 9, 2, tzinfo=timezone.utc)
        service.ticks = Mock(side_effect=lambda park, key: ([0], int(closed.timestamp())) if key == night else ([], 0))
        archive.latest_before.side_effect = [
            datetime(2026, 11, 27, 8, 45, tzinfo=timezone.utc),
            datetime(2026, 11, 25, 22, tzinfo=timezone.utc),
            None,
        ]
        try:
            noon = datetime(2026, 11, 27, 20, tzinfo=timezone.utc)
            self.assertEqual(service.dates(park, None, noon, 3), [
                {"date": "2026-11-26", "complete": None},
                {"date": "2026-11-25", "complete": None},
                {"date": "2026-11-23", "complete": True},
            ])
            self.assertEqual(archive.latest_before.call_args_list[0].args[0].minute, 5)
            self.assertEqual(archive.latest_before.call_args_list[1].args[0].day, 26)
        finally:
            service.client.close()
            service.workspace.cleanup()

    @unittest.skipUnless(importlib.util.find_spec("flask"), "Install requirements-forecast.txt for HTTP API tests")
    def test_api_limits_date_paging_and_honors_conditional_refresh(self):
        from ml.forecast_app import create_app

        backend = Mock()
        backend.day_payload.return_value = {"date": "2026-11-27", "payload": "x" * 9000}
        backend.dates.return_value = [{"date": str(index)} for index in range(32)]
        backend.forecast.side_effect = ForecastBusy("Busy")
        client = create_app(backend).test_client()
        self.assertEqual(len(client.get("/v1/parks").json["parks"]), 10)
        base = "/v1/parks/disneyland"
        first = client.get(f"{base}/day", headers={"Accept-Encoding": "gzip"})
        self.assertEqual(first.headers["Content-Encoding"], "gzip")
        self.assertEqual(json.loads(gzip.decompress(first.data))["date"], "2026-11-27")
        unchanged = client.get(f"{base}/day", headers={"If-None-Match": first.headers["ETag"]})
        self.assertEqual(unchanged.status_code, 304)
        response = client.get(f"{base}/dates?before=2026-11-30")
        self.assertEqual(len(response.json["dates"]), 31)
        self.assertEqual(response.json["next"], "30")
        park, before, _, limit = backend.dates.call_args.args
        self.assertEqual((park.slug, before, limit), ("disneyland", date(2026, 11, 30), 32))
        self.assertEqual(client.get(f"{base}/forecast?start=123").status_code, 400)
        busy = client.get(f"{base}/forecast")
        self.assertEqual(busy.status_code, 503)
        self.assertIn("Retry-After", busy.headers)
