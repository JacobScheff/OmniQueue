"""Regression checks for mobile replay, cache-only imports, and the read API."""

import gzip
import importlib.util
import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from ml.data import MISSING
from ml.forecast_admin import import_local
from ml.forecast_service import ForecastBusy, ForecastService, legal_starts
from ml.parks import get_park


class ForecastServiceTests(unittest.TestCase):
    def test_start_excludes_interpolation_using_an_observation_at_the_cutoff(self):
        day = {"rides": [{"id": "ride", "name": "Ride"}], "timestamps": [300, 900, 1200],
               "expected_timestamps": [300, 600, 900, 1200, 1500],
               "tokens": [[4], [6], [MISSING]]}
        self.assertEqual(legal_starts(day, ["ride"]), [600, 1200])
        self.assertEqual(legal_starts(day, ["ride", "absent"]), [])

    def test_import_uses_local_dataset_and_preserves_source_and_partial_days(self):
        park = get_park("disneyland")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data" / park.slug / "waits.json.gz"
            path.parent.mkdir(parents=True)
            paid = {"source": "themeparks.wiki/history", "timestamps": [300]}
            mixed = {"source": "firestore_with_history_prefix", "timestamps": [600], "history_before": 600}
            with gzip.open(path, "wt", encoding="utf-8") as target:
                json.dump({"park_id": park.id, "days": {"2026-11-19": paid},
                           "incomplete_days": {"2026-11-21": mixed}}, target)
            store = Mock()
            import_local([park], root, store=store)
        self.assertEqual(store.archive_day.call_count, 2)
        store.archive_day.assert_any_call(park.slug, "2026-11-19", paid, True)
        store.archive_day.assert_any_call(park.slug, "2026-11-21", mixed, False)

    def test_public_day_rejects_mixed_source_even_if_indexed(self):
        store = Mock()
        store.public_day.return_value = {"source": "firestore", "path": "mixed"}
        store.read_json.return_value = {"source": "firestore_with_history_prefix"}
        service = ForecastService(store)
        service.today = Mock(return_value=date(2026, 11, 27))
        try:
            with self.assertRaisesRegex(ValueError, "No recorded waits"):
                service.day(get_park("disneyland"), date(2026, 11, 21), None)
        finally:
            service.client.close()
            service.workspace.cleanup()

    @unittest.skipUnless(importlib.util.find_spec("flask"), "Install requirements-forecast.txt for HTTP API tests")
    def test_api_limits_date_paging_and_honors_conditional_refresh(self):
        from ml.forecast_app import create_app

        backend = Mock()
        backend.today.return_value = date(2026, 11, 27)
        backend.day_payload.return_value = {"date": "2026-11-27", "payload": "x" * 9000}
        backend.store.public_dates.return_value = [{"date": str(index)} for index in range(32)]
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
        backend.store.public_dates.assert_called_once_with("disneyland", "2026-11-27", 32)
        self.assertEqual(client.get(f"{base}/forecast?start=123").status_code, 400)
        busy = client.get(f"{base}/forecast")
        self.assertEqual(busy.status_code, 503)
        self.assertIn("Retry-After", busy.headers)
