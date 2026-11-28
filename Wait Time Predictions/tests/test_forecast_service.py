"""Regression checks for mobile replay and the read API."""

import gzip
import importlib.util
import json
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from ml.data import MISSING
from ml.forecast_admin import publish_model
from ml.forecast_service import ForecastBusy, ForecastService, legal_starts
from ml.forecast_store import ForecastStore, published_models
from ml.parks import get_park


class ForecastServiceTests(unittest.TestCase):
    def test_manager_reads_publication_status_in_one_batch(self):
        db = Mock(project="forecast-project")
        timestamp = datetime(2026, 11, 27, 18, tzinfo=timezone.utc)
        published = {"active_model": {"name": "Disneyland baseline", "version": "v2", "sha256": "abc"},
                     "previous_model": {"version": "v1"}, "activated_at": timestamp}
        first = Mock(id="disneyland")
        first.to_dict.return_value = published
        another = Mock(id="epcot")
        another.to_dict.return_value = None
        db.get_all.return_value = [first, another]
        with patch("google.cloud.firestore.Client", return_value=db):
            status = published_models(["disneyland", "epcot"])
        self.assertEqual(status["project"], "forecast-project")
        self.assertIsNone(status["error"])
        self.assertEqual(status["parks"]["disneyland"]["activated_at"], timestamp.isoformat())
        self.assertEqual(status["parks"]["disneyland"]["previous_version"], "v1")
        self.assertIsNone(status["parks"]["epcot"])
        db.get_all.assert_called_once()
        db.close.assert_called_once()

    def test_publish_from_latest_path_uploads_the_best_sibling(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "best.latest.pt"
            best = root / "best.pt"
            latest.write_bytes(b"latest weights")
            best.write_bytes(b"best weights")
            store = Mock()
            with patch("ml.forecast_admin.verify_model", return_value={"name": "Best"}) as verify:
                manifest = publish_model(store, get_park("disneyland"), latest)
            self.assertEqual(verify.call_count, 1)
            store.write_once.assert_called_once_with(manifest["path"], b"best weights", "application/octet-stream")
            best.unlink()
            with self.assertRaisesRegex(ValueError, "no best checkpoint"):
                publish_model(store, get_park("disneyland"), latest)

    def test_activation_replaces_the_parks_active_model_in_one_document(self):
        store = ForecastStore.__new__(ForecastStore)
        store.db = Mock()
        reference = store.park("epcot")
        first = {"version": "first"}
        second = {"version": "second"}
        reference.get.return_value.to_dict.return_value = {"active_model": first}
        with patch("google.cloud.firestore.transactional", side_effect=lambda operation: operation):
            store.activate("epcot", second)
        transaction = store.db.transaction.return_value
        target, fields = transaction.set.call_args.args
        self.assertIs(target, reference)
        self.assertEqual(fields["active_model"], second)
        self.assertEqual(fields["previous_model"], first)
        self.assertEqual(transaction.set.call_args.kwargs["merge"], ["active_model", "previous_model", "activated_at"])
        transaction.set.assert_called_once()

    def test_start_excludes_interpolation_using_an_observation_at_the_cutoff(self):
        day = {"rides": [{"id": "ride", "name": "Ride"}], "timestamps": [300, 900, 1200],
               "expected_timestamps": [300, 600, 900, 1200, 1500],
               "tokens": [[4], [6], [MISSING]]}
        self.assertEqual(legal_starts(day, ["ride"]), [600, 1200])
        self.assertEqual(legal_starts(day, ["ride", "absent"]), [])

    def test_public_day_rejects_mixed_source_even_if_indexed(self):
        store = Mock()
        mixed = gzip.compress(json.dumps({"source": "firestore_with_history_prefix"}).encode())
        store.public_day.return_value = {"source": "firestore", "archive_gzip": mixed}
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
        seed = gzip.compress(json.dumps({"source": "firestore", "timestamps": [300]}).encode())
        store.public_day.side_effect = [
            None, {"source": "firestore", "archive_gzip": seed, "complete": False},
            {"source": "firestore", "archive_gzip": seed, "complete": True},
        ]
        service = ForecastService(store)
        service.today = Mock(return_value=date(2026, 11, 27))
        collected = {"source": "firestore", "timestamps": [300, 600]}
        service.recorded_day = Mock(return_value=collected)
        try:
            for key in (date(2026, 11, 25), date(2026, 11, 26)):
                self.assertEqual(service.day(park, key, None)[2], collected)
            self.assertEqual(service.day(park, date(2026, 11, 24), None)[2]["timestamps"], [300])
            self.assertEqual(service.recorded_day.call_count, 2)
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
        prior = date(2026, 11, 25)
        overnight = [datetime(2026, 11, 27, 9, tzinfo=timezone.utc).timestamp()]
        afternoon = [datetime(2026, 11, 25, 22, tzinfo=timezone.utc).timestamp()]
        sessions = {night: overnight, prior: afternoon}
        service.ticks = Mock(side_effect=lambda park, key: (sessions.get(key, []), 0))
        archive.first_before.return_value = datetime(2026, 11, 25, 22, tzinfo=timezone.utc)
        archive.has_snapshots.return_value = True
        try:
            noon = datetime(2026, 11, 27, 20, tzinfo=timezone.utc)
            self.assertEqual(service.dates(park, None, noon, 3), [
                {"date": "2026-11-26", "complete": None},
                {"date": "2026-11-25", "complete": None},
                {"date": "2026-11-23", "complete": True},
            ])
            self.assertEqual(archive.first_before.call_args.args[0].minute, 5)
            self.assertEqual(archive.has_snapshots.call_args_list[0].args[1].minute, 5)
            self.assertEqual(archive.has_snapshots.call_args_list[1].args[0].day, 25)
        finally:
            service.client.close()
            service.workspace.cleanup()

    def test_recorded_day_reuses_count_and_reads_new_rows_without_a_descending_name_index(self):
        store = Mock()
        park = get_park("disneyland")
        service = ForecastService(store)
        ordered = Mock()
        service.archives[park.slug] = ordered
        query = ordered.interval.return_value
        # The SDK's aggregation result consists of batches of alias/value entries.
        query.count.return_value.get.side_effect = [[[Mock(value=count)]] for count in (1, 1, 2)]
        query.stream.return_value = []
        service.ticks = Mock(return_value=([300], 600))
        try:
            for _ in range(3):
                service.recorded_day(park, date(2026, 11, 27), recheck=True)
            self.assertEqual(query.stream.call_count, 2)
            query.order_by.assert_not_called()
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
