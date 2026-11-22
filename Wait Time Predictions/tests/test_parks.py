"""Park identity, history budgets, timezones and model/cache isolation."""

import contextlib
from datetime import date, datetime
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from api.collect import pack_snapshot, store_snapshot
from api import collect
from ml.backfill import backfill, reconstruct
from ml.data import expected_ticks, export, load_dataset, save_dataset, time_features
from ml.history_api import RateBudget, retry_delay
from ml.lab import ParkLab
from ml.parks import PARKS, get_park, settings_for
from ml.run import park_timestamp, predict
from ml.train import train
from ml.visualize import Visualizer, load_locations


class ParkTests(unittest.TestCase):
    def test_collector_collects_six_parks_and_keeps_successes_when_one_fails(self):
        handler = object.__new__(collect.handler)
        handler.path = "/api/collect"
        handler.headers = {"Authorization": "Bearer test"}
        handler.respond = Mock()
        ride = {"id": "ride", "name": "Ride", "entityType": "ATTRACTION", "status": "OPERATING",
                "queue": {"STANDBY": {"waitTime": 10}}}
        with patch.dict("os.environ", {"CRON_SECRET": "test", "FIRESTORE_SERVICE_ACCOUNT_JSON": "test",
                                       "COLLECTOR_PARKS": ",".join(PARKS)}), \
                patch("api.collect.firestore_client") as client, \
                patch("api.collect.park_is_open", return_value=True), \
                patch("api.collect.fetch_park_data", return_value=[ride]) as fetch, \
                patch("api.collect.store_snapshot", return_value=1) as store:
            handler.do_POST()
            self.assertEqual(handler.respond.call_args.args, (200,))
            self.assertEqual(handler.respond.call_args.kwargs["inserted"], 6)
            self.assertEqual({call.args[4]["park_id"] for call in store.call_args_list}, {p.id for p in PARKS.values()})
            client.return_value.close.assert_called_once()
            store.reset_mock()
            fetch.side_effect = lambda client, endpoint, field, park: [] if park.slug == "epcot" else [ride]
            with patch("api.collect.logger.exception"):
                handler.do_POST()
            self.assertEqual(handler.respond.call_args.args, (502,))
            self.assertEqual(handler.respond.call_args.kwargs["inserted"], 5)

    def test_legacy_disneyland_is_copied_once_without_overwriting_either_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data/themeparks-history").mkdir(parents=True)
            (root / "data/waits.json.gz").write_bytes(b"legacy")
            (root / "data/themeparks-history/day.json.gz").write_bytes(b"raw")
            disneyland = settings_for(root=root)
            disneyland.DATASET.write_bytes(b"new")
            self.assertEqual(settings_for(root=root).DATASET.read_bytes(), b"new")
            self.assertEqual((root / "data/waits.json.gz").read_bytes(), b"legacy")
            paths = [settings_for(park, root=root) for park in PARKS]
            self.assertEqual(len({settings.DATASET for settings in paths}), 6)
            self.assertEqual(len({settings.CHECKPOINT for settings in paths}), 6)
            self.assertFalse(settings_for("epcot", root=root).DATASET.exists())

    def test_florida_calendar_and_features_use_eastern_time_across_dst(self):
        park = get_park("epcot")
        for day, offset in (("2026-07-01", "-04:00"), ("2026-11-15", "-05:00")):
            opening = datetime.fromisoformat(f"{day}T08:00:00{offset}")
            schedule = [{"type": "OPERATING", "openingTime": opening.isoformat(),
                         "closingTime": f"{day}T08:15:00{offset}"}]
            ticks, _ = expected_ticks(date.fromisoformat(day), schedule, park.tz)
            self.assertEqual(ticks[0], opening.timestamp())
            self.assertEqual(park_timestamp(day, "08:05", ticks[0], park.tz), ticks[1])
            self.assertNotEqual(time_features(ticks), time_features(ticks, park.tz))

    def test_deleted_legacy_model_is_not_imported_again_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = root / "checkpoints/waits.pt"
            legacy.parent.mkdir()
            legacy.write_bytes(b"legacy model")
            settings = settings_for("disneyland", root=root)
            archived = settings.CHECKPOINT.parent / "deleted" / "test-backup.pt"
            archived.parent.mkdir()
            settings.CHECKPOINT.rename(archived)
            self.assertFalse(settings_for("disneyland", root=root).CHECKPOINT.exists())
            self.assertEqual(archived.read_bytes(), b"legacy model")
            self.assertEqual(legacy.read_bytes(), b"legacy model")

    def test_firestore_catalog_hash_and_document_id_include_park(self):
        ride = {"id": "ride", "name": "Ride", "entityType": "ATTRACTION", "status": "OPERATING",
                "queue": {"STANDBY": {"waitTime": 10}}}
        a = pack_snapshot([ride], "disneyland")
        b = pack_snapshot([ride], "epcot")
        self.assertNotEqual(a[0], b[0])
        db = Mock()
        moment = datetime.fromisoformat("2026-11-15T15:00:00+00:00")
        store_snapshot(db, moment, moment.date(), *b)
        snapshot_id = db.collection.return_value.document.call_args.args[0]
        self.assertTrue(snapshot_id.startswith(get_park("epcot").id))

    def test_data_refresh_uses_history_boundary_for_firestore_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = settings_for("epcot", root=folder, migrate=False)
            with patch.dict("os.environ", {"THEMEPARKS_API_KEY": "test"}), \
                    patch("ml.backfill.backfill", return_value=date(2026, 11, 18)) as history, \
                    patch("ml.data.export_firestore") as live:
                export(settings)
            self.assertEqual(history.call_args.kwargs["park"], "epcot")
            self.assertEqual(live.call_args.args[0].START_DATE, "2026-11-18")
            self.assertEqual(live.call_args.args[0].DATASET, settings.DATASET)

    def test_paid_backfill_filters_no_waits_and_resumes_without_history_requests(self):
        park = get_park("epcot")
        day = "2026-11-15"
        coverage = {"id": park.id, "summary": {"archiveFrom": "2021-01-01", "recordedTo": day},
                    "entities": [{"id": "ride", "name": "Ride", "entityType": "ATTRACTION",
                                  "stillListed": True, "fields": ["queue.STANDBY", "status"]},
                                 {"id": "show", "name": "Untimed show", "entityType": "ATTRACTION",
                                  "stillListed": True, "fields": ["status"]}]}
        document = {"id": park.id, "entityType": "PARK", "range": {"from": day, "to": day}, "entities": [
            {"id": ride, "name": ride.title(), "entityType": "ATTRACTION", "history": [],
             "opening": {"time": f"{day}T00:00:00-05:00", "observedAt": f"{day}T00:00:00-05:00",
                         "status": "OPERATING", "queue": {"STANDBY": {"waitTime": 10}}}}
            for ride in ("ride", "show")]}
        schedule = {"id": park.id, "schedule": [{"type": "OPERATING", "openingTime": f"{day}T08:00:00-05:00",
                                                 "closingTime": f"{day}T08:15:00-05:00"}]}

        def request(path, key):
            if path == "/me":
                return {"tier": "pro", "historyEarliestDate": day}
            if path.endswith("coverage"):
                return coverage
            return document if "?date=" in path else schedule

        with tempfile.TemporaryDirectory() as folder, patch("ml.backfill.request_json", side_effect=request) as api, \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            backfill("key", root / "waits.gz", root / "raw", park=park)
            data = load_dataset(root / "waits.gz")
            self.assertEqual(data["days"][day]["rides"], [{"id": "ride", "name": "Ride"}])
            self.assertEqual(data["days"][day]["timezone"], "America/New_York")
            api.reset_mock()
            backfill("key", root / "waits.gz", root / "raw", park=park)
            self.assertFalse(any("?date=" in call.args[0] for call in api.call_args_list))
            api.reset_mock()
            backfill("key", root / "waits.gz", root / "raw", park=park, refresh=True)
            self.assertTrue(any("?date=" in call.args[0] for call in api.call_args_list))
            with self.assertRaisesRegex(ValueError, "different park"):
                reconstruct([int(datetime.fromisoformat(f"{day}T08:00:00-05:00").timestamp())],
                            [document], {"ride"}, "disneyland")

    def test_rest_and_history_hourly_budgets_are_shared_between_instances(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "budget.sqlite3"
            RateBudget(path).acquire(history=True)
            with sqlite3.connect(path) as db:
                now = db.execute("SELECT time FROM requests").fetchone()[0]
                db.executemany("INSERT INTO requests VALUES (?, 1)", [(now,)] * 1199)
            db.close()
            with patch("ml.history_api.time.time", side_effect=[now + 1, now + 3601]), \
                    patch("ml.history_api.time.sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
                RateBudget(path).acquire(history=True)
            sleep.assert_called_once()
            self.assertEqual(retry_delay("12", 1), 12)

    def test_models_train_and_forecast_independently_for_california_and_florida(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            for slug in ("disneyland", "epcot"):
                park = get_park(slug)
                settings = settings_for(slug, root=folder, migrate=False, DEVICE="cpu", EPOCHS=1,
                                        TRAIN_SPLIT=1., VAL_SPLIT=0., TEST_SPLIT=0., HIDDEN_SIZE=8,
                                        EMBEDDING_SIZE=4, ATTENTION_HEADS=2, ATTENTION_BLOCKS=1,
                                        MODEL_ARCHITECTURE="ride_transition_v2", HORIZON_WEIGHT=0.,
                                        HISTORY=None, PREDICT_DAY="2026-11-15", PREDICT_START_TIME="08:05",
                                        STEPS=2, SAMPLES=1)
                ticks = [int(datetime(2026, 11, 15, 8, tzinfo=park.tz).timestamp()) + 300 * i for i in range(4)]
                day = {"timestamps": ticks, "tokens": [[2]] * 4, "rides": [{"id": slug, "name": "Ride"}],
                       "timezone": park.timezone}
                save_dataset(settings.DATASET, {"version": 1, "park_id": park.id, "days": {"2026-11-15": day}})
                train(settings)
                checkpoint = torch.load(settings.CHECKPOINT, weights_only=True)
                self.assertEqual(checkpoint["park_id"], park.id)
                result = predict(settings, print_output=False)
                self.assertEqual(datetime.fromisoformat(result["prediction_start"]).timestamp(), ticks[1])
                self.assertEqual(result["snapshots"][0]["rides"][0]["id"], slug)
                app = Visualizer(settings)
                try:
                    self.assertEqual(app.metadata()["timezone"], park.timezone)
                    with patch("ml.config.ROOT", Path(folder)):
                        lab = ParkLab(device="cpu")
                        try:
                            lab.checkpoint_action(app, "save", "best")
                            lab.checkpoint_action(app, "load", "saved")
                            self.assertEqual(app.settings.CHECKPOINT.name, "waits.saved.pt")
                            self.assertEqual(torch.load(app.settings.CHECKPOINT, weights_only=True)["park_id"], park.id)
                        finally:
                            lab.close()
                finally:
                    app.close()
            settings.CHECKPOINT = settings_for("disneyland", root=folder, migrate=False).CHECKPOINT
            with self.assertRaisesRegex(ValueError, "different park"):
                predict(settings, print_output=False)


if __name__ == "__main__":
    unittest.main()
