"""Regression checks for data boundaries, token masking, and autoregressive inference."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from api.collect import PARK_ID, pack_snapshot
from ml.data import (
    CLOSED, DOWN, MISSING, OVERFLOW, REFURBISHMENT, TOKEN_LABELS, UNKNOWN, VOCAB_SIZE,
    align_day, collect_day, complete_day, expected_ticks, export, interpolate_day, load_dataset,
    ride_token, save_dataset, time_features, validate_day,
)
from ml.model import WaitPredictor, token_loss
from ml.run import forecast_boundary, predict, summarize
from ml.train import Days, collate_days, epoch, train


def attraction(ride_id, status="OPERATING", wait=10):
    return {"id": ride_id, "name": ride_id, "entityType": "ATTRACTION", "status": status,
            "queue": {"STANDBY": {"waitTime": wait}}}


def sample(timestamp, rides):
    version, catalog, payload = pack_snapshot(rides)
    return {"sampled_at": datetime.fromtimestamp(timestamp, timezone.utc),
            "catalog_version": version, "rides": payload}, {version: catalog}


class DatasetTests(unittest.TestCase):
    def test_buckets_include_zero_overflow_and_each_status(self):
        cases = [(0, 0), (2, 0), (3, 1), (5, 1), (300, 60), (301, OVERFLOW), (None, UNKNOWN)]
        for minutes, token in cases:
            self.assertEqual(ride_token({"status": "OPERATING", "wait": minutes}), token)
        for status, token in [("DOWN", DOWN), ("CLOSED", CLOSED),
                              ("REFURBISHMENT", REFURBISHMENT), ("UNKNOWN", UNKNOWN)]:
            self.assertEqual(ride_token({"status": status, "wait": None}), token)

    def test_operating_day_spans_midnight_and_unions_overlapping_sessions(self):
        sessions = [
            {"type": "OPERATING", "openingTime": "2026-11-01T23:50:00-08:00", "closingTime": "2026-11-02T00:10:00-08:00"},
            {"type": "EXTRA_HOURS", "openingTime": "2026-11-01T23:55:00-08:00", "closingTime": "2026-11-02T00:15:00-08:00"},
        ]
        ticks, closing = expected_ticks(date(2026, 11, 1), sessions)
        self.assertEqual(len(ticks), 5)
        self.assertEqual(closing, ticks[-1] + 300)
        self.assertEqual(expected_ticks(date(2026, 11, 2), sessions), ([], 0))

    def test_day_requires_both_boundaries_and_fills_short_interior_gaps(self):
        ticks = [1793548800 + 300 * i for i in range(4)]
        snapshots, catalogs = [], {}
        for tick in ticks:
            snapshot, catalog = sample(tick, [attraction("a"), attraction("b", "CLOSED")])
            snapshots.append(snapshot)
            catalogs.update(catalog)
        result = complete_day(ticks, reversed(snapshots), catalogs.__getitem__)
        self.assertEqual(result["tokens"], [[2, CLOSED]] * 4)
        repaired = complete_day(ticks, snapshots[:1] + snapshots[2:], catalogs.__getitem__)
        self.assertEqual(repaired["timestamps"], ticks)
        self.assertEqual(repaired["tokens"], result["tokens"])
        for index in (0, 3):
            with self.assertRaisesRegex(ValueError, "Missing 1"):
                complete_day(ticks, snapshots[:index] + snapshots[index + 1:], catalogs.__getitem__)
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            complete_day(ticks, snapshots + snapshots[:1], catalogs.__getitem__)

    def test_interpolation_uses_surrounding_endpoints_and_rounds_waits_half_up(self):
        ticks = [1793548800 + 300 * i for i in range(7)]
        snapshots, catalogs = [], {}
        for index, wait in [(0, 0), (3, 20), (4, 5), (6, 10)]:
            snapshot, catalog = sample(ticks[index], [attraction("a", wait=wait)])
            snapshots.append(snapshot)
            catalogs.update(catalog)
        day = complete_day(ticks, snapshots, catalogs.__getitem__)
        self.assertEqual(day["timestamps"], ticks)
        self.assertEqual(day["tokens"], [[0], [1], [3], [4], [1], [2], [2]])
        self.assertEqual(day["interpolated_ticks"], {
            str(ticks[1]): [ticks[0], ticks[3]], str(ticks[2]): [ticks[0], ticks[3]],
            str(ticks[5]): [ticks[4], ticks[6]],
        })
        self.assertIs(interpolate_day(day), day)

    def test_interpolation_threshold_is_per_day_and_strictly_less_than_five(self):
        for missing in (4, 5, 6):
            with self.subTest(missing=missing):
                # Separate one-tick gaps exercise the daily total, not gap size.
                ticks = list(range(0, (2 * missing + 1) * 300, 300))
                day = {"timestamps": ticks[::2], "expected_timestamps": ticks,
                       "tokens": [[2]] * (missing + 1), "rides": [{"id": "a"}]}
                repaired = interpolate_day(day)
                if missing < 5:
                    validate_day(repaired)
                    self.assertEqual(repaired["timestamps"], ticks)
                else:
                    self.assertIs(repaired, day)
                    with self.assertRaisesRegex(ValueError, f"Missing {missing}"):
                        validate_day(repaired)

    def test_interpolation_handles_categories_without_treating_them_as_waits(self):
        day = {"timestamps": [0, 600], "expected_timestamps": [0, 300, 600],
               "tokens": [[2, DOWN, CLOSED, OVERFLOW, MISSING], [DOWN, DOWN, CLOSED, 2, 4]]}
        repaired = interpolate_day(day)
        self.assertEqual(repaired["tokens"][1], [DOWN, DOWN, CLOSED, 2, MISSING])
        with self.assertRaisesRegex(ValueError, "Attractions missing"):
            validate_day(repaired)

    def test_interpolation_does_not_bridge_scheduled_breaks_or_use_future_history(self):
        day = {"timestamps": [0, 900], "expected_timestamps": [0, 300, 900], "tokens": [[0], [6]]}
        self.assertIs(interpolate_day(day), day)
        day["expected_timestamps"] = [0, 300, 600, 900]
        repaired = interpolate_day(day)
        with self.assertRaisesRegex(ValueError, "Missing 2"):
            validate_day(repaired, before=900)
        validate_day(repaired, before=1200)

    def test_export_repairs_finished_cached_days_without_firestore_reads(self):
        ticks = [1793548800 + 300 * i for i in range(3)]
        day = {"timestamps": ticks[::2], "expected_timestamps": ticks, "tokens": [[0], [3]],
               "rides": [{"id": "a"}], "closing_at": ticks[-1] + 300,
               "checked_at": ticks[-1] + 600, "reason": "Missing 1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json.gz"
            save_dataset(path, {"version": 1, "park_id": PARK_ID, "days": {},
                                "incomplete_days": {"2026-11-01": day}})
            settings = SimpleNamespace(DATASET=path, START_DATE="2026-11-01", END_DATE="2026-11-01", REFRESH=False)
            with patch("google.cloud.firestore.Client", side_effect=AssertionError("Unexpected Firestore read")):
                export(settings)
            cached = load_dataset(path)
            self.assertEqual(cached["incomplete_days"], {})
            repaired = cached["days"]["2026-11-01"]
            self.assertEqual(repaired["tokens"], [[0], [2], [3]])
            self.assertNotIn("reason", repaired)

    def test_catalog_changes_preserve_ids_but_missing_rides_discard_the_day(self):
        first, catalogs = sample(1793548800, [attraction("a"), attraction("b", "DOWN")])
        renamed = {**attraction("b", "REFURBISHMENT"), "name": "New name"}
        second, more = sample(1793549100, [renamed, attraction("a", wait=0)])
        catalogs.update(more)
        result = complete_day([1793548800, 1793549100], [second, first], catalogs.__getitem__)
        self.assertEqual(result["tokens"], [[2, DOWN], [0, REFURBISHMENT]])
        second, more = sample(1793549100, [attraction("a")])
        catalogs.update(more)
        with self.assertRaisesRegex(ValueError, "Attractions missing"):
            complete_day([1793548800, 1793549100], [first, second], catalogs.__getitem__)

    def test_cached_export_never_constructs_a_firestore_client(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json.gz"
            dataset = {"version": 1, "park_id": PARK_ID, "days": {"2026-11-01": {}},
                       "incomplete_days": {"2026-11-02": {"checked_at": 2, "closing_at": 1}}, "discarded": {}, "catalogs": {}, "schedules": {}}
            save_dataset(path, dataset)
            self.assertEqual(load_dataset(path), dataset)
            settings = SimpleNamespace(DATASET=path, START_DATE="2026-11-01", END_DATE="2026-11-02", REFRESH=False)
            with patch("google.cloud.firestore.Client", side_effect=AssertionError("Unexpected Firestore read")):
                export(settings)

    def test_export_saves_complete_days_retries_rejections_and_preserves_cache_on_failure(self):
        sessions, all_rows, catalogs = [], [], {}
        for day in (1, 2):
            opening = datetime(2026, 11, day, 16, tzinfo=timezone.utc)
            sessions.append({"type": "OPERATING", "openingTime": opening.isoformat(),
                             "closingTime": opening.replace(minute=20).isoformat()})
            rows = []
            for offset in range(4):
                row, catalog = sample(int(opening.timestamp()) + offset * 300, [attraction("a")])
                rows.append(Mock(to_dict=Mock(return_value=row)))
                catalogs.update(catalog)
            all_rows.append(rows)
        archive = Mock()
        archive.catalog.side_effect = lambda version, **options: catalogs[version]
        archive.between.side_effect = [Mock(stream=Mock(return_value=all_rows[0])),
                                       Mock(stream=Mock(return_value=all_rows[1][:-1]))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json.gz"
            settings = SimpleNamespace(DATASET=path, START_DATE="2026-11-01", END_DATE="2026-11-02",
                                   REFRESH=False, CREDENTIALS=None, PROJECT="test-project")
            progress_output = io.StringIO()
            with patch.dict("os.environ", {"FIRESTORE_SERVICE_ACCOUNT_JSON": ""}), \
                    patch("google.cloud.firestore.Client") as client, \
                    patch("view.Archive", return_value=archive), \
                    patch("api.collect.fetch_park_data", return_value=sessions) as schedule, \
                    redirect_stderr(progress_output):
                export(settings)
                self.assertIn("0/2 days", progress_output.getvalue())
                self.assertIn("2/2 days", progress_output.getvalue())
                self.assertIn("Firestore 0/4 snapshots", progress_output.getvalue())
                saved = load_dataset(path)
                self.assertEqual(list(saved["days"]), ["2026-11-01"])
                self.assertIn("2026-11-02", saved["incomplete_days"])
                self.assertEqual(schedule.call_count, 1)
                self.assertEqual(archive.catalog.call_count, 1)
                client.return_value.close.assert_called_once()
                settings.START_DATE = "2026-11-02"
                settings.REFRESH = True
                archive.between.side_effect = None
                archive.between.return_value = Mock(stream=Mock(return_value=all_rows[1]))
                export(settings)
                saved = load_dataset(path)
                self.assertEqual(len(saved["days"]), 2)
                self.assertEqual(saved["incomplete_days"], {})
                previous = path.read_bytes()
                archive.between.return_value.stream.side_effect = RuntimeError("Network failed")
                with self.assertRaisesRegex(RuntimeError, "Network failed"):
                    export(settings)
                self.assertEqual(path.read_bytes(), previous)

    def test_full_day_sequences_preserve_history_and_mask_gaps_and_padding(self):
        day = {"timestamps": [0, 300, 600, 1200, 1500, 1800],
               "rides": [{"id": "b"}, {"id": "a"}], "tokens": [[DOWN, 0]] * 6}
        rides = [{"id": "a"}, {"id": "b"}, {"id": "new"}]
        self.assertEqual(align_day(day, rides)[0].tolist(), [0, DOWN, UNKNOWN])
        short_day = {**day, "timestamps": [2000, 2300, 2600], "tokens": day["tokens"][:3]}
        days = Days([day, short_day], rides)
        self.assertEqual(len(days), 2)
        inputs, features, targets = collate_days([days[0], days[1]])
        self.assertEqual(inputs.shape, (2, 5, 3))
        torch.testing.assert_close(inputs[0], align_day(day, rides)[:-1])
        torch.testing.assert_close(features[0], torch.tensor(time_features(day["timestamps"][1:])))
        self.assertTrue((targets[0, 2] == UNKNOWN).all())
        self.assertTrue((targets[1, 2:] == UNKNOWN).all())
        _, count = token_loss(torch.zeros(2, 5, 3, VOCAB_SIZE), targets)
        self.assertEqual(count.item(), 12)

    def test_empty_day_is_discarded_and_failed_request_is_left_uncached(self):
        from google.api_core.exceptions import DeadlineExceeded

        sessions = [
            {"type": "OPERATING", "openingTime": f"2026-11-{day:02}T08:00:00-08:00",
             "closingTime": f"2026-11-{day:02}T08:20:00-08:00"}
            for day in (1, 2)
        ]
        empty = Mock(stream=Mock(return_value=iter([])))
        failed = Mock(stream=Mock(side_effect=DeadlineExceeded("Request timed out")))
        archive = Mock()
        archive.between.side_effect = [empty, failed]
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(DATASET=Path(directory) / "data.json.gz", START_DATE="2026-11-01",
                                       END_DATE="2026-11-02", REFRESH=False, CREDENTIALS=None, PROJECT="test-project")
            with patch.dict("os.environ", {"FIRESTORE_SERVICE_ACCOUNT_JSON": ""}), \
                    patch("google.cloud.firestore.Client") as client, \
                    patch("view.Archive", return_value=archive), \
                    patch("api.collect.fetch_park_data", return_value=sessions), \
                    redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Firestore failed while checking 2026-11-02"):
                    export(settings)
                client.return_value.close.assert_called_once()
            cached = load_dataset(settings.DATASET)
            self.assertIn("Missing 4 of 4 scheduled snapshots", cached["incomplete_days"]["2026-11-01"]["reason"])
            self.assertEqual(cached["incomplete_days"]["2026-11-01"]["timestamps"], [])
            self.assertNotIn("2026-11-02", cached["incomplete_days"])
            self.assertNotIn("2026-11-02", cached["days"])
            for query in (empty, failed):
                self.assertEqual(query.stream.call_args.kwargs["timeout"], 30)
                retry = query.stream.call_args.kwargs["retry"]
                self.assertFalse(retry._predicate(DeadlineExceeded("Request timed out")))

    def test_old_rejection_is_refetched_and_live_day_refreshes_until_complete(self):
        class Clock(datetime):
            current = datetime(2026, 11, 1, 16, 10, tzinfo=timezone.utc)

            @classmethod
            def now(cls, tz=None):
                return cls.current.astimezone(tz) if tz else cls.current.replace(tzinfo=None)

        sessions = [{"type": "OPERATING", "openingTime": "2026-11-01T08:00:00-08:00",
                     "closingTime": "2026-11-01T09:00:00-08:00"}]
        documents, catalogs = [], {}
        for index in range(12):
            row, catalog = sample(1793548800 + index * 300, [attraction("a")])
            documents.append(Mock(to_dict=Mock(return_value=row)))
            catalogs.update(catalog)
        archive = Mock()
        archive.catalog.side_effect = lambda version, **options: catalogs[version]
        archive.between.return_value.stream.return_value = documents[:2]
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(DATASET=Path(directory) / "data.json.gz", START_DATE="2026-11-01",
                                       END_DATE=None, REFRESH=False, CREDENTIALS=None, PROJECT="test-project")
            save_dataset(settings.DATASET, {"version": 1, "park_id": PARK_ID, "days": {},
                                           "discarded": {"2026-11-01": "Old rejected day"}, "catalogs": {}, "schedules": {}})
            with patch("ml.data.datetime", Clock), patch.dict("os.environ", {"FIRESTORE_SERVICE_ACCOUNT_JSON": ""}), \
                    patch("google.cloud.firestore.Client") as client, patch("view.Archive", return_value=archive), \
                    patch("api.collect.fetch_park_data", return_value=sessions), \
                    redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                export(settings)
                cached = load_dataset(settings.DATASET)
                self.assertEqual(cached["days"], {})
                self.assertEqual(len(cached["incomplete_days"]["2026-11-01"]["timestamps"]), 2)
                self.assertEqual(cached["discarded"], {})
                Clock.current += timedelta(minutes=2)
                export(settings)
                self.assertEqual(client.call_count, 1)
                Clock.current += timedelta(minutes=3)
                archive.between.return_value.stream.return_value = documents[:3]
                export(settings)
                self.assertEqual(len(load_dataset(settings.DATASET)["incomplete_days"]["2026-11-01"]["timestamps"]), 3)
                Clock.current = datetime(2026, 11, 1, 17, 1, tzinfo=timezone.utc)
                archive.between.return_value.stream.return_value = documents
                export(settings)
                cached = load_dataset(settings.DATASET)
                self.assertIn("2026-11-01", cached["days"])
                self.assertEqual(cached["incomplete_days"], {})
                Clock.current += timedelta(minutes=5)
                export(settings)
                self.assertEqual(client.call_count, 3)


class ModelTests(unittest.TestCase):
    def test_partial_day_predicts_with_complete_prefix_and_rejects_missing_prefix_data(self):
        ticks = [1793548800 + tick * 300 for tick in range(24)]
        snapshots, catalogs = [], {}
        for tick in ticks[:6]:
            snapshot, catalog = sample(tick, [attraction("a"), attraction("b", "DOWN")])
            snapshots.append(snapshot)
            catalogs.update(catalog)
        later, catalog = sample(ticks[10], [attraction("a")])
        snapshots.append(later)
        catalogs.update(catalog)
        day = collect_day(ticks, snapshots, catalogs.__getitem__)
        self.assertEqual(day["tokens"][-1][1], MISSING)
        model = WaitPredictor(2, embedding_size=4, hidden_size=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save({"version": 1, "model_config": {"num_rides": 2, "embedding_size": 4, "hidden_size": 8},
                        "model_state": model.state_dict(), "rides": day["rides"], "token_labels": TOKEN_LABELS,
                        "park_id": PARK_ID}, checkpoint)
            settings = SimpleNamespace(DATASET=root / "data.json.gz", CHECKPOINT=checkpoint, DEVICE="cpu",
                                       SEED=42, PREDICT_DAY="2026-11-01", PREDICT_START_TIME="08:30", HISTORY=None,
                                       STEPS=None, SAMPLES=2, TEMPERATURE=1.0)

            def cache(observations):
                save_dataset(settings.DATASET, {"version": 1, "park_id": PARK_ID, "days": {},
                                               "incomplete_days": {"2026-11-01": observations}})

            cache(day)
            with redirect_stdout(io.StringIO()):
                result = predict(settings, target_time="09:00")
            self.assertEqual(len(result["snapshots"]), 1)
            self.assertEqual(result["as_of"], datetime.fromtimestamp(ticks[5], timezone.utc).isoformat())
            for missing_index in (0, 2):
                broken = {**day, "timestamps": day["timestamps"][:missing_index] + day["timestamps"][missing_index + 1:],
                          "tokens": day["tokens"][:missing_index] + day["tokens"][missing_index + 1:]}
                cache(broken)
                output = io.StringIO()
                with patch.object(WaitPredictor, "forecast") as forecast, redirect_stdout(output):
                    with self.assertRaisesRegex(ValueError, "Missing 1 of 6 scheduled snapshots"):
                        predict(settings, target_time="09:00")
                    forecast.assert_not_called()
                self.assertEqual(output.getvalue(), "")
            broken = {**day, "tokens": [row[:] for row in day["tokens"]]}
            broken["tokens"][1][0] = MISSING
            cache(broken)
            with patch.object(WaitPredictor, "forecast") as forecast, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Attractions missing"):
                    predict(settings, target_time="09:00")
                forecast.assert_not_called()

    def test_prediction_uses_day_start_through_origin_even_with_a_legacy_context_limit(self):
        model = WaitPredictor(1, embedding_size=4, hidden_size=8)
        rides = [{"id": "a", "name": "A"}]
        timestamps = [1793548800 + tick * 300 + (600 if tick >= 8 else 0) for tick in range(80)]
        day = {"rides": rides, "timestamps": timestamps, "tokens": [[tick % 10] for tick in range(80)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save({
                "version": 1, "model_config": {"num_rides": 1, "embedding_size": 4, "hidden_size": 8},
                "model_state": model.state_dict(), "rides": rides, "token_labels": TOKEN_LABELS,
                "context": 1, "park_id": PARK_ID,
            }, checkpoint)
            dataset = root / "waits.json.gz"
            save_dataset(dataset, {"version": 1, "park_id": PARK_ID, "days": {"2026-11-01": day}})
            settings = SimpleNamespace(
                STEPS=2, SAMPLES=2, TEMPERATURE=1.0, SEED=42, DEVICE="cpu", CHECKPOINT=checkpoint,
                HISTORY=None, DATASET=dataset, PREDICT_DAY=None, PREDICT_START_TIME="13:15",
            )
            output = io.StringIO()
            with patch.object(WaitPredictor, "forecast", autospec=True, return_value=torch.zeros(2, 2, 1, dtype=torch.long)) as forecast, redirect_stdout(output):
                result = predict(settings)
            history = forecast.call_args.args[1]
            self.assertEqual(len(history), 61)
            torch.testing.assert_close(history, torch.tensor(day["tokens"][:61]))
            self.assertIn("2026-11-01 13:15 PST", output.getvalue())
            self.assertIn("A | 0 min | p10 0, p90 0 | DOWN 0%", output.getvalue())
            self.assertEqual(set(root.iterdir()), {checkpoint, dataset})
            self.assertEqual(result["as_of"], datetime.fromtimestamp(timestamps[60], timezone.utc).isoformat())
            self.assertEqual(result["prediction_start"], datetime.fromtimestamp(timestamps[61], timezone.utc).isoformat())

    def test_forecast_cutoff_uses_park_time_and_handles_overnight_sessions(self):
        timestamps = [1793548800 + tick * 300 for tick in range(80)]
        day = {"timestamps": timestamps, "tokens": [[0]] * len(timestamps)}
        self.assertEqual(forecast_boundary(day, "2026-11-01", "12:00"), (48, timestamps[48]))
        self.assertEqual(forecast_boundary(day, "2026-11-01", "12:03"), (49, timestamps[49]))
        with self.assertRaisesRegex(ValueError, "after the day's first"):
            forecast_boundary(day, "2026-11-01", "08:00")
        overnight = [1793548800 + 15 * 3600 + tick * 300 for tick in range(36)]
        self.assertEqual(forecast_boundary({"timestamps": overnight, "tokens": [[0]] * len(overnight)}, "2026-11-01", "00:30"), (18, overnight[18]))

    def test_remainder_of_day_is_generated_without_consuming_later_observations(self):
        model = WaitPredictor(1, embedding_size=4, hidden_size=8)
        rides = [{"id": "a", "name": "A"}]
        timestamps = [1793548800 + tick * 300 for tick in range(40)]
        day = {"rides": rides, "timestamps": timestamps, "tokens": [[0]] * 40}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            torch.save({"version": 1, "model_config": {"num_rides": 1, "embedding_size": 4, "hidden_size": 8},
                        "model_state": model.state_dict(), "rides": rides, "token_labels": TOKEN_LABELS,
                        "park_id": PARK_ID}, checkpoint)
            settings = SimpleNamespace(
                STEPS=None, SAMPLES=3, TEMPERATURE=1.0, SEED=42, DEVICE="cpu", CHECKPOINT=checkpoint,
                HISTORY=None, DATASET=root / "data.json.gz", PREDICT_DAY="2026-11-01", PREDICT_START_TIME="10:00",
            )
            save_dataset(settings.DATASET, {"version": 1, "park_id": PARK_ID, "days": {"2026-11-01": day}})
            with redirect_stdout(io.StringIO()):
                first = predict(settings)
            self.assertEqual(len(first["snapshots"]), 16)
            self.assertEqual(first["snapshots"][0]["time"], datetime.fromtimestamp(timestamps[24], timezone.utc).isoformat())
            self.assertEqual(first["snapshots"][-1]["time"], datetime.fromtimestamp(timestamps[-1], timezone.utc).isoformat())
            # A requested time rolls through every intermediate prediction even
            # when the configured output horizon is shorter.
            settings.STEPS = 2
            output = io.StringIO()
            with redirect_stdout(output):
                selected = predict(settings, target_time="11:00")
            self.assertEqual(selected["snapshots"], [first["snapshots"][12]])
            self.assertIn("\n2026-11-01 11:00 PST\n", output.getvalue())
            self.assertNotIn("\n2026-11-01 10:00 PST\n", output.getvalue())
            with redirect_stdout(io.StringIO()):
                rounded = predict(settings, target_time="11:03")
            self.assertEqual(rounded["snapshots"], [first["snapshots"][13]])
            with self.assertRaisesRegex(ValueError, "at or after the prediction start"):
                predict(settings, target_time="09:55")
            settings.STEPS = None
            day["tokens"][24:] = [[REFURBISHMENT]] * 16
            save_dataset(settings.DATASET, {"version": 1, "park_id": PARK_ID, "days": {"2026-11-01": day}})
            with redirect_stdout(io.StringIO()):
                second = predict(settings)
            self.assertEqual(second, first)
            self.assertEqual(set(root.iterdir()), {checkpoint, settings.DATASET})

    def test_training_can_explicitly_use_every_day_and_saves_the_final_epoch(self):
        for day_count in (1, 3):
            with self.subTest(day_count=day_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                days = {}
                for index in range(day_count):
                    rides = [{"id": "common-a", "name": "A"}, {"id": "common-b", "name": "B"},
                             {"id": f"new-{index}", "name": f"New {index}"}]
                    days[f"2026-11-{index + 1:02}"] = {
                        "rides": rides,
                        "timestamps": [1793548800 + index * 86400 + tick * 300 for tick in range(40 + index)],
                        "tokens": [[index + 1, DOWN, index + 3] for tick in range(40 + index)],
                    }
                settings = SimpleNamespace(
                    DATASET=root / "data.json.gz", CHECKPOINT=root / "model.pt", DEVICE="cpu", SEED=42,
                    EPOCHS=2, BATCH_SIZE=2, HIDDEN_SIZE=8, EMBEDDING_SIZE=4, LEARNING_RATE=0.001,
                    TRAIN_SPLIT=1.0, VAL_SPLIT=0.0, TEST_SPLIT=0.0,
                )
                save_dataset(settings.DATASET, {"version": 1, "park_id": PARK_ID, "days": days,
                                               "incomplete_days": {"2026-11-13": {
                                                   "rides": [{"id": "excluded", "name": "Partial-day ride"}],
                                                   "timestamps": [1794585600], "tokens": [[1]],
                                               }}})
                output = io.StringIO()
                with patch("ml.train.epoch", wraps=epoch) as training_epoch, redirect_stdout(output):
                    train(settings)
                self.assertEqual(training_epoch.call_count, 2)
                loader = training_epoch.call_args.args[1]
                self.assertEqual(len(loader.dataset), day_count)
                self.assertEqual([inputs[0, 0].item() for inputs, _, _ in loader.dataset], list(range(1, day_count + 1)))
                checkpoint = torch.load(settings.CHECKPOINT, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["train_dates"], sorted(days))
                self.assertEqual(checkpoint["epoch"], settings.EPOCHS)
                self.assertEqual(len(checkpoint["rides"]), day_count + 2)
                self.assertEqual(checkpoint["validation_dates"], [])
                self.assertEqual(checkpoint["test_dates"], [])
                self.assertIsNone(checkpoint["validation_loss"])
                self.assertNotIn("val NLL", output.getvalue())
                for name, value in training_epoch.call_args.args[0].state_dict().items():
                    torch.testing.assert_close(checkpoint["model_state"][name], value)
                self.assertNotIn("Forecast for", output.getvalue())
                self.assertEqual(set(root.iterdir()), {settings.DATASET, settings.CHECKPOINT})

    def test_closed_unknown_and_refurbishment_targets_have_zero_gradient(self):
        logits = torch.randn(1, 1, 5, VOCAB_SIZE, requires_grad=True)
        loss, count = token_loss(logits, torch.tensor([[[0, DOWN, CLOSED, REFURBISHMENT, UNKNOWN]]]))
        loss.backward()
        self.assertEqual(count.item(), 2)
        self.assertGreater(logits.grad[..., :2, :].abs().sum().item(), 0)
        self.assertEqual(logits.grad[..., 2:, :].abs().sum().item(), 0)
        empty_logits = torch.randn(1, 1, 1, VOCAB_SIZE, requires_grad=True)
        empty_loss, count = token_loss(empty_logits, torch.tensor([[[CLOSED]]]))
        self.assertEqual(count.item(), 0)
        self.assertEqual(empty_loss.item(), 0)
        empty_loss.backward()
        self.assertEqual(empty_logits.grad.abs().sum().item(), 0)

    def test_refurbishment_stays_locked_and_predictions_feed_the_next_tick(self):
        model = WaitPredictor(3, embedding_size=4, hidden_size=8)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-100)
            # Always DOWN for learned outputs, plus locked and unobserved rides.
            model.output.bias.reshape(3, VOCAB_SIZE)[:, DOWN] = 100
        history = torch.tensor([[0, REFURBISHMENT, UNKNOWN], [5, REFURBISHMENT, UNKNOWN]])
        features = torch.tensor(time_features([300, 600, 900, 1200]), dtype=torch.float32)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            paths = model.forecast(history, features, steps=3, samples=4)
        self.assertEqual(paths.shape, (4, 3, 3))
        self.assertTrue((paths[:, :, 0] == DOWN).all())
        self.assertTrue((paths[:, :, 1] == REFURBISHMENT).all())
        self.assertTrue((paths[:, :, 2] == UNKNOWN).all())
        self.assertEqual(forward.call_count, 3)
        self.assertTrue(torch.equal(forward.call_args_list[1].args[0][:, 0], paths[:, 0]))

    def test_training_is_causal_and_calendar_features_are_for_the_target_tick(self):
        model = WaitPredictor(1, embedding_size=4, hidden_size=8).eval()
        features = torch.tensor([time_features([300, 600, 900])], dtype=torch.float32)
        left, _ = model(torch.tensor([[[1], [2], [3]]]), features)
        right, _ = model(torch.tensor([[[1], [2], [DOWN]]]), features)
        torch.testing.assert_close(left[:, :2], right[:, :2])
        self.assertEqual(left.shape, (1, 3, 1, VOCAB_SIZE))

    def test_summary_keeps_down_separate_from_conditional_waits_and_overflow(self):
        paths = torch.tensor([[[DOWN]], [[OVERFLOW]], [[2]], [[2]]])
        row = summarize(paths, [{"id": "a", "name": "A"}], [0])[0]["rides"][0]
        self.assertEqual(row["down_probability"], 0.25)
        self.assertEqual(row["operating_probability"], 0.75)
        self.assertEqual(row["wait_minutes"], {"p10": 10, "p50": 10, "p90": "300+"})


if __name__ == "__main__":
    unittest.main()
