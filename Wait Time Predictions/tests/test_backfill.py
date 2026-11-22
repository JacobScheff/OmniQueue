"""History reconstruction must preserve time boundaries and missing data."""

import unittest
from datetime import datetime

from ml.backfill import PARK_ID, reconstruct
from ml.data import CLOSED, DOWN, MISSING, UNKNOWN, merge_history_prefix, validate_day


def tick(value):
    return int(datetime.fromisoformat(value).timestamp())


def document(day, opening, history):
    return {"id": PARK_ID, "entityType": "PARK", "range": {"from": day, "to": day},
            "entities": [{"id": "ride", "name": "Ride", "entityType": "ATTRACTION",
                          "opening": opening, "history": history}]}


def state(time, status="OPERATING", wait=10, **extra):
    return {"time": time, "status": status, "queue": {"STANDBY": {"waitTime": wait}}, **extra}


class BackfillTests(unittest.TestCase):
    def test_api_prefix_is_kept_but_can_never_fill_new_firebase_buckets(self):
        previous = {"history_before": 600, "rides": [{"id": "ride", "name": "Ride"}],
                    "timestamps": [0, 300, 600, 900], "tokens": [[1], [2], [50], [50]],
                    "interpolated_ticks": {"300": [0, 600], "900": [600, 1200]}}
        live = {"rides": [{"id": "ride", "name": "Renamed"}], "timestamps": [600, 1200],
                "tokens": [[3], [4]], "expected_timestamps": [0, 300, 600, 900, 1200]}
        result = merge_history_prefix(previous, live)
        self.assertEqual(result["timestamps"], [0, 300, 600, 1200])
        self.assertEqual(result["tokens"], [[1], [2], [3], [4]])
        self.assertEqual(result["rides"][0]["name"], "Renamed")
        self.assertNotIn("900", result["interpolated_ticks"])
        later = {**live, "timestamps": [600, 900, 1200], "tokens": [[3], [5], [4]]}
        refreshed = merge_history_prefix(result, later)
        self.assertEqual(refreshed["tokens"], [[1], [2], [3], [5], [4]])
        self.assertEqual(refreshed["history_before"], 600)
        self.assertEqual(refreshed["source"], "firestore_with_history_prefix")

    def test_regular_firestore_refresh_is_unchanged_without_an_api_prefix(self):
        live = {"rides": [], "timestamps": [], "tokens": []}
        self.assertIs(merge_history_prefix(None, live), live)
        self.assertIs(merge_history_prefix({"source": "firestore"}, live), live)

    def test_samples_as_of_tick_without_future_leakage(self):
        times = [tick(f"2026-11-09T08:{minute}:00-08:00") for minute in ("00", "05", "10")]
        opening = state("2026-11-09T00:00:00-08:00", "CLOSED", None,
                        observedAt="2026-11-08T23:00:00-08:00")
        changes = [state("2026-11-09T08:00:54-08:00", wait=20),
                   state("2026-11-09T08:10:00-08:00", "DOWN", None)]
        day = reconstruct(times, [document("2026-11-09", opening, changes)], {"ride"})
        self.assertEqual(day["tokens"], [[CLOSED], [4], [DOWN]])
        validate_day(day)

    def test_missing_opening_stays_missing_until_first_observation(self):
        times = [tick("2026-11-09T08:00:00-08:00"), tick("2026-11-09T08:05:00-08:00")]
        changes = [state("2026-11-09T08:01:00-08:00", wait=None)]
        day = reconstruct(times, [document("2026-11-09", {}, changes)], {"ride"})
        self.assertEqual(day["tokens"], [[MISSING], [UNKNOWN]])
        with self.assertRaisesRegex(ValueError, "Attractions missing"):
            validate_day(day)

    def test_operating_day_after_midnight_uses_next_calendar_opening(self):
        times = [tick("2026-11-09T23:55:00-08:00"), tick("2026-11-10T00:00:00-08:00")]
        opening = state("2026-11-09T00:00:00-08:00", wait=5,
                        observedAt="2026-11-08T23:00:00-08:00")
        overnight = state("2026-11-10T00:00:00-08:00", wait=15,
                          observedAt="2026-11-09T23:58:00-08:00")
        day = reconstruct(times, [document("2026-11-09", opening, []),
                                  document("2026-11-10", overnight, [])], {"ride"})
        self.assertEqual(day["tokens"], [[1], [3]])

    def test_degraded_opening_is_rejected(self):
        opening = state("2026-11-09T00:00:00-08:00", degraded=True)
        with self.assertRaisesRegex(ValueError, "Incomplete opening"):
            reconstruct([tick("2026-11-09T08:00:00-08:00")],
                        [document("2026-11-09", opening, [])], {"ride"})

    def test_entity_missing_on_next_calendar_day_is_not_invented(self):
        times = [tick("2026-11-09T23:55:00-08:00"), tick("2026-11-10T00:00:00-08:00")]
        opening = state("2026-11-09T00:00:00-08:00", observedAt="2026-11-08T23:00:00-08:00")
        missing = document("2026-11-10", {}, [])
        missing["entities"] = []
        day = reconstruct(times, [document("2026-11-09", opening, []), missing], {"ride"})
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            validate_day(day)


if __name__ == "__main__":
    unittest.main()
