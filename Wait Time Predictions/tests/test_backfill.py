"""History reconstruction must preserve time boundaries and missing data."""

import unittest
from datetime import datetime

from ml.backfill import PARK_ID, reconstruct
from ml.data import CLOSED, DOWN, MISSING, UNKNOWN, validate_day


def tick(value):
    return int(datetime.fromisoformat(value).timestamp())


def document(day, opening, history):
    return {"id": PARK_ID, "entityType": "PARK", "range": {"from": day, "to": day},
            "entities": [{"id": "ride", "name": "Ride", "entityType": "ATTRACTION",
                          "opening": opening, "history": history}]}


def state(time, status="OPERATING", wait=10, **extra):
    return {"time": time, "status": status, "queue": {"STANDBY": {"waitTime": wait}}, **extra}


class BackfillTests(unittest.TestCase):
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
