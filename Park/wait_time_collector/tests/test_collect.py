import struct
import unittest

from api import collect


def attraction(ride_id, status, wait=None):
    return {
        "id": ride_id,
        "name": f"Ride {ride_id}",
        "entityType": "ATTRACTION",
        "status": status,
        "queue": {"STANDBY": {"waitTime": wait}},
    }


class SnapshotFormatTests(unittest.TestCase):
    def test_records_follow_catalog_order_and_preserve_wait_and_status(self):
        _, catalog, rides = collect.pack_snapshot([
            attraction("f", "REFURBISHMENT"),
            attraction("e", "CLOSED"),
            attraction("d", "DOWN", 15),
            attraction("c", "OPERATING"),
            attraction("b", "OPERATING", 300),
            attraction("a", "OPERATING", 0),
            {"entityType": "SHOW"},
        ])

        self.assertEqual(catalog["encoding_version"], 1)
        self.assertEqual([ride["id"] for ride in catalog["attractions"]], list("abcdef"))
        self.assertEqual(len(rides), 18)
        self.assertEqual(list(struct.iter_unpack("<HB", rides)), [
            (0, 1), (300, 1), (65535, 1), (65535, 2), (65535, 3), (65535, 4),
        ])

    def test_catalog_version_depends_on_ride_metadata_and_not_wait_times(self):
        a = attraction("a", "OPERATING", 10)
        b = attraction("b", "CLOSED")
        version, _, _ = collect.pack_snapshot([a, b])

        self.assertEqual(collect.pack_snapshot([b, attraction("a", "DOWN")])[0], version)
        self.assertNotEqual(collect.pack_snapshot([a])[0], version)
        self.assertNotEqual(collect.pack_snapshot([a, {**b, "name": "Renamed"}])[0], version)

    def test_reserved_wait_value_and_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            collect.pack_snapshot([attraction("a", "OPERATING", 65535)])
        with self.assertRaises(ValueError):
            collect.pack_snapshot([attraction("a", "OPERATING"), attraction("a", "CLOSED")])
