import unittest

from api.collect import pack_snapshot
from view import decode


class DecodeTests(unittest.TestCase):
    def test_catalog_order_zero_wait_and_closure_status(self):
        _, catalog, packed = pack_snapshot([
            {"id": "b", "name": "Closed ride", "entityType": "ATTRACTION", "status": "REFURBISHMENT"},
            {"id": "a", "name": "Open ride", "entityType": "ATTRACTION", "status": "OPERATING",
             "queue": {"STANDBY": {"waitTime": 0}}},
        ])
        rows = decode({"rides": packed}, catalog)
        self.assertEqual(rows, [
            {"id": "a", "name": "Open ride", "status": "OPERATING", "wait": 0},
            {"id": "b", "name": "Closed ride", "status": "REFURBISHMENT", "wait": None},
        ])

    def test_mismatched_catalog_and_unknown_format_fail_explicitly(self):
        with self.assertRaises(ValueError):
            decode({"rides": b"\x00\x00\x01"}, {"encoding_version": 1, "attractions": []})
        with self.assertRaises(ValueError):
            decode({"rides": b""}, {"encoding_version": 2, "attractions": []})
