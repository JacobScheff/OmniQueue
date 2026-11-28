"""Regression tests scoped to the one-time local wait history migration."""

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from ml.parks import get_park
from scripts.import_forecast_history import import_local


class ForecastHistoryImportTests(unittest.TestCase):
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
            with patch("scripts.import_forecast_history.archive_day") as archive:
                import_local([park], root, store=store)
        self.assertEqual(archive.call_count, 2)
        archive.assert_any_call(store, park.slug, "2026-11-19", paid, True)
        archive.assert_any_call(store, park.slug, "2026-11-21", mixed, False)
