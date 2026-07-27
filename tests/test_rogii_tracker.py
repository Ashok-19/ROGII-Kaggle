import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rogii_tool", ROOT / "tools" / "rogii.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrackerTests(unittest.TestCase):
    def test_seed_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.sqlite"
            tracker = MODULE.Tracker(ROOT, db)
            tracker.init(reset=True)
            state = tracker.state()
            self.assertEqual(state["kpis"]["db_integrity"], "ok")
            self.assertEqual(state["meta"]["discussion_topics_archived"], "132")
            self.assertGreaterEqual(len(state["submissions"]), 1)
            self.assertAlmostEqual(state["kpis"]["leader_score"], 4.679)
            self.assertAlmostEqual(state["leaderboard_bands"][0]["score"], 4.679)
            self.assertEqual(state["leaderboard_bands"][0]["team"], "shu01")

    def test_validation_archive_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = MODULE.Tracker(ROOT, Path(tmp) / "validation.sqlite")
            result = tracker.validate()
            self.assertEqual(result["status"], "ok", result)
            self.assertFalse(result["errors"])

    def test_metric_parser(self):
        self.assertEqual(MODULE.parse_metric_arg("cv:rmse=7.1"), {"split": "cv", "name": "rmse", "value": 7.1})
        self.assertEqual(MODULE.parse_metric_arg("rmse=8"), {"split": "cv", "name": "rmse", "value": 8.0})


if __name__ == "__main__":
    unittest.main()
