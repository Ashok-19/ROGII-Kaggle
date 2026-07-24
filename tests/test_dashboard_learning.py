import csv
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rogii_tool_learning", ROOT / "tools" / "rogii.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LearningDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "dashboard").mkdir(parents=True)
        (self.root / "experiments/E001/results").mkdir(parents=True)
        (self.root / "data/train").mkdir(parents=True)
        (self.root / "dashboard/index.html").write_text("<html>control</html>", encoding="utf-8")
        (self.root / "dashboard/learn.html").write_text("<html>learning</html>", encoding="utf-8")
        (self.root / "dashboard/learning_content.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2026-07-18",
                    "title": "Learning",
                    "feature_groups": [{"group": "Geometry", "features": []}],
                    "breakthroughs": [{"id": "B001"}],
                }
            ),
            encoding="utf-8",
        )
        self._write_csv(
            self.root / "experiments/E001/results/data_profile.csv",
            [
                "well_id",
                "path",
                "total_rows",
                "known_rows",
                "hidden_rows",
                "last_visible_md",
                "last_visible_z",
                "last_visible_tvt",
            ],
            [["aaaaaaaa", "data/train/aaaaaaaa__horizontal_well.csv", 6, 3, 3, 102, -12, 22]],
        )
        self._write_csv(
            self.root / "experiments/E001/results/well_metrics.csv",
            [
                "well_id",
                "split",
                "rows_scored",
                "rmse",
                "mean_error",
                "sse",
                "regime",
                "datum_sse",
                "trend_sse",
                "shape_sse",
            ],
            [["aaaaaaaa", "cv", 3, 2.0, 1.0, 12.0, "short_hidden", 6.0, 3.0, 3.0]],
        )
        summary = {
            "actual_target_metrics": {
                "last_known_tvt": {
                    "rmse": 15.9,
                    "median_well_rmse": 2.0,
                    "worst_5pct_sse_share": 0.4,
                    "worst_10pct_sse_share": 0.5,
                }
            },
            "data": {"well_count": 1, "hidden_rows": 3, "known_rows": 3, "total_rows": 6},
            "controls": {"reference_baseline": {"pass": True}},
        }
        (self.root / "experiments/E001/results/summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        self._write_csv(
            self.root / "data/train/aaaaaaaa__horizontal_well.csv",
            ["MD", "X", "Y", "Z", "ANCC", "TVT", "GR", "TVT_input"],
            [
                [100, 0, 0, -10, -20, 20, 50, 20],
                [101, 1, 0, -11, -20.1, 21, 52, 21],
                [102, 2, 0, -12, -20.2, 22, 54, 22],
                [103, 3, 0, -13, -20.3, 23, 56, ""],
                [104, 4, 0, -14, -20.4, 24, 58, ""],
                [105, 5, 0, -15, -20.5, 25, 60, ""],
            ],
        )
        self._write_csv(
            self.root / "data/train/aaaaaaaa__typewell.csv",
            ["TVT", "GR", "Geology"],
            [[19, 48, ""], [21, 52, ""], [23, 56, ""], [25, 60, ""]],
        )
        self.tracker = MODULE.Tracker(self.root, self.root / "tracking/test.sqlite")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_csv(path: Path, fieldnames, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            writer.writerows(rows)

    def test_learning_state_builds_catalog_and_recommendations(self):
        state = self.tracker.learning_state()
        self.assertEqual(len(state["well_catalog"]), 1)
        item = state["well_catalog"][0]
        self.assertEqual(item["well_id"], "aaaaaaaa")
        self.assertAlmostEqual(item["datum_share"], 0.5)
        self.assertAlmostEqual(item["trend_share"], 0.25)
        self.assertEqual(state["recommended_wells"][0]["well_id"], "aaaaaaaa")
        self.assertEqual(state["e001"]["data"]["hidden_rows"], 3)
        self.assertIsNone(state["e003"])
        self.assertIsNone(state["e004"])
        self.assertIsNone(state["e005"])
        self.assertIsNone(state["e006"])
        self.assertIsNone(state["e007"])
        self.assertIsNone(state["e008"])
        self.assertIsNone(state["e009"])

    def test_learning_well_preserves_visibility_boundary(self):
        payload = self.tracker.learning_well("aaaaaaaa", max_points=4)
        self.assertEqual(payload["metadata"]["known_rows"], 3)
        self.assertEqual(payload["metadata"]["hidden_rows"], 3)
        self.assertIn(2, payload["original_indices"])
        self.assertIn(3, payload["original_indices"])
        sampled = dict(zip(payload["original_indices"], payload["series"]["TVT_input"]))
        self.assertEqual(sampled[2], 22.0)
        self.assertIsNone(sampled[3])
        self.assertEqual(payload["metadata"]["formation_names"], ["ANCC"])
        self.assertEqual(payload["e001_metrics"]["regime"], "short_hidden")
        self.assertEqual(payload["e003_metrics"], {})

    def test_learning_well_rejects_invalid_ids(self):
        with self.assertRaises(ValueError):
            self.tracker.learning_well("../../bad")
        with self.assertRaises(ValueError):
            self.tracker.learning_well("AAAAAAAA")

    def test_learning_http_routes(self):
        handler = type("TestLearningHandler", (MODULE.DashboardHandler,), {})
        handler.tracker = self.tracker
        handler.index_path = self.root / "dashboard/index.html"
        handler.learning_path = self.root / "dashboard/learn.html"
        server = MODULE.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urlopen(base + "/learn", timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(b"learning", response.read())
            with urlopen(base + "/api/learning", timeout=5) as response:
                payload = json.load(response)
                self.assertEqual(len(payload["well_catalog"]), 1)
            with urlopen(base + "/api/learning/well?well_id=aaaaaaaa", timeout=5) as response:
                payload = json.load(response)
                self.assertEqual(payload["metadata"]["hidden_start_md"], 103.0)
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/api/learning/well?well_id=bad", timeout=5)
            self.assertEqual(caught.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_repository_learning_assets_have_required_hooks(self):
        content = json.loads((ROOT / "dashboard/learning_content.json").read_text(encoding="utf-8"))
        html = (ROOT / "dashboard/learn.html").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(content["feature_groups"]), 5)
        self.assertGreaterEqual(len(content["breakthroughs"]), 4)
        self.assertEqual(content["updated_at"], "2026-07-25")
        self.assertTrue(any(item["id"] == "B005" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B006" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B007" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B008" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B009" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B010" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B011" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B012" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B013" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B014" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B015" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B016" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B017" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B018" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B019" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B020" for item in content["breakthroughs"]))
        self.assertTrue(any(item["id"] == "B021" for item in content["breakthroughs"]))
        self.assertTrue(any(item["signal"] == "E011 packaging" for item in content["idea_questions"]))
        self.assertTrue(any(item["signal"] == "PF branch uncertainty" for item in content["idea_questions"]))
        self.assertTrue(any(item["signal"] == "hidden-scale runtime" for item in content["idea_questions"]))
        self.assertTrue(any(item["signal"] == "regime specialists" for item in content["idea_questions"]))
        self.assertTrue(
            any(
                item["local_path"] == "archive/competition/knowledge_context_2026-07-22.md"
                for item in content["sources"]
            )
        )
        self.assertTrue(
            any(item["local_path"] == "experiments/E011/RESULT.md" for item in content["sources"])
        )
        self.assertTrue(
            any(
                item["local_path"] == "experiments/E011/verification.json"
                for item in content["sources"]
            )
        )
        self.assertIn("trend_transfer", html)
        self.assertIn("risk_action", html)
        self.assertIn("deployment_gap", html)
        self.assertIn("remote_parity", html)
        self.assertIn("gr_path_gate", html)
        self.assertIn("nested_fusion", html)
        self.assertIn("selfcorr_gate", html)
        self.assertIn("residual_action_gate", html)
        self.assertIn("consensus_gate", html)
        for hook in ("/api/learning", "wellSelect", "geometryPlot", "targetPlot", "grPlot", "presetE003", "e003ScoreLadder", "e004ScoreLadder", "e004ContractCards", "presetOracle"):
            self.assertIn(hook, html)
        self.assertIn("Learning simulator", html)
        self.assertIn("Illegal truth-fit demo", html)

    def test_repository_e003_learning_payload_is_promoted_but_not_deployed(self):
        tracker = MODULE.Tracker(ROOT, ROOT / "tracking/rogii.sqlite")
        state = tracker.learning_state()
        self.assertIsNotNone(state["e003"])
        self.assertEqual(state["e003"]["status"], "promoted")
        self.assertTrue(state["e003"]["decision"]["signed_action_authorized"])
        well_id = state["well_catalog"][0]["well_id"]
        payload = tracker.learning_well(well_id)
        self.assertEqual(payload["e003_metrics"]["evidence_label"], "cross_fitted_oof")
        self.assertFalse(payload["e003_metrics"]["deployment_ready"])

    def test_repository_e004_learning_payload_is_kaggle_verified_and_deployment_ready(self):
        tracker = MODULE.Tracker(ROOT, ROOT / "tracking/rogii.sqlite")
        state = tracker.learning_state()
        self.assertIsNotNone(state["e004"])
        self.assertEqual(state["e004"]["status"], "deployment_ready")
        self.assertEqual(state["e004"]["selected_candidate"], "geometry_prefix")
        self.assertAlmostEqual(state["e004"]["selected_candidate_metrics"]["rmse"], 15.491306398267565)
        self.assertTrue(state["e004"]["deployment"]["local_model_ready"])
        self.assertTrue(state["e004"]["deployment"]["local_notebook_parity"])
        self.assertTrue(state["e004"]["deployment"]["remote_kaggle_mcp_parity"])
        self.assertTrue(state["e004"]["deployment"]["deployment_ready"])
        self.assertEqual(
            state["e004"]["controls"]["remote_kaggle_mcp_parity"]["status"],
            "passed_byte_identical",
        )

    def test_repository_e005_learning_payload_records_strict_rejection(self):
        tracker = MODULE.Tracker(ROOT, ROOT / "tracking/rogii.sqlite")
        state = tracker.learning_state()
        self.assertIsNotNone(state["e005"])
        self.assertEqual(state["e005"]["status"], "rejected")
        self.assertEqual(state["e005"]["selected_candidate"], "pf_gr_path")
        self.assertAlmostEqual(state["e005"]["selected_candidate_metrics"]["rmse"], 15.350204071523788)
        self.assertFalse(state["e005"]["promotion"]["promoted"])
        self.assertFalse(state["e005"]["promotion"]["gates"]["p90_vs_e004"])
        self.assertFalse(state["e005"]["promotion"]["gates"]["repeated_maps"])
        self.assertFalse(state["e005"]["deployment"]["statistically_authorized"])
        self.assertFalse(state["e005"]["deployment"]["submission_made"])

    def test_repository_e006_learning_payload_is_promoted_and_deployment_ready(self):
        tracker = MODULE.Tracker(ROOT, ROOT / "tracking/rogii.sqlite")
        state = tracker.learning_state()
        self.assertIsNotNone(state["e006"])
        self.assertEqual(state["e006"]["status"], "deployment_ready")
        self.assertEqual(state["e006"]["selected_candidate"], "nested_conservative_grid")
        self.assertAlmostEqual(
            state["e006"]["candidate_metrics"]["nested_conservative_grid"]["rmse"],
            14.933140787236884,
        )
        self.assertEqual(state["e006"]["map_wins"]["nested_conservative_grid"], 5)
        self.assertEqual(state["e006"]["outer_cell_wins"]["nested_conservative_grid"], 25)
        self.assertTrue(state["e006"]["deployment"]["local_notebook_parity"])
        self.assertTrue(state["e006"]["deployment"]["private_internet_disabled_kaggle_parity"])
        self.assertTrue(state["e006"]["deployment"]["deployment_ready"])
        self.assertFalse(state["e006"]["deployment"]["submission_made"])

    def test_repository_e007_learning_payload_records_strict_rejection(self):
        tracker = MODULE.Tracker(ROOT, ROOT / "tracking/rogii.sqlite")
        state = tracker.learning_state()
        self.assertIsNotNone(state["e007"])
        self.assertEqual(state["e007"]["status"], "rejected")
        self.assertIsNone(state["e007"]["selected_candidate"])
        self.assertEqual(state["e007"]["reported_candidate"], "visible_fixed_0p10")
        self.assertAlmostEqual(
            state["e007"]["candidate_metrics"]["visible_fixed_0p10"]["rmse"],
            14.830478417885569,
        )
        self.assertEqual(state["e007"]["map_wins"]["visible_fixed_0p10"], 5)
        self.assertEqual(state["e007"]["outer_cell_wins"]["visible_fixed_0p10"], 20)
        gates = state["e007"]["gates_by_candidate"]["visible_fixed_0p10"]
        self.assertFalse(gates["p90_vs_e006"])
        self.assertFalse(gates["spatial_stress"])
        self.assertFalse(gates["typewell_stress"])
        self.assertFalse(state["e007"]["deployment"]["statistically_authorized"])
        self.assertFalse(state["e007"]["deployment"]["submission_made"])

    def test_repository_e008_learning_payload_records_strict_rejection(self):
        tracker = MODULE.Tracker(ROOT, ROOT / "tracking/rogii.sqlite")
        state = tracker.learning_state()
        self.assertIsNotNone(state["e008"])
        self.assertEqual(state["e008"]["status"], "rejected")
        self.assertIsNone(state["e008"]["selected_candidate"])
        self.assertEqual(state["e008"]["reported_candidate"], "ridge_residual_datum")
        self.assertAlmostEqual(
            state["e008"]["candidate_metrics"]["ridge_residual_datum"]["rmse"],
            14.795138041468634,
        )
        self.assertEqual(state["e008"]["map_wins"]["ridge_residual_datum"], 2)
        self.assertEqual(state["e008"]["outer_cell_wins"]["ridge_residual_datum"], 14)
        self.assertTrue(state["e008"]["controls"]["e007_evidence"]["pass"])
        gates = state["e008"]["gates_by_candidate"]["ridge_residual_datum"]
        self.assertFalse(gates["repeated_maps"])
        self.assertFalse(gates["outer_cells"])
        self.assertFalse(gates["spatial_stress"])
        self.assertFalse(gates["typewell_stress"])
        self.assertFalse(state["e008"]["deployment"]["statistically_authorized"])
        self.assertFalse(state["e008"]["deployment"]["submission_made"])


    def test_repository_e009_learning_payload_records_strict_rejection(self):
        tracker = MODULE.Tracker(ROOT, ROOT / "tracking/rogii.sqlite")
        state = tracker.learning_state()
        self.assertIsNotNone(state["e009"])
        self.assertEqual(state["e009"]["status"], "rejected")
        self.assertIsNone(state["e009"]["selected_candidate"])
        self.assertEqual(state["e009"]["reported_candidate"], "consensus_majority")
        self.assertAlmostEqual(
            state["e009"]["candidate_metrics"]["consensus_majority"]["rmse"],
            14.793453219822904,
        )
        self.assertEqual(state["e009"]["map_wins"]["consensus_majority"], 5)
        self.assertEqual(state["e009"]["outer_cell_wins"]["consensus_majority"], 15)
        gates = state["e009"]["gates_by_candidate"]["consensus_majority"]
        self.assertFalse(gates["outer_cells"])
        self.assertFalse(gates["spatial_stress"])
        self.assertFalse(gates["typewell_stress"])
        self.assertFalse(gates["controls"])
        self.assertFalse(state["e009"]["controls"]["sign_flipped"]["pass"])
        self.assertAlmostEqual(
            state["e009"]["branch_metrics"]["ridge_all_a25_f64"]["rmse"],
            14.635273314979647,
        )
        self.assertFalse(state["e009"]["deployment"]["statistically_authorized"])
        self.assertFalse(state["e009"]["deployment"]["submission_made"])


if __name__ == "__main__":
    unittest.main()
