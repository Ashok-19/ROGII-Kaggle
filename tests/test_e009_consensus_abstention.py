import csv
import gzip
import io
import json
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.consensus_abstention import (
    ALL_CANDIDATES,
    ELIGIBLE,
    RULES,
    _apply_rule,
    _consensus_evidence,
    _deduplicate_predictions,
    _deranged_bank,
    _feature_selector,
    _fit_bank,
    _select_nested_rule,
    _sign_flipped_bank,
    _iter_parent_wells,
    run_e009,
    validate_e009_config,
)
from rogii_validation.harness import DataValidationError, scan_profiles
from rogii_validation.residual_action import ActionSufficient


class E009ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "experiments/E009/config.json").read_text())
        self.rules = {row["name"]: row for row in self.config["rules"]}

    def test_frozen_config_and_candidate_order(self):
        validate_e009_config(self.config)
        self.assertEqual(tuple(self.config["candidate_order"]), ALL_CANDIDATES)
        self.assertEqual(tuple(self.config["eligible_candidates"]), ELIGIBLE)
        self.assertEqual(tuple(row["name"] for row in self.config["rules"]), RULES)
        self.assertEqual(len(self.config["models"]), 6)
        broken = json.loads(json.dumps(self.config))
        broken["models"][0]["alpha"] = 0
        with self.assertRaises(DataValidationError):
            validate_e009_config(broken)

    def test_consensus_edge_cases_exactly_abstain(self):
        strict = self.rules["consensus_strict"]
        cases = [
            ([0, 0, 0, 0, 0, 0], "small_action"),
            ([0.5, 0.6, 0.4, 0.7, 0.8, 0.9], "small_action"),
            ([4, 4, 4, -4, -4, -4], "split_sign"),
            ([2, 2, 2, 2, 20, 20], "relative_dispersion"),
            ([2, 2, 2, 2, float("nan"), 2], "invalid_model_output"),
        ]
        for values, reason in cases:
            action, evidence, actual = _apply_rule(values, strict)
            self.assertEqual(action, 0.0)
            self.assertEqual(actual, reason)
        action, evidence, reason = _apply_rule([4, 4.2, 3.8, 4.1, 3.9, 4.0], strict)
        self.assertNotEqual(action, 0.0)
        self.assertEqual(reason, "action")
        self.assertTrue(evidence.valid)

    def test_absolute_dispersion_rule_and_duplicate_invariance(self):
        tight = self.rules["consensus_tight_absolute"]
        action, _, reason = _apply_rule([5, 5, 5, 20, 20, 20], tight)
        self.assertEqual(action, 0.0)
        self.assertIn(reason, {"split_sign", "absolute_dispersion"})
        values = [3.0, 3.2, 2.8, 3.1, 2.9, 3.05]
        first = _apply_rule(values, self.rules["consensus_strict"])[0]
        second = _apply_rule(values + [values[0]], self.rules["consensus_strict"])[0]
        self.assertEqual(first, second)
        self.assertEqual(len(_deduplicate_predictions([1, 1, 1 + 1e-13, 2])), 2)

    def test_consensus_evidence_nonfinite_and_sign_fraction(self):
        invalid = _consensus_evidence([1, float("inf"), 2])
        self.assertFalse(invalid.valid)
        item = _consensus_evidence([2, 3, 4, 5, -1, -2])
        self.assertTrue(item.valid)
        self.assertAlmostEqual(item.sign_fraction, 4 / 6)
        self.assertGreaterEqual(item.relative_mad, 0)

    def test_feature_family_selectors_are_distinct_and_legal(self):
        names = [
            "visible_tvt_mean", "hidden_z_mean", "md_hidden_span", "e006_path_mean",
            "e007_visible_gr_coverage", "selfcorr_raw_minus_e006_mean", "backtest_0p5_rmse",
        ]
        all_names = _feature_selector(names, "all")
        no_e007 = _feature_selector(names, "no_e007")
        path = _feature_selector(names, "path_evidence")
        visible = _feature_selector(names, "visible_geometry")
        self.assertGreater(len(all_names), len(no_e007))
        self.assertTrue(all(not name.startswith(("e007_", "selfcorr_")) for name in no_e007))
        self.assertIn("e006_path_mean", path)
        self.assertNotIn("e006_path_mean", visible)
        with self.assertRaises(DataValidationError):
            _feature_selector(names, "unknown")

    @staticmethod
    def toy_records(count=30):
        records = {}
        for index in range(count):
            well_id = f"{index + 1:08x}"
            datum = (index - count / 2) * 0.3 + math.sin(index) * 0.2
            truth = [10 + index + datum]
            base = [10 + index]
            sufficient = ActionSufficient.from_paths(well_id, truth, base)
            records[well_id] = {
                "features": {
                    "visible_tvt_mean": float(index),
                    "hidden_z_mean": float(index % 7),
                    "md_hidden_span": float(index * index),
                    "e006_path_mean": float(index) * 0.5,
                    "e007_visible_gr_coverage": 0.5 + 0.01 * index,
                    "selfcorr_raw_minus_e006_mean": math.sin(index / 3),
                    "backtest_0p5_rmse": abs(math.cos(index / 4)),
                },
                "targets": {"residual_datum_ft": sufficient.datum_target},
                "sufficient": sufficient,
            }
        return records

    def test_all_six_model_branches_fit_and_are_deterministic(self):
        records = self.toy_records()
        ids = sorted(records)
        features = sorted(records[ids[0]]["features"])
        first, models = _fit_bank(records=records, train_ids=ids[:20], test_ids=ids[20:], feature_names=features, config=self.config)
        second, _ = _fit_bank(records=records, train_ids=ids[:20], test_ids=ids[20:], feature_names=features, config=self.config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(set(first), {row["name"] for row in self.config["models"]})
        self.assertTrue(all(set(values) == set(ids[20:]) for values in first.values()))
        self.assertTrue(all(math.isfinite(value) and abs(value) <= 30 for values in first.values() for value in values.values()))
        self.assertTrue(all(model.selected_names for model in models.values()))

    def test_nested_selection_exercises_action_and_fallback(self):
        records = self.toy_records(20)
        ids = sorted(records)
        perfect = {name: {well_id: records[well_id]["targets"]["residual_datum_ft"] for well_id in ids} for name in RULES}
        selected, rows = _select_nested_rule(records, perfect, ids, self.config)
        self.assertIn(selected, RULES)
        self.assertEqual(sum(bool(row["selected"]) for row in rows), 1)
        bad = {name: {well_id: -100.0 for well_id in ids} for name in RULES}
        selected_bad, rows_bad = _select_nested_rule(records, bad, ids, self.config)
        self.assertIsNone(selected_bad)
        self.assertTrue(next(row for row in rows_bad if row["rule"] == "e006_fallback")["selected"])

    def test_shuffle_and_sign_flip_are_deterministic_and_nonmutating(self):
        ids = [f"{index + 1:08x}" for index in range(8)]
        bank = {row["name"]: {well_id: float(index + model_index) for index, well_id in enumerate(ids)} for model_index, row in enumerate(self.config["models"])}
        first = _deranged_bank(bank, ids, "ctx")
        second = _deranged_bank(bank, ids, "ctx")
        self.assertEqual(first, second)
        self.assertNotEqual(first, bank)
        flipped = _sign_flipped_bank(bank, self.config)
        names = [row["name"] for row in self.config["models"]]
        self.assertEqual(flipped[names[0]], bank[names[0]])
        self.assertEqual(flipped[names[1]][ids[3]], -bank[names[1]][ids[3]])

    def test_one_row_suffix_sufficient_statistic(self):
        sufficient = ActionSufficient.from_paths("aaaaaaaa", [12.0], [10.0])
        self.assertEqual(sufficient.trend_target, 0.0)
        self.assertAlmostEqual(sufficient.metric(2.0, 0.0).rmse, 0.0)

    def test_parent_reader_rejects_noncanonical_and_noncontiguous_rows(self):
        fields = ["id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e006_nested_fusion"]
        rows = [
            {"id":"aaaaaaaa_2","well_id":"aaaaaaaa","row_index":2,"hidden_index":0,"target":10,"last_known_tvt":9,"e006_nested_fusion":9.5},
            {"id":"aaaaaaaa_3","well_id":"aaaaaaaa","row_index":3,"hidden_index":1,"target":11,"last_known_tvt":9,"e006_nested_fusion":10.5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.gz"
            with path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                        writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                        writer.writeheader(); writer.writerows(rows)
            parsed = list(_iter_parent_wells(path))
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0][0], "aaaaaaaa")
            rows[1]["hidden_index"] = 3
            with path.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                        writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                        writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(DataValidationError):
                list(_iter_parent_wells(path))

    @staticmethod
    def _write_gzip(path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)

    def _build_full_fixture(self, root: Path):
        train_dir = root / "data/train"
        train_dir.mkdir(parents=True)
        (root / "folds").mkdir()
        (root / "experiments/E008/results").mkdir(parents=True)
        (root / "artifacts/E008").mkdir(parents=True)
        (root / "experiments/E009").mkdir(parents=True)
        well_ids = [f"{index + 1:08x}" for index in range(20)]
        parent_rows = []
        feature_rows = []
        target_rows = []
        for well_index, well_id in enumerate(well_ids):
            known = 12
            total = 18
            datum = 0.45 * (well_index - 9.5) + 0.2 * math.sin(well_index)
            base_last = 1000.0 + 2.5 * well_index + 0.15 * (known - 1)
            horizontal_rows = []
            residual_values = []
            for row_index in range(total):
                md = 100.0 + row_index
                x = 500.0 * well_index + row_index * (1.0 + 0.01 * well_index)
                y = 200.0 * well_index + row_index * 0.3 * ((well_index % 3) - 1)
                z = 2000.0 + 0.05 * row_index + 0.002 * well_index * row_index
                gr = 50.0 + 4.0 * math.sin((row_index + well_index) / 2.5)
                visible_tvt = 1000.0 + 2.5 * well_index + 0.15 * row_index
                if row_index < known:
                    tvt = visible_tvt
                    tvt_input = f"{tvt:.8f}"
                else:
                    hidden_index = row_index - known
                    e006 = base_last + 0.18 * (hidden_index + 1) + 0.01 * well_index * hidden_index
                    tvt = e006 + datum + 0.05 * math.sin(hidden_index + well_index)
                    residual_values.append(tvt - e006)
                    tvt_input = ""
                    parent_rows.append({
                        "id": f"{well_id}_{row_index}", "well_id": well_id,
                        "row_index": row_index, "hidden_index": hidden_index,
                        "target": f"{tvt:.8f}", "last_known_tvt": f"{base_last:.8f}",
                        "e006_nested_fusion": f"{e006:.8f}",
                    })
                horizontal_rows.append([md, x, y, z, tvt, gr, tvt_input])
            with (train_dir / f"{well_id}__horizontal_well.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["MD", "X", "Y", "Z", "TVT", "GR", "TVT_input"])
                writer.writerows(horizontal_rows)
            with (train_dir / f"{well_id}__typewell.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["TVT", "GR"])
                for row_index in range(16):
                    writer.writerow([970 + 4 * row_index, 45 + well_index * 0.2 + math.sin(row_index)])
            feature_rows.append({
                "well_id": well_id,
                "visible_tvt_mean": 1000.0 + 2.5 * well_index,
                "hidden_z_mean": 2000.0 + 0.08 * well_index,
                "md_hidden_span": 6.0 + 0.1 * well_index,
                "e006_path_mean": base_last + 0.8,
                "e007_visible_gr_coverage": 0.70 + 0.01 * (well_index % 10),
                "selfcorr_raw_minus_e006_mean": math.sin(well_index / 3),
                "backtest_0p5_rmse": abs(math.cos(well_index / 4)),
                "hidden_rows": float(total - known),
                "hidden_gr_missing_fraction": 0.05 * (well_index % 4),
            })
            target_rows.append({"well_id": well_id, "residual_datum_ft": sum(residual_values) / len(residual_values)})
        _, data = scan_profiles(train_dir)
        for version in range(5):
            fold = {
                "schema_version": 1, "version": f"v{version + 1}", "seed": 900 + version,
                "strategy": "fixture", "data_signature": data["data_signature"],
                "fingerprint": hashlib.sha256(f"e009-fixture-{version}".encode()).hexdigest(),
                "n_folds": 5, "well_count": len(well_ids),
                "assignments": {well_id: (index + version) % 5 for index, well_id in enumerate(well_ids)},
                "fold_summary": [],
            }
            (root / "folds" / f"v{version + 1}.json").write_text(json.dumps(fold), encoding="utf-8")
        feature_path = root / "experiments/E008/results/legal_features.csv"
        with feature_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(feature_rows)
        target_path = root / "experiments/E008/results/residual_targets.csv"
        with target_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(target_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(target_rows)
        oof_path = root / "artifacts/E008/oof_predictions.csv.gz"
        self._write_gzip(oof_path, list(parent_rows[0]), parent_rows)
        summary_path = root / "experiments/E008/results/summary.json"
        summary_path.write_text('{"fixture": true}\n', encoding="utf-8")
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        def rmse(column):
            return math.sqrt(sum((float(row[column]) - float(row["target"])) ** 2 for row in parent_rows) / len(parent_rows))
        config = json.loads(json.dumps(self.config))
        config["expected_wells"] = len(well_ids)
        config["data_signature"] = data["data_signature"]
        config["parents"].update({
            "e008_legal_features": "experiments/E008/results/legal_features.csv",
            "e008_legal_features_sha256": sha(feature_path),
            "e008_residual_targets": "experiments/E008/results/residual_targets.csv",
            "e008_residual_targets_sha256": sha(target_path),
            "e008_oof": "artifacts/E008/oof_predictions.csv.gz",
            "e008_oof_sha256": sha(oof_path),
            "e008_summary": "experiments/E008/results/summary.json",
            "e008_summary_sha256": sha(summary_path),
            "e006_expected_rmse": rmse("e006_nested_fusion"),
            "last_known_expected_rmse": rmse("last_known_tvt"),
        })
        config["stress"].update({"spatial_bins": 2, "typewell_clusters": 2})
        config["controls"].update({
            "maximum_shuffled_gain_vs_e006": 999.0,
            "maximum_sign_flipped_gain_vs_e006": 999.0,
            "oracle_minimum_gain_vs_best_legal": -999.0,
            "minimum_nested_action_fraction": 0.0,
            "maximum_nested_action_fraction": 1.0,
        })
        config["promotion"].update({
            "minimum_gain_vs_e006": -999.0, "minimum_gain_vs_last_known": -999.0,
            "minimum_map_wins": 0, "minimum_map_gain": -999.0, "minimum_outer_cell_wins": 0,
            "maximum_p90_deterioration_vs_e006": 999.0, "maximum_p90_deterioration_vs_last_known": 999.0,
            "maximum_worst5_sse_share_increase_vs_e006": 1.0,
            "maximum_worst5_sse_share_increase_vs_last_known": 1.0,
            "require_positive_spatial_group_gain_vs_e006": False,
            "require_positive_typewell_group_gain_vs_e006": False,
            "maximum_special_slice_deterioration_vs_e006": 999.0,
        })
        (root / "experiments/E009/config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return config

    def test_full_fixture_run_is_deterministic_complete_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._build_full_fixture(root)
            summaries = []
            for label in ("first", "second"):
                summaries.append(run_e009(
                    root=root, train_dir=root / "data/train",
                    output_dir=root / label / "results", artifact_dir=root / label / "artifacts",
                    config=config, code_sha="e" * 40,
                ))
            self.assertEqual(summaries[0]["reported_candidate"], summaries[1]["reported_candidate"])
            self.assertEqual(summaries[0]["candidate_metrics"], summaries[1]["candidate_metrics"])
            self.assertEqual(summaries[0]["branch_metrics"], summaries[1]["branch_metrics"])
            first = root / "first/results"
            second = root / "second/results"
            dynamic = {"summary.json", "control_metrics.csv", "artifact_manifest.json"}
            for path in first.iterdir():
                if path.name not in dynamic:
                    self.assertEqual(path.read_bytes(), (second / path.name).read_bytes(), path.name)
            self.assertEqual(
                (root / "first/artifacts/oof_predictions.csv.gz").read_bytes(),
                (root / "second/artifacts/oof_predictions.csv.gz").read_bytes(),
            )
            self.assertTrue(summaries[0]["controls"]["parent_artifacts"]["pass"])
            self.assertTrue(summaries[0]["controls"]["data_integrity"]["pass"])
            self.assertTrue(summaries[0]["controls"]["oof_identity"]["pass"])
            self.assertTrue(summaries[0]["controls"]["pooled_sse_consistency"]["pass"])
            manifest = json.loads((first / "artifact_manifest.json").read_text())
            self.assertEqual(len(manifest["parent_artifacts"]), 4)
            self.assertEqual(len(manifest["fold_files"]), 5)
            self.assertGreaterEqual(len(manifest["files"]), 20)
            broken = json.loads(json.dumps(config))
            broken["parents"]["e008_summary_sha256"] = "0" * 64
            with self.assertRaises(DataValidationError):
                run_e009(
                    root=root, train_dir=root / "data/train",
                    output_dir=root / "broken/results", artifact_dir=root / "broken/artifacts",
                    config=broken, code_sha="e" * 40,
                )

    def test_official_artifact_hashes_when_available(self):
        manifest_path = ROOT / "experiments/E009/results/artifact_manifest.json"
        if not manifest_path.exists():
            self.skipTest("official E009 artifacts not generated yet")
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            path = ROOT / "experiments/E009/results" / item["path"]
            import hashlib
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
        for item in manifest["external_artifacts"]:
            path = ROOT / item["path"]
            import hashlib
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
