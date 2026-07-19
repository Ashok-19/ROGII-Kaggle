import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.harness import DataValidationError, scan_profiles
from rogii_validation.residual_action import (
    ALL_CANDIDATES,
    ActionSufficient,
    _audit_parent_alignment,
    _build_contexts,
    _deranged_targets,
    _duplicate_predictions,
    _fit_predict,
    _prepare_model,
    _read_e007_diagnostics,
    _select_scale,
    _soft_cap,
    _validate_feature_names,
    run_e008,
    validate_e008_config,
)


class E008ResidualActionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "experiments/E008/config.json").read_text(encoding="utf-8"))

    def test_frozen_config_and_candidate_order(self):
        validate_e008_config(self.config)
        self.assertEqual(tuple(self.config["candidate_order"]), ALL_CANDIDATES)
        self.assertEqual(self.config["ridge"]["alpha"], 25.0)
        self.assertEqual(self.config["feature_contract"]["maximum_screened_features"], 32)
        self.assertEqual(self.config["targets"]["datum_soft_cap_ft"], 30.0)
        self.assertEqual(self.config["targets"]["trend_soft_cap_ft"], 60.0)
        self.assertTrue(self.config["feature_contract"]["exclude_formation_surfaces"])
        self.assertTrue(self.config["feature_contract"]["exclude_direct_typewell_features"])
        self.assertTrue(self.config["feature_contract"]["exclude_absolute_spatial_coordinates"])

    def test_action_sufficient_matches_direct_paths_and_oracle_targets(self):
        truth = [10.0, 12.0, 15.0, 19.0, 24.0]
        base = [11.0, 11.5, 14.0, 18.5, 22.0]
        sufficient = ActionSufficient.from_paths("aaaaaaaa", truth, base)
        for datum, trend in ((0.0, 0.0), (1.25, -3.5), (-2.0, 5.0)):
            prediction = [
                anchor + datum + trend * (index / (len(base) - 1) - 0.5)
                for index, anchor in enumerate(base)
            ]
            direct = math.sqrt(sum((value - target) ** 2 for value, target in zip(prediction, truth)) / len(truth))
            self.assertAlmostEqual(sufficient.metric(datum, trend).rmse, direct, places=12)
        oracle = sufficient.metric(sufficient.datum_target, sufficient.trend_target)
        self.assertLessEqual(oracle.rmse, sufficient.metric(0.0, 0.0).rmse)
        with self.assertRaises(DataValidationError):
            ActionSufficient.from_paths("bad", [], [])
        with self.assertRaises(DataValidationError):
            sufficient.metric(float("nan"), 0.0)

    def test_soft_cap_is_finite_symmetric_and_rejects_bad_inputs(self):
        self.assertEqual(_soft_cap(0.0, 30.0), 0.0)
        self.assertAlmostEqual(_soft_cap(12.0, 30.0), -_soft_cap(-12.0, 30.0))
        self.assertLessEqual(abs(_soft_cap(1e9, 30.0)), 30.0)
        with self.assertRaises(DataValidationError):
            _soft_cap(float("nan"), 30.0)
        with self.assertRaises(DataValidationError):
            _soft_cap(1.0, 0.0)

    def test_feature_leakage_scan_accepts_legal_and_rejects_forbidden_columns(self):
        legal = ["hidden_z_mean", "visible_tvt_slope", "e006_path_mean", "e007_visible_gr_coverage"]
        self.assertTrue(_validate_feature_names(legal)["pass"])
        for forbidden in (
            "hidden_tvt_mean",
            "oracle_residual",
            "typewell_gr_mean",
            "spatial_x_mid",
            "ancc_hidden_slope",
            "target_rmse",
        ):
            with self.assertRaises(DataValidationError, msg=forbidden):
                _validate_feature_names([forbidden, "e007_visible_gr_coverage"])
        with self.assertRaises(DataValidationError):
            _validate_feature_names(["visible_tvt_mean"])

    @staticmethod
    def _toy_records():
        records = {}
        for index in range(12):
            well_id = f"{index + 1:08x}"
            truth = [10.0 + index, 11.0 + index, 12.0 + index]
            base = [value - (0.4 * index - 2.0) for value in truth]
            sufficient = ActionSufficient.from_paths(well_id, truth, base)
            records[well_id] = {
                "features": {
                    "geometry": float(index),
                    "visible_gr": float(index % 4),
                    "e007_visible_gr_coverage": 0.4 + 0.04 * index,
                    "e007_visible_pseudo_holdout_gain": (-1.0) ** index * 0.1,
                },
                "targets": {
                    "residual_datum_ft": sufficient.datum_target,
                    "residual_trend_ft": sufficient.trend_target,
                },
                "sufficient": sufficient,
            }
        return records

    def test_split_preparation_uses_training_statistics_only(self):
        records = self._toy_records()
        ids = sorted(records)
        first = _prepare_model(
            records=records,
            train_ids=ids[:8],
            test_ids=ids[8:],
            feature_names=sorted(records[ids[0]]["features"]),
            target_names=("residual_datum_ft", "residual_trend_ft"),
            maximum_features=3,
        )
        altered = self._toy_records()
        for well_id in ids[8:]:
            altered[well_id]["features"]["geometry"] = 1e12
        second = _prepare_model(
            records=altered,
            train_ids=ids[:8],
            test_ids=ids[8:],
            feature_names=sorted(records[ids[0]]["features"]),
            target_names=("residual_datum_ft", "residual_trend_ft"),
            maximum_features=3,
        )
        self.assertEqual(first.selected_names, second.selected_names)
        self.assertEqual(first.medians, second.medians)
        self.assertEqual(first.means, second.means)
        self.assertEqual(first.scales, second.scales)
        self.assertNotEqual(first.test_x, second.test_x)

    def test_ridge_predictions_are_deterministic_capped_and_duplicate_invariant(self):
        records = self._toy_records()
        ids = sorted(records)
        features = sorted(records[ids[0]]["features"])
        first, model = _fit_predict(
            records=records,
            train_ids=ids[:8],
            test_ids=ids[8:],
            feature_names=features,
            target_names=("residual_datum_ft", "residual_trend_ft"),
            config=self.config,
        )
        second, _ = _fit_predict(
            records=records,
            train_ids=ids[:8],
            test_ids=ids[8:],
            feature_names=features,
            target_names=("residual_datum_ft", "residual_trend_ft"),
            config=self.config,
        )
        duplicate = _duplicate_predictions(model, ids[8:], self.config)
        self.assertEqual(first, second)
        for well_id in ids[8:]:
            self.assertLessEqual(abs(first[well_id][0]), 30.0)
            self.assertLessEqual(abs(first[well_id][1]), 60.0)
            self.assertAlmostEqual(first[well_id][0], duplicate[well_id][0], places=10)
            self.assertAlmostEqual(first[well_id][1], duplicate[well_id][1], places=10)

    def test_target_derangement_is_deterministic_and_has_no_fixed_points(self):
        records = self._toy_records()
        ids = sorted(records)[:8]
        first = _deranged_targets(records, ids, "context")
        second = _deranged_targets(records, ids, "context")
        self.assertEqual(first, second)
        originals = {well_id: records[well_id]["targets"] for well_id in ids}
        self.assertTrue(all(first[well_id] is not originals[well_id] for well_id in ids))
        with self.assertRaises(DataValidationError):
            _deranged_targets(records, ids[:1], "context")

    def test_conservative_scale_selection_can_choose_smallest_near_best_or_zero(self):
        records = self._toy_records()
        ids = sorted(records)
        good = {
            well_id: (
                float(records[well_id]["targets"]["residual_datum_ft"]),
                float(records[well_id]["targets"]["residual_trend_ft"]),
            )
            for well_id in ids
        }
        selected, rows = _select_scale(records, good, ids, self.config)
        self.assertIn(selected, self.config["ridge"]["action_scales"])
        self.assertEqual(sum(bool(row["selected"]) for row in rows), 1)
        bad = {well_id: (-100.0, 100.0) for well_id in ids}
        selected_bad, _ = _select_scale(records, bad, ids, self.config)
        self.assertEqual(selected_bad, 0.0)

    def test_diagnostic_reader_rejects_duplicate_and_nonfinite_rows(self):
        fields = [
            "well_id",
            "visible_gr_coverage",
            "hidden_gr_coverage",
            "template_states",
            "query_anchors",
            "median_best_fingerprint_distance",
            "median_fingerprint_margin",
            "visible_pseudo_holdout_gain",
            "pseudo_base_rmse",
            "pseudo_fixed_rmse",
            "fallback_reason",
            "maximum_absolute_correction",
        ]
        row = {
            "well_id": "aaaaaaaa",
            "visible_gr_coverage": 1,
            "hidden_gr_coverage": 1,
            "template_states": 50,
            "query_anchors": 10,
            "median_best_fingerprint_distance": 0.1,
            "median_fingerprint_margin": 0.2,
            "visible_pseudo_holdout_gain": 0.1,
            "pseudo_base_rmse": 1,
            "pseudo_fixed_rmse": 0.9,
            "fallback_reason": "",
            "maximum_absolute_correction": 3,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            self.assertIn("aaaaaaaa", _read_e007_diagnostics(path))
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)
            with self.assertRaises(DataValidationError):
                _read_e007_diagnostics(path)
            row["well_id"] = "bbbbbbbb"
            row["pseudo_base_rmse"] = "nan"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaises(DataValidationError):
                _read_e007_diagnostics(path)

    @staticmethod
    def _write_gzip(path, fields, rows):
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)

    def test_parent_alignment_detects_value_and_id_mismatch(self):
        e006_fields = ["id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e004_geometry_prefix", "nested_conservative_grid"]
        e007_fields = ["id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e004_geometry_prefix", "e006_nested_fusion", "selfcorr_raw_bounded"]
        left = [{"id": "aaaaaaaa_2", "well_id": "aaaaaaaa", "row_index": 2, "hidden_index": 0, "target": 10, "last_known_tvt": 9, "e004_geometry_prefix": 9.5, "nested_conservative_grid": 9.8}]
        right = [{"id": "aaaaaaaa_2", "well_id": "aaaaaaaa", "row_index": 2, "hidden_index": 0, "target": 10, "last_known_tvt": 9, "e004_geometry_prefix": 9.5, "e006_nested_fusion": 9.8, "selfcorr_raw_bounded": 10.1}]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.gz", Path(tmp) / "b.gz"
            self._write_gzip(a, e006_fields, left)
            self._write_gzip(b, e007_fields, right)
            self.assertTrue(_audit_parent_alignment(a, b)["pass"])
            right[0]["e006_nested_fusion"] = 99
            self._write_gzip(b, e007_fields, right)
            with self.assertRaises(DataValidationError):
                _audit_parent_alignment(a, b)

    def test_context_builder_has_no_overlap(self):
        ids = [f"{index + 1:08x}" for index in range(10)]
        folds = []
        for version in range(5):
            folds.append({
                "version": f"v{version + 1}",
                "n_folds": 5,
                "assignments": {well_id: (index + version) % 5 for index, well_id in enumerate(ids)},
            })
        spatial = {well_id: index % 2 for index, well_id in enumerate(ids)}
        typewell = {well_id: (index // 2) % 2 for index, well_id in enumerate(ids)}
        config = json.loads(json.dumps(self.config))
        config["stress"]["spatial_bins"] = 2
        config["stress"]["typewell_clusters"] = 2
        contexts, audits = _build_contexts(ids, folds, spatial, typewell, config)
        self.assertEqual(len(contexts), 29)
        self.assertTrue(all(row["pass"] for row in audits))
        self.assertTrue(all(not (set(context.train_ids) & set(context.test_ids)) for context in contexts))

    def _build_fixture(self, root: Path):
        (root / "data/train").mkdir(parents=True)
        (root / "folds").mkdir()
        (root / "experiments/E006/results").mkdir(parents=True)
        (root / "experiments/E007/results").mkdir(parents=True)
        (root / "artifacts/E006").mkdir(parents=True)
        (root / "artifacts/E007").mkdir(parents=True)
        (root / "experiments/E008").mkdir(parents=True)
        ids = [f"{index + 1:08x}" for index in range(10)]
        parent_rows = []
        for well_index, well_id in enumerate(ids):
            known, total = 15, 30
            last_visible = 1000 + 3 * well_index + 0.2 * (known - 1)
            datum = (well_index - 4.5) * 0.7
            trend = ((well_index % 5) - 2) * 1.2
            horizontal_rows = []
            for row_index in range(total):
                md = 100 + row_index
                x = 100 * well_index + row_index * (1 + 0.02 * well_index)
                y = 50 * well_index + row_index * 0.4 * ((well_index % 3) - 1)
                z = 2000 + 0.08 * row_index + 0.01 * well_index * row_index
                gr = 50 + 5 * math.sin((row_index + well_index) / 3) + 0.5 * well_index
                visible = 1000 + 3 * well_index + 0.2 * row_index
                if row_index < known:
                    tvt = visible
                    tvt_input = f"{tvt:.8f}"
                else:
                    hidden_index = row_index - known
                    basis = hidden_index / (total - known - 1) - 0.5
                    e006 = last_visible + 0.15 * (hidden_index + 1) + 0.03 * well_index * hidden_index
                    tvt = e006 + datum + trend * basis + 0.15 * math.sin(hidden_index / 2 + well_index)
                    tvt_input = ""
                    parent_rows.append({
                        "id": f"{well_id}_{row_index}",
                        "well_id": well_id,
                        "row_index": row_index,
                        "hidden_index": hidden_index,
                        "target": tvt,
                        "last": last_visible,
                        "e004": last_visible + 0.08 * (hidden_index + 1),
                        "e006": e006,
                        "raw": e006 + 3 * math.sin((hidden_index + well_index) / 4),
                    })
                horizontal_rows.append([md, x, y, z, tvt, gr, tvt_input])
            with (root / "data/train" / f"{well_id}__horizontal_well.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["MD", "X", "Y", "Z", "TVT", "GR", "TVT_input"])
                writer.writerows(horizontal_rows)
            with (root / "data/train" / f"{well_id}__typewell.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["TVT", "GR"])
                for index in range(12):
                    writer.writerow([980 + 5 * index, 45 + well_index + math.sin(index)])
        _, data = scan_profiles(root / "data/train")
        for version in range(5):
            fold = {
                "schema_version": 1,
                "version": f"v{version + 1}",
                "seed": 100 + version,
                "strategy": "fixture",
                "data_signature": data["data_signature"],
                "fingerprint": hashlib.sha256(f"fixture-{version}".encode()).hexdigest(),
                "n_folds": 5,
                "well_count": len(ids),
                "assignments": {well_id: (index + version) % 5 for index, well_id in enumerate(ids)},
                "fold_summary": [],
            }
            (root / "folds" / f"v{version + 1}.json").write_text(json.dumps(fold), encoding="utf-8")
        e006_fields = ["id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e004_geometry_prefix", "nested_conservative_grid"]
        e006_rows = [{
            "id": row["id"], "well_id": row["well_id"], "row_index": row["row_index"], "hidden_index": row["hidden_index"],
            "target": f"{row['target']:.8f}", "last_known_tvt": f"{row['last']:.8f}",
            "e004_geometry_prefix": f"{row['e004']:.8f}", "nested_conservative_grid": f"{row['e006']:.8f}",
        } for row in parent_rows]
        e007_fields = ["id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt", "e004_geometry_prefix", "e006_nested_fusion", "selfcorr_raw_bounded"]
        e007_rows = [{
            "id": row["id"], "well_id": row["well_id"], "row_index": row["row_index"], "hidden_index": row["hidden_index"],
            "target": f"{row['target']:.8f}", "last_known_tvt": f"{row['last']:.8f}",
            "e004_geometry_prefix": f"{row['e004']:.8f}", "e006_nested_fusion": f"{row['e006']:.8f}",
            "selfcorr_raw_bounded": f"{row['raw']:.8f}",
        } for row in parent_rows]
        self._write_gzip(root / "artifacts/E006/oof_predictions.csv.gz", e006_fields, e006_rows)
        self._write_gzip(root / "artifacts/E007/oof_predictions.csv.gz", e007_fields, e007_rows)
        diagnostic_fields = [
            "well_id", "visible_gr_coverage", "hidden_gr_coverage", "template_states", "query_anchors",
            "median_best_fingerprint_distance", "median_fingerprint_margin", "visible_pseudo_holdout_gain",
            "pseudo_base_rmse", "pseudo_fixed_rmse", "fallback_reason", "maximum_absolute_correction",
        ]
        diagnostic_path = root / "experiments/E007/results/selfcorr_diagnostics.csv"
        with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostic_fields, lineterminator="\n")
            writer.writeheader()
            for index, well_id in enumerate(ids):
                writer.writerow({
                    "well_id": well_id, "visible_gr_coverage": 1, "hidden_gr_coverage": 1,
                    "template_states": 60 + index, "query_anchors": 8,
                    "median_best_fingerprint_distance": 0.1 + 0.01 * index,
                    "median_fingerprint_margin": 0.3, "visible_pseudo_holdout_gain": 0.2 - 0.01 * index,
                    "pseudo_base_rmse": 1, "pseudo_fixed_rmse": 0.8, "fallback_reason": "",
                    "maximum_absolute_correction": 3,
                })
        model = root / "experiments/E006/results/model.json"
        model.write_text('{"fixture":true}\n', encoding="utf-8")
        (root / "experiments/E006/results/summary.json").write_text("{}", encoding="utf-8")
        (root / "experiments/E007/results/summary.json").write_text("{}", encoding="utf-8")
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        def rmse(column):
            return math.sqrt(sum((float(row[column]) - float(row["target"])) ** 2 for row in e007_rows) / len(e007_rows))
        config = json.loads(json.dumps(self.config))
        config["expected_wells"] = len(ids)
        config["data_signature"] = data["data_signature"]
        config["fold_files"] = [f"folds/v{index}.json" for index in range(1, 6)]
        config["parents"].update({
            "e006_oof": "artifacts/E006/oof_predictions.csv.gz",
            "e006_oof_sha256": sha(root / "artifacts/E006/oof_predictions.csv.gz"),
            "e006_expected_rmse": rmse("e006_nested_fusion"),
            "e004_expected_rmse": rmse("e004_geometry_prefix"),
            "last_known_expected_rmse": rmse("last_known_tvt"),
            "e006_summary": "experiments/E006/results/summary.json",
            "e006_model": "experiments/E006/results/model.json",
            "e006_model_sha256": sha(model),
            "e007_oof": "artifacts/E007/oof_predictions.csv.gz",
            "e007_oof_sha256": sha(root / "artifacts/E007/oof_predictions.csv.gz"),
            "e007_diagnostics": "experiments/E007/results/selfcorr_diagnostics.csv",
            "e007_diagnostics_sha256": sha(diagnostic_path),
            "e007_summary": "experiments/E007/results/summary.json",
        })
        config["stress"].update({"spatial_bins": 2, "typewell_clusters": 2})
        config["controls"].update({
            "maximum_shuffled_gain_vs_e006": 999, "maximum_shuffled_abs_datum_correlation": 1,
            "maximum_shuffled_abs_trend_correlation": 1, "oracle_minimum_gain_vs_best_legal": -999,
            "minimum_e007_feature_selection_maps": 0, "maximum_e007_ablation_advantage_ft": 999,
            "maximum_absolute_emitted_correction_ft": 100, "minimum_nonzero_outer_cells": 0,
            "minimum_maps_with_nonzero_median_scale": 0, "maximum_outer_scale_range": 1,
        })
        config["promotion"].update({
            "minimum_gain_vs_e006": -999, "minimum_gain_vs_last_known": -999,
            "minimum_map_wins": 0, "minimum_map_gain": -999, "minimum_outer_cell_wins": 0,
            "maximum_p90_deterioration_vs_e006": 999, "maximum_p90_deterioration_vs_last_known": 999,
            "maximum_worst5_sse_share_increase_vs_e006": 1,
            "maximum_worst5_sse_share_increase_vs_last_known": 1,
            "require_positive_spatial_group_gain_vs_e006": False,
            "require_positive_typewell_group_gain_vs_e006": False,
            "maximum_special_slice_deterioration_vs_e006": 999,
        })
        (root / "experiments/E008/config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return config

    def test_full_fixture_run_is_deterministic_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._build_fixture(root)
            summaries = []
            for run in ("first", "second"):
                summaries.append(run_e008(
                    root=root,
                    train_dir=root / "data/train",
                    output_dir=root / run / "results",
                    artifact_dir=root / run / "artifacts",
                    config=config,
                    code_sha="f" * 40,
                ))
            self.assertEqual(summaries[0]["reported_candidate"], summaries[1]["reported_candidate"])
            self.assertAlmostEqual(
                summaries[0]["candidate_metrics"][summaries[0]["reported_candidate"]]["rmse"],
                summaries[1]["candidate_metrics"][summaries[1]["reported_candidate"]]["rmse"],
            )
            first, second = root / "first/results", root / "second/results"
            for path in first.iterdir():
                if path.name in {"summary.json", "control_metrics.csv", "artifact_manifest.json"}:
                    continue
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes(), path.name)
            self.assertEqual(
                (root / "first/artifacts/oof_predictions.csv.gz").read_bytes(),
                (root / "second/artifacts/oof_predictions.csv.gz").read_bytes(),
            )
            manifest = json.loads((first / "artifact_manifest.json").read_text())
            self.assertEqual(len(manifest["parent_artifacts"]), 4)
            self.assertEqual(len(manifest["fold_files"]), 5)
            self.assertTrue(summaries[0]["controls"]["feature_leakage"]["pass"])
            self.assertTrue(summaries[0]["controls"]["oof_identity"]["pass"])

    def test_official_result_artifacts_match_manifest_when_available(self):
        manifest_path = ROOT / "experiments/E008/results/artifact_manifest.json"
        if not manifest_path.exists():
            self.skipTest("official E008 artifacts not generated yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads((ROOT / "experiments/E008/results/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["experiment_id"], "E008")
        self.assertIn(summary["status"], {"promoted", "rejected"})
        for item in manifest["files"]:
            path = ROOT / "experiments/E008/results" / item["path"]
            self.assertEqual(path.stat().st_size, item["bytes"], item["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"], item["path"])
        for item in manifest["external_artifacts"]:
            path = ROOT / item["path"]
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
