from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if np is not None:
    from rogii_validation.e010_paths import (
        basis_matrix,
        candidate_count,
        coefficient_grid,
        dtw_path,
        grid_oracle,
        grid_values,
        hmm_path,
        reconstruct_path,
        valid_coefficients,
    )
    from rogii_validation.gr_path import read_well
    from rogii_validation.harness import DataValidationError, scan_profiles
    from rogii_validation.nonlinear_selector import (
        RAW_FAMILIES,
        SELECTOR_BRANCHES,
        _decode_predictions,
        _deranged_targets,
        _family_definitions,
        _fit_model,
        _grid_clip,
        E010_OOF_FILENAME,
        E010_RESULT_FILENAMES,
        finalize_e010_outputs,
        _prepare_features,
        _screen_ids,
        _selector_feature_names,
        _target_matrix,
        run_e010,
        validate_e010_config,
        FamilyTarget,
        WellRecord,
    )


@unittest.skipIf(np is None, "E010 scientific-runtime tests require NumPy")
class E010Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "experiments/E010/config.json").read_text())

    def test_frozen_config_and_candidate_count(self):
        validate_e010_config(self.config)
        total, counts = candidate_count(self.config)
        self.assertGreaterEqual(total, 100000)
        self.assertEqual(set(counts), {row["name"] for row in self.config["candidate_bank"]["families"]})
        self.assertEqual(tuple(self.config["candidate_bank"]["raw_anchors"]), RAW_FAMILIES)
        self.assertEqual(tuple(row["name"] for row in self.config["selector"]["branches"]), SELECTOR_BRANCHES)
        self.assertEqual(_grid_clip(1.25, {"minimum": -5.0, "maximum": 5.0, "step": 2.5}), 1.25)
        self.assertEqual(_grid_clip(9.0, {"minimum": -5.0, "maximum": 5.0, "step": 2.5}), 5.0)
        broken = copy.deepcopy(self.config)
        broken["candidate_bank"]["coefficient_grids"]["datum_ft"]["step"] = 0
        with self.assertRaises(DataValidationError):
            validate_e010_config(broken)
        broken = copy.deepcopy(self.config)
        broken["selector"]["branches"] = list(reversed(broken["selector"]["branches"]))
        with self.assertRaises(DataValidationError):
            validate_e010_config(broken)
        broken = copy.deepcopy(self.config)
        broken["deployment"]["internet"] = True
        with self.assertRaises(DataValidationError):
            validate_e010_config(broken)

    def test_basis_grid_oracle_one_row_ties_clipping_and_nonfinite(self):
        affine = {"name": "a", "anchor": "e006_nested_fusion", "basis": "affine"}
        quadratic = {"name": "q", "anchor": "e006_nested_fusion", "basis": "quadratic"}
        hinge = {"name": "h", "anchor": "e006_nested_fusion", "basis": "hinge", "location": 0.5}
        jump = {"name": "j", "anchor": "e006_nested_fusion", "basis": "jump", "location": 0.5, "width": 0.05}
        self.assertEqual(basis_matrix(1, affine).shape, (1, 2))
        self.assertTrue(np.allclose(basis_matrix(1, affine), [[1.0, 0.0]]))
        self.assertAlmostEqual(float(basis_matrix(101, hinge)[:, 2].mean()), 0.0, places=12)
        self.assertAlmostEqual(float(basis_matrix(101, jump)[:, 2].mean()), 0.0, places=12)
        coefficients = np.asarray([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
        oracle = grid_oracle([10.0], [10.0], affine, coefficients)
        self.assertEqual(oracle.candidate_index, 0)
        self.assertEqual(oracle.rmse, 0.0)
        path = reconstruct_path([0.0, 0.0, 0.0], quadratic, [1000.0, 1000.0, 1000.0], 5.0)
        self.assertTrue(np.all(np.abs(path) <= 5.0 + 1e-12))
        with self.assertRaises(DataValidationError):
            basis_matrix(0, affine)
        with self.assertRaises(DataValidationError):
            reconstruct_path([1.0], affine, [math.nan, 0.0], 5.0)
        with self.assertRaises(DataValidationError):
            grid_oracle([1.0], [1.0, 2.0], affine, coefficients)
        bad_grid = {"minimum": 0.0, "maximum": 1.0, "step": -1.0}
        with self.assertRaises(DataValidationError):
            grid_values(bad_grid)

    def test_selector_feature_and_target_preparation_is_training_only_and_deterministic(self):
        records = {}
        targets = {}
        families = ("e006_nested_fusion", "e006_affine")
        for index in range(20):
            well_id = f"{index + 1:08x}"
            truth = np.asarray([10.0 + index])
            base = np.asarray([10.0 + index])
            records[well_id] = WellRecord(
                well_id, (f"{well_id}_0",), (0,), truth,
                {name: base.copy() for name in RAW_FAMILIES},
                {
                    "backtest_0p5_rmse": float(index), "backtest_0p7_rmse": float(index % 3),
                    "backtest_0p85_rmse": float(index % 5), "e006_path_mean": float(index),
                    "visible_tvt_mean": None if index == 0 else float(index * 2),
                },
                (float(index), 0.0), (0.0, 1.0, 10.0), 0.0,
            )
            targets[well_id] = {
                "e006_nested_fusion": FamilyTarget("e006_nested_fusion", (0, 0, 0), 1 + index * 0.1, 1, 0),
                "e006_affine": FamilyTarget("e006_affine", (index * 0.1, 0, 0), 0.5 + index * 0.05, 1, 0),
            }
        ids = sorted(records)
        y = _target_matrix(ids[:15], targets, families)
        names = sorted(records[ids[0]].features)
        first = _prepare_features(records=records, train_ids=ids[:15], test_ids=ids[15:], all_names=names, selector="all", maximum_features=4, train_y=y)
        second = _prepare_features(records=records, train_ids=ids[:15], test_ids=ids[15:], all_names=names, selector="all", maximum_features=4, train_y=y)
        self.assertEqual(first.names, second.names)
        self.assertTrue(np.array_equal(first.train_x, second.train_x))
        self.assertTrue(np.all(np.isfinite(first.test_x)))
        self.assertTrue(any(name.startswith("backtest_0p5_") for name in _selector_feature_names(names, "path_and_multicut")))
        with self.assertRaises(DataValidationError):
            _selector_feature_names(names, "unknown")
        shuffled = _deranged_targets(y, ids[:15], "test")
        self.assertFalse(np.array_equal(shuffled, y))
        self.assertEqual(sorted(map(tuple, shuffled)), sorted(map(tuple, y)))
        ridge = next(row for row in self.config["selector"]["branches"] if row["name"] == "ridge_all_f64")
        p1 = _fit_model(spec=ridge, records=records, train_ids=ids[:15], test_ids=ids[15:], feature_names=names, train_y=y)
        p2 = _fit_model(spec=ridge, records=records, train_ids=ids[:15], test_ids=ids[15:], feature_names=names, train_y=y)
        self.assertTrue(np.array_equal(p1.values, p2.values))

    @staticmethod
    def _gzip(path: Path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                    writer.writeheader(); writer.writerows(rows)

    def _fixture(self, root: Path):
        train_dir = root / "data/train"; train_dir.mkdir(parents=True)
        (root / "folds").mkdir()
        (root / "artifacts/E005").mkdir(parents=True)
        (root / "artifacts/E006").mkdir(parents=True)
        (root / "artifacts/E009").mkdir(parents=True)
        (root / "experiments/E008/results").mkdir(parents=True)
        (root / "experiments/E009/results").mkdir(parents=True)
        well_ids = []
        e005_rows = []; e006_rows = []; e009_rows = []; feature_rows = []; wide_rows = []; residual_rows = []
        for row_bin, hidden in enumerate((4, 8, 12)):
            for missing_bin in range(2):
                for member in range(5):
                    index = len(well_ids)
                    well_id = f"{index + 1:08x}"; well_ids.append(well_id)
                    known = 30; total = known + hidden
                    datum = (index - 14.5) * 0.12
                    horizontal_rows = []
                    typewell_rows = []
                    for tvt_i in range(800):
                        tvt = 900.0 + tvt_i * 0.25
                        gr = 55.0 + 8.0 * math.sin(tvt / 7.0) + 3.0 * math.cos(tvt / 13.0)
                        typewell_rows.append([tvt, gr])
                    for row_index in range(total):
                        md = 1000.0 + row_index * (1.0 + 0.01 * member)
                        x = index * 100.0 + row_index
                        y = row_bin * 1000.0 + missing_bin * 500.0 + row_index * 0.2
                        z = 2000.0 + row_index * 0.05
                        visible = 1000.0 + index * 0.8 + 0.12 * row_index
                        if row_index < known:
                            tvt = visible; tvt_input = f"{tvt:.8f}"
                        else:
                            h = row_index - known
                            base = visible + 0.08 * (h + 1)
                            tvt = base + datum + 0.4 * math.sin((h + index) / 3.0)
                            tvt_input = ""
                        gr = 55.0 + 8.0 * math.sin(tvt / 7.0) + 3.0 * math.cos(tvt / 13.0)
                        if row_index >= known and missing_bin and (row_index - known) % 2:
                            gr_value = ""
                        else:
                            gr_value = f"{gr:.8f}"
                        horizontal_rows.append([md, x, y, z, tvt, gr_value, tvt_input])
                    with (train_dir / f"{well_id}__horizontal_well.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle); writer.writerow(["MD", "X", "Y", "Z", "TVT", "GR", "TVT_input"]); writer.writerows(horizontal_rows)
                    with (train_dir / f"{well_id}__typewell.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle); writer.writerow(["TVT", "GR"]); writer.writerows(typewell_rows)
                    residual_values = []
                    for h in range(hidden):
                        row_index = known + h
                        target = float(horizontal_rows[row_index][4])
                        last = float(horizontal_rows[known - 1][4])
                        e004 = last + 0.06 * (h + 1)
                        align = e004 + 0.2 * math.sin(h / 2)
                        pf = e004 + 0.5 * math.sin((h + index) / 2)
                        trellis = e004 + 0.4 * math.cos((h + index) / 3)
                        e006 = 0.5 * e004 + 0.5 * pf
                        residual_values.append(target - e006)
                        common = {"id": f"{well_id}_{row_index}", "well_id": well_id, "row_index": row_index, "hidden_index": h, "target": target}
                        e005_rows.append({**common, "e004_geometry_prefix": e004, "align_visible_path": align, "pf_gr_path": pf, "trellis_gr_path": trellis})
                        e006_rows.append({**common, "last_known_tvt": last, "e004_geometry_prefix": e004, "nested_conservative_grid": e006})
                        e009_rows.append({**common, "last_known_tvt": last, "e006_nested_fusion": e006})
                    feature_rows.append({
                        "well_id": well_id,
                        "backtest_0p5_mean": datum * 0.7, "backtest_0p5_rmse": abs(datum) + 0.2,
                        "backtest_0p7_mean": datum * 0.85, "backtest_0p7_rmse": abs(datum) + 0.1,
                        "backtest_0p85_mean": datum * 0.95, "backtest_0p85_rmse": abs(datum) + 0.05,
                        "e006_path_mean": float(np.mean([row["nested_conservative_grid"] for row in e006_rows[-hidden:]])),
                        "e006_minus_last_mean": float(np.mean([row["nested_conservative_grid"] - row["last_known_tvt"] for row in e006_rows[-hidden:]])),
                        "visible_tvt_mean": 1000.0 + index * 0.8,
                        "hidden_rows": hidden, "known_rows": known, "hidden_fraction": hidden / total,
                        "hidden_gr_missing_fraction": 0.5 if missing_bin else 0.0,
                    })
                    wide_rows.append({"well_id": well_id, "ridge_all_a25_f64": datum * 0.9})
                    residual_rows.append({"well_id": well_id, "residual_datum_ft": sum(residual_values) / len(residual_values)})
        _, profile = scan_profiles(train_dir)
        for version in range(5):
            fold = {
                "schema_version": 1, "version": f"v{version + 1}", "seed": 1000 + version,
                "strategy": "fixture", "data_signature": profile["data_signature"],
                "fingerprint": hashlib.sha256(f"e010-fixture-{version}".encode()).hexdigest(),
                "n_folds": 5, "well_count": len(well_ids),
                "assignments": {well_id: (index + version) % 5 for index, well_id in enumerate(well_ids)},
                "fold_summary": [],
            }
            (root / "folds" / f"v{version + 1}.json").write_text(json.dumps(fold), encoding="utf-8")
        self._gzip(root / "artifacts/E005/oof_predictions.csv.gz", list(e005_rows[0]), e005_rows)
        self._gzip(root / "artifacts/E006/oof_predictions.csv.gz", list(e006_rows[0]), e006_rows)
        self._gzip(root / "artifacts/E009/oof_predictions.csv.gz", list(e009_rows[0]), e009_rows)
        for path, rows in ((root / "experiments/E008/results/legal_features.csv", feature_rows), (root / "experiments/E009/results/model_predictions.csv", wide_rows), (root / "experiments/E008/results/residual_targets.csv", residual_rows)):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
        config = copy.deepcopy(self.config)
        config["expected_wells"] = len(well_ids); config["data_signature"] = profile["data_signature"]
        for key, item in config["parents"].items():
            item["sha256"] = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
        config["parents"]["e006_oof"]["expected_rmse"] = 999.0
        config["development_screen"]["wells"] = 6
        for variant in config["development_screen"]["hmm_variants"]:
            variant["maximum_observations"] = 16; variant["grid_step_ft"] = 5.0; variant["rate_states"] = 5
        for variant in config["development_screen"]["dtw_variants"]:
            variant["maximum_observations"] = 16; variant["offset_step_ft"] = 5.0
        for key in ("datum_ft", "linear_ft", "shape_ft"):
            config["candidate_bank"]["coefficient_grids"][key] = {"minimum": -5.0, "maximum": 5.0, "step": 5.0}
        config["controls"].update({
            "minimum_candidate_count": 0, "minimum_families_with_unique_oracle_wins": 0,
            "oracle_minimum_gain_vs_best_legal": -999.0, "shuffled_target_maximum_gain_vs_e006": 999.0,
            "sign_flipped_target_maximum_gain_vs_e006": 999.0, "maximum_absolute_emitted_correction_ft": 999.0,
        })
        config["stress"].update({"spatial_groups": 2, "typewell_groups": 2, "maximum_special_slice_deterioration_vs_e006": 999.0})
        config["promotion"].update({
            "minimum_gain_vs_e006": -999.0, "minimum_gain_vs_last_known": -999.0,
            "minimum_map_wins": 0, "minimum_outer_cell_wins": 0, "minimum_map_gain": -999.0,
            "maximum_p90_deterioration_vs_e006": 999.0, "maximum_worst5_sse_share_increase_vs_e006": 1.0,
            "require_positive_every_spatial_group": False, "require_positive_every_typewell_group": False,
            "require_bank_oracle_below_5": False, "maximum_runtime_minutes": 10, "maximum_rss_mb": 4096,
        })
        for branch in config["selector"]["branches"]:
            if branch["model"] in {"extra_trees", "random_forest"}:
                branch["n_estimators"] = 20; branch["min_samples_leaf"] = 2
        (root / "experiments/E010").mkdir(parents=True)
        (root / "experiments/E010/config.json").write_text(json.dumps(config, indent=2) + "\n")
        return config, well_ids

    def test_hmm_dtw_missing_gr_active_determinism_and_screen_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, well_ids = self._fixture(root)
            records = {}
            for well_id in well_ids:
                well = read_well(root / f"data/train/{well_id}__horizontal_well.csv", root / f"data/train/{well_id}__typewell.csv", require_truth=True)
                truth = np.asarray(well.truth[well.known_rows:])
                base = truth + 1.0
                records[well_id] = WellRecord(well_id, tuple(f"{well_id}_{well.known_rows+i}" for i in range(well.hidden_rows)), tuple(range(well.known_rows, len(well.md))), truth, {name: base.copy() for name in RAW_FAMILIES}, {"hidden_gr_missing_fraction": well_ids.index(well_id) % 2 * 0.5}, (0,0), (0,1,1), well_ids.index(well_id) % 2 * 0.5)
            selected = _screen_ids(records, 6)
            self.assertEqual(len(selected), 6); self.assertEqual(len(set(selected)), 6)
            active_id = well_ids[0]; active = read_well(root / f"data/train/{active_id}__horizontal_well.csv", root / f"data/train/{active_id}__typewell.csv", require_truth=True)
            base = np.asarray(active.truth[active.known_rows:]) + 1.0
            hmm_variant = config["development_screen"]["hmm_variants"][0]
            dtw_variant = config["development_screen"]["dtw_variants"][0]
            h1 = hmm_path(active, base, hmm_variant, 160); h2 = hmm_path(active, base, hmm_variant, 160)
            d1 = dtw_path(active, base, dtw_variant, 160); d2 = dtw_path(active, base, dtw_variant, 160)
            self.assertTrue(np.allclose(h1.path, h2.path)); self.assertTrue(np.allclose(d1.path, d2.path))
            self.assertTrue(np.all(np.isfinite(h1.path))); self.assertTrue(np.all(np.isfinite(d1.path)))
            missing_id = well_ids[5]
            missing = read_well(root / f"data/train/{missing_id}__horizontal_well.csv", root / f"data/train/{missing_id}__typewell.csv", require_truth=True)
            missing_base = np.asarray(missing.truth[missing.known_rows:]) + 1.0
            # Half-missing remains active or safely falls back, but always finite.
            self.assertTrue(np.all(np.isfinite(hmm_path(missing, missing_base, hmm_variant, 160).path)))
            self.assertTrue(np.all(np.isfinite(dtw_path(missing, missing_base, dtw_variant, 160).path)))

    def test_full_fixture_run_is_complete_deterministic_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external_tmp:
            root = Path(tmp); config, _ = self._fixture(root)
            first = run_e010(
                root=root,
                train_dir=root / "data/train",
                output_dir=root / "first/results",
                artifact_dir=root / "first/artifacts",
                config=config,
                code_sha="a" * 40,
            )
            external_artifact_dir = Path(external_tmp) / "artifacts"
            second = run_e010(
                root=root,
                train_dir=root / "data/train",
                output_dir=root / "second/results",
                artifact_dir=external_artifact_dir,
                config=config,
                code_sha="a" * 40,
            )
            self.assertEqual(first["candidate_metrics"], second["candidate_metrics"])
            self.assertEqual(first["family_metrics"], second["family_metrics"])
            self.assertTrue(first["controls"]["parent_hashes"]["pass"])
            self.assertTrue(first["controls"]["membership"]["pass"])
            self.assertTrue(first["controls"]["oof_identity"]["pass"])
            self.assertTrue(first["controls"]["basis_reconstruction"]["pass"])
            self.assertEqual(
                (root / "first/artifacts" / E010_OOF_FILENAME).read_bytes(),
                (external_artifact_dir / E010_OOF_FILENAME).read_bytes(),
            )
            second_manifest = json.loads((root / "second/results/artifact_manifest.json").read_text())
            self.assertEqual(second_manifest["external_artifacts"][0]["path"], E010_OOF_FILENAME)
            self.assertEqual(second_manifest["external_artifacts"][0]["base"], "artifact_dir")
            dynamic = {"summary.json", "control_metrics.csv", "artifact_manifest.json", "screen_metrics.csv", "screen_well_metrics.csv"}
            for path in (root / "first/results").iterdir():
                if path.name not in dynamic:
                    self.assertEqual(path.read_bytes(), (root / "second/results" / path.name).read_bytes(), path.name)

            original_manifest = (root / "first/results/artifact_manifest.json").read_bytes()
            (root / "first/results/artifact_manifest.json").unlink()
            recovered = finalize_e010_outputs(
                root=root,
                output_dir=root / "first/results",
                artifact_dir=root / "first/artifacts",
                config=config,
                code_sha="a" * 40,
            )
            self.assertEqual(recovered["candidate_metrics"], first["candidate_metrics"])
            self.assertEqual((root / "first/results/artifact_manifest.json").read_bytes(), original_manifest)
            self.assertEqual(
                {path.name for path in (root / "first/results").iterdir()},
                {*E010_RESULT_FILENAMES, "artifact_manifest.json"},
            )

            (root / "first/results/summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(DataValidationError):
                finalize_e010_outputs(
                    root=root,
                    output_dir=root / "first/results",
                    artifact_dir=root / "first/artifacts",
                    config=config,
                    code_sha="a" * 40,
                )
            broken = copy.deepcopy(config); broken["parents"]["e005_oof"]["sha256"] = "0" * 64
            with self.assertRaises(DataValidationError):
                run_e010(
                    root=root,
                    train_dir=root / "data/train",
                    output_dir=root / "broken/results",
                    artifact_dir=root / "broken/artifacts",
                    config=broken,
                    code_sha="a" * 40,
                )


if __name__ == "__main__":
    unittest.main()
