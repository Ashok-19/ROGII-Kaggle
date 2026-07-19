from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.rogii_validation.harness import generate_fold_maps, scan_profiles
from src.rogii_validation.learnability import _metric_from_sufficient, run_e003


HORIZONTAL_FIELDS = [
    "MD", "X", "Y", "Z", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA",
    "TVT", "GR", "TVT_input",
]


class E003LearnabilityTests(unittest.TestCase):
    def _make_fixture(self, root: Path, wells: int = 20) -> tuple[Path, dict]:
        train_dir = root / "data" / "train"
        fold_dir = root / "folds"
        train_dir.mkdir(parents=True)
        fold_dir.mkdir(parents=True)
        for well_number in range(wells):
            well_id = f"{well_number:08x}"
            known_rows = 48
            hidden_rows = 22 + well_number % 7
            total_rows = known_rows + hidden_rows
            horizontal = train_dir / f"{well_id}__horizontal_well.csv"
            with horizontal.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=HORIZONTAL_FIELDS, lineterminator="\n")
                writer.writeheader()
                last_visible = 300.0 + 0.7 * well_number + 0.03 * (known_rows - 1)
                datum = 3.0 + 0.35 * well_number + 0.08 * hidden_rows
                trend = -4.0 + 0.45 * (well_number % 9)
                for row_index in range(total_rows):
                    md = float(row_index)
                    x = 1000.0 + 80.0 * well_number + 1.5 * row_index
                    y = 500.0 + 12.0 * well_number + 0.3 * row_index
                    z = -900.0 - 3.0 * well_number - 0.28 * row_index
                    if row_index < known_rows:
                        tvt = 300.0 + 0.7 * well_number + 0.03 * row_index
                        visible = tvt
                    else:
                        hidden_index = row_index - known_rows
                        centered = hidden_index / max(1, hidden_rows - 1) - 0.5
                        tvt = last_visible + datum + trend * centered
                        visible = ""
                    surfaces = {
                        "ANCC": z + 15.0 + 0.02 * well_number,
                        "ASTNU": z + 30.0 + 0.01 * row_index,
                        "ASTNL": z + 45.0 + 0.015 * row_index,
                        "EGFDU": z + 60.0 + 0.02 * row_index,
                        "EGFDL": z + 75.0 + 0.025 * row_index,
                        "BUDA": z + 90.0 + 0.03 * row_index,
                    }
                    writer.writerow({
                        "MD": md,
                        "X": x,
                        "Y": y,
                        "Z": z,
                        **surfaces,
                        "TVT": tvt,
                        "GR": 70.0 + 10.0 * ((row_index + well_number) % 11) / 10.0,
                        "TVT_input": visible,
                    })
            typewell = train_dir / f"{well_id}__typewell.csv"
            with typewell.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["TVT", "GR", "Geology"], lineterminator="\n")
                writer.writeheader()
                for index in range(80):
                    writer.writerow({"TVT": 280.0 + index, "GR": 60.0 + (index % 13), "Geology": ""})
        profiles, data = scan_profiles(train_dir)
        fold_maps = generate_fold_maps(profiles, data["data_signature"], n_folds=5)
        fold_files = []
        for fold_map in fold_maps:
            relative = f"folds/{fold_map['version']}.json"
            fold_files.append(relative)
            (root / relative).write_text(json.dumps(fold_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config = {
            "schema_version": 1,
            "experiment_id": "E003",
            "data_signature": data["data_signature"],
            "expected_wells": wells,
            "fold_files": fold_files,
            "feature_version": "fixture",
            "visible_slope_windows": [8, 16, 32],
            "visible_backtest_fractions": [0.5, 0.7, 0.85],
            "feature_screening": {"maximum_features": 24},
            "ridge": {"alpha": 2.0},
            "forest": {
                "trees": 4,
                "max_depth": 2,
                "min_leaf": 3,
                "row_fraction": 0.85,
                "feature_subsample": 6,
                "threshold_quantiles": [0.25, 0.5, 0.75],
            },
            "targets": [
                "datum_correction_ft", "trend_correction_ft", "u_slope_delta_ft_per_md", "log1p_baseline_rmse"
            ],
            "action_candidates": [
                "ridge_datum", "ridge_datum_trend", "forest_datum", "forest_datum_trend"
            ],
            "risk_fraction": 0.2,
            "sign_materiality_ft": 1.0,
            "reference_metrics": {
                "baseline_rmse": None,
                "oracle_datum_rmse": None,
                "oracle_datum_trend_rmse": None,
            },
            "spatial_stress_bins": 5,
            "promotion": {
                "minimum_action_rmse_gain": 0.01,
                "minimum_feature_gain_over_mean_action": 0.01,
                "minimum_action_map_wins": 3,
                "maximum_p90_well_rmse_deterioration": 2.0,
                "maximum_worst_5pct_sse_share_increase": 0.2,
                "minimum_datum_correlation": 0.2,
                "minimum_material_datum_sign_accuracy": 0.5,
                "require_positive_spatial_action_gain": True,
                "minimum_risk_spearman": 0.05,
                "minimum_risk_auc": 0.5,
                "minimum_risk_map_passes": 2,
                "require_positive_spatial_risk_spearman": False,
                "maximum_runtime_minutes": 2,
            },
            "controls": {
                "minimum_positive_control_correlation": 0.9,
                "maximum_duplicate_prediction_delta": 1e-7,
                "maximum_shuffled_abs_datum_correlation": 0.8,
                "maximum_shuffled_action_gain": 5.0,
                "oracle_datum_rmse_tolerance": 0.02,
                "oracle_datum_trend_rmse_tolerance": 0.02,
            },
        }
        return train_dir, config

    def test_sufficient_metric_exact_oracle(self) -> None:
        residuals = [2.0, 3.0, 4.0, 5.0, 6.0]
        n = len(residuals)
        sufficient = {
            "rows": float(n),
            "sum_i": sum(range(n)),
            "sum_i_sq": sum(index * index for index in range(n)),
            "sum_r": sum(residuals),
            "sum_r_sq": sum(value * value for value in residuals),
            "sum_i_r": sum(index * value for index, value in enumerate(residuals)),
        }
        metric = _metric_from_sufficient("fixture", sufficient, 4.0, 4.0)
        self.assertLess(metric.rmse, 1e-12)

    def test_fixture_run_is_deterministic_and_learns_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir, config = self._make_fixture(root)
            first_output = root / "first"
            second_output = root / "second"
            first = run_e003(root=root, train_dir=train_dir, output_dir=first_output, config=config)
            second = run_e003(root=root, train_dir=train_dir, output_dir=second_output, config=config)
            self.assertTrue(all(detail["pass"] for detail in first["controls"].values()))
            self.assertTrue(first["decision"]["signed_action_authorized"])
            self.assertGreater(first["selected_action_gain"], 1.0)
            self.assertLess(first["oracle_headroom"]["datum_and_trend"]["rmse"], 1e-6)
            first_files = sorted(path.name for path in first_output.iterdir())
            second_files = sorted(path.name for path in second_output.iterdir())
            self.assertEqual(first_files, second_files)
            for name in first_files:
                self.assertEqual(
                    hashlib.sha256((first_output / name).read_bytes()).hexdigest(),
                    hashlib.sha256((second_output / name).read_bytes()).hexdigest(),
                    name,
                )

    def test_official_promoted_result_artifacts_match_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        experiment = root / "experiments" / "E003"
        manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((experiment / "results" / "summary.json").read_text(encoding="utf-8"))
        artifact_manifest = json.loads(
            (experiment / "results" / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "promoted")
        self.assertEqual(summary["status"], "promoted")
        self.assertTrue(summary["decision"]["signed_action_authorized"])
        self.assertTrue(summary["decision"]["risk_detection_authorized"])
        self.assertTrue(all(detail["pass"] for detail in summary["controls"].values()))
        self.assertAlmostEqual(summary["selected_action_metrics"]["rmse"], 10.92797409180751, places=10)
        for item in artifact_manifest["files"]:
            path = experiment / "results" / item["path"]
            self.assertTrue(path.exists(), item["path"])
            self.assertEqual(path.stat().st_size, item["bytes"], item["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"], item["path"])
        for item in artifact_manifest["fold_files"]:
            path = root / item["path"]
            self.assertEqual(path.stat().st_size, item["bytes"], item["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"], item["path"])


if __name__ == "__main__":
    unittest.main()
