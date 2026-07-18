from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.rogii_validation.harness import generate_fold_maps, scan_profiles
from src.rogii_validation.structural import huber_line, run_e002


class E002StructuralTests(unittest.TestCase):
    def test_huber_line_resists_one_large_outlier(self) -> None:
        md = [float(index) for index in range(100)]
        values = [20.0 + 0.03 * value for value in md]
        values[50] += 80.0
        _, slope, scale = huber_line(
            md,
            values,
            window_rows=100,
            huber_k=1.5,
            iterations=8,
        )
        self.assertAlmostEqual(slope, 0.03, places=4)
        self.assertGreaterEqual(scale, 0.0)

    def test_fixture_run_is_promoted_and_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / "data" / "train"
            fold_dir = root / "folds"
            train_dir.mkdir(parents=True)
            fold_dir.mkdir(parents=True)
            for well_number in range(10):
                well_id = f"{well_number:08x}"
                path = train_dir / f"{well_id}__horizontal_well.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["MD", "Z", "TVT", "TVT_input"])
                    writer.writeheader()
                    slope = 0.01 + 0.001 * well_number
                    for row_index in range(110 + 3 * well_number):
                        md = float(row_index)
                        z = -1000.0 - 2.0 * well_number - 0.45 * md
                        u = 150.0 + slope * md
                        tvt = u - z
                        writer.writerow(
                            {
                                "MD": md,
                                "Z": z,
                                "TVT": tvt,
                                "TVT_input": tvt if row_index < 80 else "",
                            }
                        )
            profiles, data = scan_profiles(train_dir)
            fold_maps = generate_fold_maps(profiles, data["data_signature"], n_folds=5)
            fold_files: list[str] = []
            for fold_map in fold_maps:
                relative = f"folds/{fold_map['version']}.json"
                fold_files.append(relative)
                (root / relative).write_text(
                    json.dumps(fold_map, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            config = {
                "schema_version": 1,
                "experiment_id": "E002",
                "data_signature": data["data_signature"],
                "expected_wells": 10,
                "fold_files": fold_files,
                "candidates": {
                    "last_known_tvt": {},
                    "constant_u": {},
                    "robust_linear_u": {
                        "window_rows": 60,
                        "huber_k": 1.5,
                        "iterations": 6,
                        "slope_min": -0.08,
                        "slope_max": 0.08,
                    },
                    "quadratic_u": {
                        "window_rows": 70,
                        "linear_window_rows": 60,
                        "huber_k": 1.5,
                        "iterations": 6,
                        "slope_min": -0.08,
                        "slope_max": 0.08,
                        "curvature_min": -0.00005,
                        "curvature_max": 0.00005,
                        "max_abs_u_drift": 180.0,
                    },
                    "constrained_spline_u": {
                        "linear_window_rows": 60,
                        "huber_k": 1.5,
                        "iterations": 6,
                        "slope_min": -0.08,
                        "slope_max": 0.08,
                        "transition_fraction": 1.0,
                        "transition_min_rows": 10,
                        "transition_max_rows": 100,
                        "endpoint_slope_fraction": 0.0,
                        "endpoint_displacement_fraction": 0.8,
                        "max_abs_u_drift": 120.0,
                    },
                },
                "sign_verification": {"required_plus_to_minus_rms_ratio_max": 0.30},
                "promotion": {
                    "minimum_pooled_rmse_gain": 0.15,
                    "minimum_improved_fold_cells": 20,
                    "total_fold_cells": 25,
                    "maximum_p90_well_rmse_deterioration": 0.25,
                    "maximum_worst_5pct_sse_share_increase": 0.02,
                    "require_long_hidden_improvement": True,
                    "maximum_runtime_minutes": 20,
                },
            }
            first_output = root / "first" / "results"
            second_output = root / "second" / "results"
            first_artifacts = root / "first" / "artifacts"
            second_artifacts = root / "second" / "artifacts"
            first = run_e002(
                root=root,
                train_dir=train_dir,
                fold_dir=fold_dir,
                output_dir=first_output,
                artifact_dir=first_artifacts,
                config=config,
            )
            second = run_e002(
                root=root,
                train_dir=train_dir,
                fold_dir=fold_dir,
                output_dir=second_output,
                artifact_dir=second_artifacts,
                config=config,
            )
            self.assertEqual(first["status"], "promoted")
            self.assertEqual(first["selected_candidate"], "robust_linear_u")
            self.assertEqual(first["best_challenger"], "robust_linear_u")
            self.assertGreater(first["trend_transfer_diagnostic"]["visible_to_hidden_u_slope_correlation_oracle"], 0.99)
            self.assertLess(first["candidate_metrics"]["robust_linear_u"]["rmse"], 1e-8)
            self.assertTrue(all(detail["pass"] for detail in first["controls"].values()))
            self.assertLess(first["sign_verification"]["plus_to_minus_ratio"], 0.05)
            first_files = sorted(path.name for path in first_output.iterdir())
            second_files = sorted(path.name for path in second_output.iterdir())
            self.assertEqual(first_files, second_files)
            for name in first_files:
                self.assertEqual(
                    hashlib.sha256((first_output / name).read_bytes()).hexdigest(),
                    hashlib.sha256((second_output / name).read_bytes()).hexdigest(),
                    name,
                )
            self.assertEqual(
                hashlib.sha256((first_artifacts / "oof_predictions.csv.gz").read_bytes()).hexdigest(),
                hashlib.sha256((second_artifacts / "oof_predictions.csv.gz").read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
