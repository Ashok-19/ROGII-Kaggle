from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.rogii_validation.deployment import run_e004
from src.rogii_validation.e004_inference import (
    DeploymentDataError,
    build_prediction_map,
    extract_well_features,
    predict_coefficients,
    reconstruct_hidden,
    write_submission,
)
from src.rogii_validation.harness import generate_fold_maps, scan_profiles

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e004_notebook_builder", ROOT / "tools/build_e004_notebook.py")
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)

TRAIN_FIELDS = [
    "MD", "X", "Y", "Z", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA",
    "TVT", "GR", "TVT_input",
]
TEST_FIELDS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


class E004DeploymentTests(unittest.TestCase):
    @staticmethod
    def _write_horizontal(path: Path, well_number: int, *, train: bool, bad_mode: str = "") -> tuple[int, int]:
        known_rows = 36
        hidden_rows = 12 + well_number % 6
        total = known_rows + hidden_rows
        fields = TRAIN_FIELDS if train else TEST_FIELDS
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            last_visible = 250.0 + 0.8 * well_number + 0.04 * (known_rows - 1)
            hidden_z_delta = -0.22 * hidden_rows
            datum = 1.5 + 0.32 * well_number + 0.45 * hidden_z_delta
            trend = -3.0 + 0.6 * (well_number % 7) + 0.15 * hidden_rows
            for index in range(total):
                md = float(index)
                if bad_mode == "md" and index == 8:
                    md = 7.0
                x = 1000.0 + 45.0 * well_number + 1.7 * index
                y = 500.0 + 5.0 * well_number + 0.25 * index
                z = -800.0 - 2.0 * well_number - 0.22 * index
                if index < known_rows:
                    tvt = 250.0 + 0.8 * well_number + 0.04 * index + 0.003 * index * index / known_rows
                    visible: float | str = tvt
                else:
                    hidden_index = index - known_rows
                    centered = hidden_index / max(1, hidden_rows - 1) - 0.5
                    tvt = last_visible + datum + trend * centered
                    visible = ""
                if bad_mode == "gap" and index == known_rows + 2:
                    visible = tvt
                gr: float | str = "" if bad_mode == "all_gr_missing" else 65.0 + ((index + 2 * well_number) % 17)
                row = {"MD": md, "X": x, "Y": y, "Z": z, "GR": gr, "TVT_input": visible}
                if train:
                    row.update({
                        "ANCC": z + 15.0,
                        "ASTNU": z + 30.0,
                        "ASTNL": z + 45.0,
                        "EGFDU": z + 60.0,
                        "EGFDL": z + 75.0,
                        "BUDA": z + 90.0,
                        "TVT": tvt,
                    })
                writer.writerow(row)
        return known_rows, hidden_rows

    @staticmethod
    def _write_typewell(path: Path, well_number: int, *, train: bool) -> None:
        fields = ["TVT", "GR", "Geology"] if train else ["TVT", "GR"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index in range(70 + well_number % 5):
                row = {"TVT": 220.0 + index, "GR": 55.0 + ((index + well_number) % 19)}
                if train:
                    row["Geology"] = "sand" if index % 3 else ""
                writer.writerow(row)

    def _make_fixture(self, root: Path, wells: int = 25) -> tuple[dict, Path, Path, Path]:
        train = root / "data/train"
        test = root / "data/test"
        folds = root / "folds"
        train.mkdir(parents=True)
        test.mkdir(parents=True)
        folds.mkdir(parents=True)
        boundaries: dict[str, tuple[int, int]] = {}
        for number in range(wells):
            well_id = f"{number:08x}"
            boundaries[well_id] = self._write_horizontal(train / f"{well_id}__horizontal_well.csv", number, train=True)
            self._write_typewell(train / f"{well_id}__typewell.csv", number, train=True)
        test_ids = [f"{number:08x}" for number in (0, 1, 2)]
        for number, well_id in zip((0, 1, 2), test_ids):
            self._write_horizontal(test / f"{well_id}__horizontal_well.csv", number, train=False)
            self._write_typewell(test / f"{well_id}__typewell.csv", number, train=False)
        sample = root / "data/sample_submission.csv"
        ids: list[str] = []
        for well_id in reversed(test_ids):
            known, hidden = boundaries[well_id]
            ids.extend(f"{well_id}_{index}" for index in range(known, known + hidden))
        with sample.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "tvt"], lineterminator="\n")
            writer.writeheader()
            for key in ids:
                writer.writerow({"id": key, "tvt": 0.0})
        profiles, data = scan_profiles(train)
        fold_files = []
        for fold_map in generate_fold_maps(profiles, data["data_signature"], n_folds=5):
            relative = f"folds/{fold_map['version']}.json"
            fold_files.append(relative)
            (root / relative).write_text(json.dumps(fold_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        config = {
            "schema_version": 1,
            "experiment_id": "E004",
            "data_signature": data["data_signature"],
            "expected_wells": wells,
            "fold_files": fold_files,
            "visible_slope_windows": [8, 16, 32],
            "visible_backtest_fractions": [0.5, 0.7, 0.85],
            "feature_screening": {"maximum_features": 24},
            "ridge": {"alpha": 2.0},
            "feature_families": ["geometry", "prefix", "gr", "typewell", "spatial"],
            "ablation_candidates": {
                "full_deployable": ["geometry", "prefix", "gr", "typewell", "spatial"],
                "no_gr": ["geometry", "prefix", "typewell", "spatial"],
                "geometry_prefix": ["geometry", "prefix"],
            },
            "sign_materiality_ft": 1.0,
            "spatial_stress_bins": 5,
            "promotion": {
                "minimum_rmse_gain": 0.01,
                "minimum_map_wins": 3,
                "maximum_p90_deterioration": 2.0,
                "maximum_worst5_increase": 0.2,
                "minimum_datum_correlation": 0.1,
                "minimum_datum_sign_accuracy": 0.5,
                "maximum_runtime_minutes": 2,
            },
            "deployment": {"maximum_visible_test_abs_z": 20.0},
        }
        return config, train, test, sample

    def test_full_fixture_is_deterministic_surface_free_and_notebook_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, train, test, sample = self._make_fixture(root)
            first_out, second_out = root / "first", root / "second"
            first_art, second_art = root / "artifacts1", root / "artifacts2"
            first = run_e004(root=root, train_dir=train, test_dir=test, sample_submission=sample, output_dir=first_out, artifact_dir=first_art, config=config, code_sha="a" * 40)
            second = run_e004(root=root, train_dir=train, test_dir=test, sample_submission=sample, output_dir=second_out, artifact_dir=second_art, config=config, code_sha="a" * 40)
            self.assertTrue(first["deployment"]["local_model_ready"])
            self.assertEqual(first["status"], "blocked")
            self.assertGreater(first["selected_candidate_metrics"]["gain_vs_baseline"], 0.5)
            model = json.loads((first_out / "model.json").read_text(encoding="utf-8"))
            self.assertFalse(model["surfaces_required"])
            self.assertTrue(all(not name.startswith(("ancc", "astn", "egfd", "buda")) for name in model["selected_features"]))
            for name in sorted(path.name for path in first_out.iterdir()):
                self.assertEqual(
                    hashlib.sha256((first_out / name).read_bytes()).hexdigest(),
                    hashlib.sha256((second_out / name).read_bytes()).hexdigest(),
                    name,
                )
            self.assertEqual((first_art / "submission.csv").read_bytes(), (second_art / "submission.csv").read_bytes())
            notebook = root / "notebooks/e004.ipynb"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            parity = BUILDER.finalize_local_parity(root, first_out, first_art, notebook, config_path)
            self.assertTrue(parity["byte_identical"])
            self.assertTrue(parity["local_model_ready"])
            self.assertFalse(parity["deployment_ready"])
            finalized = json.loads((first_out / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(finalized["deployment"]["local_notebook_parity"])

    def test_train_and_test_authoring_features_are_exactly_equal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, train, test, _ = self._make_fixture(root)
            train_features = extract_well_features(train / "00000000__horizontal_well.csv", train / "00000000__typewell.csv", visible_slope_windows=[8, 16, 32], visible_backtest_fractions=[0.5, 0.7, 0.85], require_truth=True)["features"]
            test_features = extract_well_features(test / "00000000__horizontal_well.csv", test / "00000000__typewell.csv", visible_slope_windows=[8, 16, 32], visible_backtest_fractions=[0.5, 0.7, 0.85], require_truth=False)["features"]
            self.assertEqual(train_features, test_features)

    def test_all_missing_gr_is_accepted_with_explicit_missingness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            horizontal = root / "aaaaaaaa__horizontal_well.csv"
            typewell = root / "aaaaaaaa__typewell.csv"
            self._write_horizontal(horizontal, 0, train=False, bad_mode="all_gr_missing")
            self._write_typewell(typewell, 0, train=False)
            features = extract_well_features(horizontal, typewell, visible_slope_windows=[8], visible_backtest_fractions=[0.5], require_truth=False)["features"]
            self.assertEqual(features["whole_gr_missing_fraction"], 1.0)
            self.assertIsNone(features["whole_gr_mean"])

    def test_noncontiguous_visibility_and_nonincreasing_md_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            typewell = root / "aaaaaaaa__typewell.csv"
            self._write_typewell(typewell, 0, train=False)
            gap = root / "gap__horizontal_well.csv"
            self._write_horizontal(gap, 0, train=False, bad_mode="gap")
            with self.assertRaises(DeploymentDataError):
                extract_well_features(gap, typewell, visible_slope_windows=[8], visible_backtest_fractions=[0.5])
            md = root / "md__horizontal_well.csv"
            self._write_horizontal(md, 0, train=False, bad_mode="md")
            with self.assertRaises(DeploymentDataError):
                extract_well_features(md, typewell, visible_slope_windows=[8], visible_backtest_fractions=[0.5])

    def test_one_hidden_row_reconstruction_is_centered_and_finite(self) -> None:
        values = reconstruct_hidden(100.0, 1, 3.0, 50.0)
        self.assertEqual(values, [103.0])

    def test_model_shape_and_nonfinite_coefficients_are_rejected(self) -> None:
        base = {
            "selected_features": ["x"],
            "feature_medians": [0.0],
            "feature_means": [0.0],
            "feature_scales": [1.0],
            "ridge_coefficients": [[1.0, 2.0]],
            "target_means": [0.0, 0.0],
            "target_scales": [1.0, 1.0],
        }
        broken = dict(base)
        broken["feature_scales"] = []
        with self.assertRaises(DeploymentDataError):
            predict_coefficients(broken, {"x": 1.0})
        nonfinite = dict(base)
        nonfinite["ridge_coefficients"] = [[math.inf, 2.0]]
        with self.assertRaises(DeploymentDataError):
            predict_coefficients(nonfinite, {"x": 1.0})

    def test_missing_typewell_and_bad_sample_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, train, test, sample = self._make_fixture(root)
            output, artifact = root / "out", root / "art"
            run_e004(root=root, train_dir=train, test_dir=test, sample_submission=sample, output_dir=output, artifact_dir=artifact, config=config, code_sha="b" * 40)
            model = json.loads((output / "model.json").read_text(encoding="utf-8"))
            missing_dir = root / "missing"
            missing_dir.mkdir()
            self._write_horizontal(missing_dir / "aaaaaaaa__horizontal_well.csv", 0, train=False)
            with self.assertRaises(DeploymentDataError):
                build_prediction_map(model, missing_dir)
            bad_sample = root / "bad_sample.csv"
            with bad_sample.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "tvt"], lineterminator="\n")
                writer.writeheader()
                writer.writerow({"id": "00000000_36", "tvt": 0})
                writer.writerow({"id": "00000000_36", "tvt": 0})
            with self.assertRaises(DeploymentDataError):
                write_submission(model, test, bad_sample, root / "bad.csv")


if __name__ == "__main__":
    unittest.main()
