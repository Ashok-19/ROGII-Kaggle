import copy
import csv
import json
import math
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.e004_inference import DeploymentDataError
from rogii_validation.e011_inference import (
    build_e011_prediction_map,
    load_e011_model,
    predict_e011_well,
    predict_spline_coefficients,
    profile_e011_test_input,
    spline4_correction,
    validate_e011_model,
    write_e011_submission,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "experiments/E011/deployment/model.json"


class E011InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_e011_model(MODEL_PATH)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.test_dir = self.root / "test"
        self.test_dir.mkdir()
        self.well_id = "1234abcd"
        self.horizontal = self.test_dir / f"{self.well_id}__horizontal_well.csv"
        self.typewell = self.test_dir / f"{self.well_id}__typewell.csv"
        self.sample = self.root / "sample_submission.csv"
        self.output = self.root / "submission.csv"
        self._write_horizontal()
        self._write_typewell()
        self._write_sample([f"{self.well_id}_{index}" for index in range(5, 8)])

    def tearDown(self):
        self.temp.cleanup()

    def _write_horizontal(self, *, rows=None, fieldnames=None):
        names = fieldnames or ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT", "ANCC"]
        values = rows or [
            [100, 1000, 2000, -100, 50, 200, 200, -10],
            [101, 1001, 2000.5, -101, 51, 201, 201, -11],
            [102, 1002, 2001, -102, 49, 202, 202, -12],
            [103, 1003, 2001.5, -103, 52, 203, 203, -13],
            [104, 1004, 2002, -104, 50, 204, 204, -14],
            [105, 1005, 2002.5, -105, 53, "", 9999, 9999],
            [106, 1006, 2003, -106, 54, "", -9999, -9999],
            [107, 1007, 2003.5, -107, 55, "", 1234, 1234],
        ]
        with self.horizontal.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(names)
            writer.writerows(values)

    def _write_typewell(self, *, fieldnames=None, rows=None):
        names = fieldnames or ["TVT", "GR", "Geology"]
        values = rows or [[180 + index, 45 + (index % 7), "ignored"] for index in range(40)]
        with self.typewell.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(names)
            writer.writerows(values)

    def _write_sample(self, ids, *, fieldnames=("id", "tvt")):
        with self.sample.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fieldnames)
            for key in ids:
                writer.writerow([key, 0])

    def test_repository_package_receipts_pass(self):
        parity = json.loads((ROOT / "experiments/E011/deployment/feature_parity.json").read_text())
        local = json.loads((ROOT / "experiments/E011/deployment/local_parity.json").read_text())
        fit = json.loads((ROOT / "experiments/E011/deployment/model_fit_receipt.json").read_text())
        self.assertEqual(parity["status"], "PASS")
        self.assertEqual(parity["wells"], 773)
        self.assertEqual(parity["features"], 142)
        self.assertEqual(parity["mismatched_cells"], 0)
        self.assertTrue(local["direct_and_prototype_notebook_byte_identical"])
        self.assertEqual(local["status"], "HISTORICAL_LOCAL_PROTOTYPE_PASS")
        self.assertFalse(local["canonical_kaggle_notebook_executed"])
        self.assertFalse(local["canonical_kaggle_parity_established"])
        self.assertEqual(fit["status"], "PASS")
        self.assertLessEqual(fit["runtime_coefficient_max_abs_delta"], 1e-10)

    def test_model_contract_rejects_tampering(self):
        validate_e011_model(self.model)
        mutations = []
        for key, value in (
            ("schema_version", 2),
            ("model_name", "other"),
            ("representation", "spline5"),
            ("training_wells", 772),
            ("ridge_alpha", 2.0),
            ("shrinkage", 1.0),
            ("coefficient_absolute_bound_ft", 81.0),
            ("internet_required", True),
            ("external_artifacts_required", True),
            ("surfaces_required", True),
        ):
            item = copy.deepcopy(self.model)
            item[key] = value
            mutations.append(item)
        missing_feature = copy.deepcopy(self.model)
        missing_feature["selected_features"] = missing_feature["selected_features"][:-1]
        mutations.append(missing_feature)
        nan_coefficient = copy.deepcopy(self.model)
        nan_coefficient["ridge_coefficients"][0][0] = float("nan")
        mutations.append(nan_coefficient)
        for item in mutations:
            with self.subTest(keys={key: item.get(key) for key in ("schema_version", "model_name", "representation")}):
                with self.assertRaises(DeploymentDataError):
                    validate_e011_model(item)

    def test_spline_edge_cases_and_exact_zero_fallback(self):
        self.assertEqual(spline4_correction(1, [1, 2, 3, 4]), [0.0])
        self.assertEqual(spline4_correction(5, [10, 20, 30, 40]), [0.0, 10.0, 20.0, 30.0, 40.0])
        zeros = spline4_correction(101, [0, 0, 0, 0])
        self.assertEqual(zeros, [0.0] * 101)
        with self.assertRaises(DeploymentDataError):
            spline4_correction(0, [0, 0, 0, 0])
        with self.assertRaises(DeploymentDataError):
            spline4_correction(4, [0, 0, 0])
        with self.assertRaises(DeploymentDataError):
            spline4_correction(4, [0, math.inf, 0, 0])

    def test_coefficient_prediction_imputes_missing_and_clips(self):
        values = predict_spline_coefficients(self.model, {})
        self.assertEqual(len(values), 4)
        self.assertTrue(all(math.isfinite(value) and abs(value) <= 80 for value in values))
        features = {name: float("nan") for name in self.model["selected_features"]}
        self.assertEqual(values, predict_spline_coefficients(self.model, features))
        forced = copy.deepcopy(self.model)
        forced["ridge_scaled_intercept"] = [1e9] * 4
        self.assertEqual(predict_spline_coefficients(forced, {}), (80.0, 80.0, 80.0, 80.0))

    def test_minimal_short_suffix_and_missing_horizontal_gr_are_safe(self):
        rows = [
            [100, 0, 0, -10, "", 20, 20, 1],
            [101, 1, 0, -11, "", 21, 21, 1],
            [102, 2, 0, -12, "", 22, 22, 1],
            [103, 3, 0, -13, "", "", 9000, 1],
        ]
        self._write_horizontal(rows=rows)
        self._write_sample([f"{self.well_id}_3"])
        values, detail = predict_e011_well(self.model, self.horizontal, self.typewell)
        self.assertEqual(len(values), 1)
        self.assertTrue(math.isfinite(values[0]))
        self.assertTrue(detail["e006_fallback_reason"])
        self.assertTrue(detail["selfcorr_fallback_reason"])
        result = write_e011_submission(self.model, self.test_dir, self.sample, self.output)
        self.assertEqual(result["rows"], 1)

    def test_hidden_truth_and_formation_surfaces_cannot_change_prediction(self):
        first, _ = predict_e011_well(self.model, self.horizontal, self.typewell)
        with self.horizontal.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for row in rows[6:]:
            row[6] = str(float(row[6]) * -777.0)
            row[7] = str(float(row[7]) + 1e7)
        with self.horizontal.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        second, _ = predict_e011_well(self.model, self.horizontal, self.typewell)
        self.assertEqual(first, second)

    def test_absolute_xy_translation_cannot_change_prediction(self):
        first, _ = predict_e011_well(self.model, self.horizontal, self.typewell)
        with self.horizontal.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for row in rows[1:]:
            row[1] = str(float(row[1]) + 1_000_000)
            row[2] = str(float(row[2]) - 2_000_000)
        with self.horizontal.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        second, _ = predict_e011_well(self.model, self.horizontal, self.typewell)
        self.assertEqual(first, second)

    def test_test_input_profile_uses_only_legal_columns(self):
        first = profile_e011_test_input(self.test_dir, self.sample)
        self.assertEqual(first["wells"], 1)
        self.assertEqual(first["hidden_rows"], 3)
        self.assertEqual(first["sample_rows"], 3)
        with self.horizontal.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for row in rows[6:]:
            row[6] = str(float(row[6]) * -12345.0)
            row[7] = str(float(row[7]) + 987654.0)
        with self.horizontal.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        second = profile_e011_test_input(self.test_dir, self.sample)
        self.assertEqual(first["input_files_signature"], second["input_files_signature"])
        self.assertEqual(first["sample_id_order_sha256"], second["sample_id_order_sha256"])

    def test_repeated_execution_is_deterministic(self):
        first = write_e011_submission(self.model, self.test_dir, self.sample, self.output)
        first_bytes = self.output.read_bytes()
        second_path = self.root / "second.csv"
        second = write_e011_submission(self.model, self.test_dir, self.sample, second_path)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first_bytes, second_path.read_bytes())
        self.assertTrue(all(math.isfinite(value) for value in (first["minimum_prediction"], first["maximum_prediction"])))

    def test_sample_order_is_preserved_but_set_must_match(self):
        reverse = [f"{self.well_id}_{index}" for index in reversed(range(5, 8))]
        self._write_sample(reverse)
        write_e011_submission(self.model, self.test_dir, self.sample, self.output)
        with self.output.open(newline="", encoding="utf-8") as handle:
            self.assertEqual([row["id"] for row in csv.DictReader(handle)], reverse)
        self._write_sample([reverse[0], reverse[0], reverse[1]])
        with self.assertRaises(DeploymentDataError):
            write_e011_submission(self.model, self.test_dir, self.sample, self.output)
        self._write_sample(reverse[:-1])
        with self.assertRaises(DeploymentDataError):
            write_e011_submission(self.model, self.test_dir, self.sample, self.output)
        self._write_sample(reverse + ["ffffffff_0"])
        with self.assertRaises(DeploymentDataError):
            write_e011_submission(self.model, self.test_dir, self.sample, self.output)
        self._write_sample(reverse, fieldnames=("tvt", "id"))
        with self.assertRaises(DeploymentDataError):
            write_e011_submission(self.model, self.test_dir, self.sample, self.output)

    def test_required_schema_and_boundary_fail_closed(self):
        with self.horizontal.open(newline="", encoding="utf-8") as handle:
            valid_rows = list(csv.reader(handle))[1:]
        cases = []
        missing_md_names = ["X", "Y", "Z", "GR", "TVT_input", "TVT", "ANCC"]
        missing_md_rows = [row[1:] for row in valid_rows]
        cases.append((missing_md_names, missing_md_rows))
        nonfinite = copy.deepcopy(valid_rows)
        nonfinite[2][0] = "nan"
        cases.append((None, nonfinite))
        duplicate_md = copy.deepcopy(valid_rows)
        duplicate_md[3][0] = duplicate_md[2][0]
        cases.append((None, duplicate_md))
        noncontiguous = copy.deepcopy(valid_rows)
        noncontiguous[5][5] = ""
        noncontiguous[6][5] = "206"
        cases.append((None, noncontiguous))
        too_short = copy.deepcopy(valid_rows)
        too_short[2][5] = ""
        cases.append((None, too_short))
        no_hidden = copy.deepcopy(valid_rows)
        for index, row in enumerate(no_hidden):
            row[5] = str(200 + index)
        cases.append((None, no_hidden))
        for fieldnames, rows in cases:
            self._write_horizontal(fieldnames=fieldnames, rows=rows)
            with self.subTest(first_row=rows[0]):
                with self.assertRaises(DeploymentDataError):
                    build_e011_prediction_map(self.model, self.test_dir)

    def test_typewell_contract_fail_closed(self):
        self.typewell.unlink()
        with self.assertRaises(DeploymentDataError):
            build_e011_prediction_map(self.model, self.test_dir)
        self._write_typewell(fieldnames=["TVT"], rows=[[1], [2]])
        with self.assertRaises(DeploymentDataError):
            build_e011_prediction_map(self.model, self.test_dir)
        self._write_typewell(rows=[[1, "", ""], [2, "", ""]])
        with self.assertRaises(DeploymentDataError):
            build_e011_prediction_map(self.model, self.test_dir)
        self._write_typewell(rows=[[1, 2, ""]])
        with self.assertRaises(DeploymentDataError):
            build_e011_prediction_map(self.model, self.test_dir)

    def test_partial_failure_does_not_emit_output(self):
        bad = "deadbeef"
        with self.horizontal.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        other = self.test_dir / f"{bad}__horizontal_well.csv"
        with other.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        # Deliberately omit the second typewell.
        self.assertFalse(self.output.exists())
        with self.assertRaises(DeploymentDataError):
            write_e011_submission(self.model, self.test_dir, self.sample, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
