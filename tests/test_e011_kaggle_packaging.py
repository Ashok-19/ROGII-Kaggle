import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = importlib.util.spec_from_file_location(
    "build_e011_notebook", ROOT / "tools/build_e011_notebook.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class E011KagglePackagingTests(unittest.TestCase):
    def contract(self):
        return {
            "BUNDLE_FILENAME": BUILDER.BUNDLE_FILENAME,
            "DATASET_REF": BUILDER.DEFAULT_DATASET_REF,
            "DATASET_VERSION": 1,
            "RUN_RECEIPT_FILENAME": BUILDER.RUN_RECEIPT_FILENAME,
            "WELL_DIAGNOSTICS_FILENAME": BUILDER.WELL_DIAGNOSTICS_FILENAME,
            "SUBMISSION_FILENAME": BUILDER.SUBMISSION_FILENAME,
        }

    def test_notebook_is_simple_user_run_contract(self):
        notebook = BUILDER.notebook_payload(self.contract())
        receipt = BUILDER.static_validate_notebook(notebook)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["code_cells"], 2)
        self.assertEqual(
            [
                cell["execution_count"]
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ],
            [None, None],
        )
        metadata = notebook["metadata"]["rogii"]
        self.assertTrue(metadata["user_execution_required"])
        self.assertFalse(metadata["assistant_execution_authorized"])
        self.assertFalse(metadata["submission_authorized"])
        self.assertEqual(
            metadata["expected_outputs"],
            [
                BUILDER.SUBMISSION_FILENAME,
                BUILDER.RUN_RECEIPT_FILENAME,
                BUILDER.WELL_DIAGNOSTICS_FILENAME,
            ],
        )

    def test_notebook_has_only_correctness_critical_runtime_checks(self):
        notebook = BUILDER.notebook_payload(self.contract())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ).lower()
        self.assertIn("sample_submission.csv", source)
        self.assertIn("write_e011_submission", source)
        self.assertIn("submission.csv", source)
        self.assertNotIn("threadpool", source)
        self.assertNotIn("omp_num_threads", source)
        self.assertNotIn("internet_expected", source)
        self.assertNotIn("accelerator_expected", source)
        self.assertNotIn("sha256", source)
        self.assertNotIn("scan_profiles", source)

    def test_notebook_contains_no_submission_operation(self):
        notebook = BUILDER.notebook_payload(self.contract())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ).lower()
        self.assertNotIn("kaggle competitions submit", source)
        self.assertNotIn("competitions submit", source)
        self.assertNotIn("kaggle_create_code_competition_submission", source)
        self.assertIn("kaggle_submission_made", source)

    def test_bundle_contains_runtime_code_and_model(self):
        paths = BUILDER.bundle_paths(ROOT)
        self.assertIn("src/rogii_validation/e011_inference.py", paths)
        self.assertIn("src/rogii_validation/e006_inference.py", paths)
        self.assertIn("experiments/E011/deployment/model.json", paths)
        self.assertEqual(len(paths), len(set(paths)))

    def test_archive_member_validation_rejects_unsafe_paths(self):
        for value in ("", "/absolute", "../escape", "a/../b", "a//b", "./a"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BUILDER.safe_member_name(value)
        self.assertEqual(
            BUILDER.safe_member_name("src/rogii_validation/e011_inference.py"),
            "src/rogii_validation/e011_inference.py",
        )

    def test_current_model_is_loadable(self):
        model_path = ROOT / "experiments/E011/deployment/model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        self.assertEqual(model["model_name"], "spline4_ridge_equal_s075")
        BUILDER.validate_e011_model(model)


if __name__ == "__main__":
    unittest.main()
