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
            "SOURCE_COMMIT": "0" * 40,
            "MODEL_SHA256": "1" * 64,
            "BUNDLE_FILENAME": BUILDER.BUNDLE_FILENAME,
            "BUNDLE_SHA256": "2" * 64,
            "BUNDLE_BYTES": 123,
            "INPUT_RECEIPT_FILENAME": BUILDER.INPUT_RECEIPT_FILENAME,
            "INPUT_RECEIPT_SHA256": "3" * 64,
            "DATASET_REF": BUILDER.DEFAULT_DATASET_REF,
            "DATASET_VERSION": 1,
            "TRAIN_WELLS": BUILDER.TRAIN_WELLS,
            "TRAIN_SIGNATURE": BUILDER.TRAIN_SIGNATURE,
            "VISIBLE_FIXTURE": {
                "wells": 3,
                "rows": 10,
                "known_rows": 7,
                "hidden_rows": 3,
                "sample_rows": 3,
                "well_ids_sha256": "4" * 64,
                "sample_id_order_sha256": "5" * 64,
                "input_files_signature": "6" * 64,
            },
            "THREAD_LIMIT": 2,
            "RESULT_ARCHIVE_FILENAME": BUILDER.RESULT_ARCHIVE_FILENAME,
            "OUTPUT_MANIFEST_FILENAME": BUILDER.OUTPUT_MANIFEST_FILENAME,
            "RUN_RECEIPT_FILENAME": BUILDER.RUN_RECEIPT_FILENAME,
            "WELL_DIAGNOSTICS_FILENAME": BUILDER.WELL_DIAGNOSTICS_FILENAME,
            "SUBMISSION_FILENAME": BUILDER.SUBMISSION_FILENAME,
        }

    def test_notebook_is_static_user_run_contract(self):
        notebook = BUILDER.notebook_payload(self.contract())
        receipt = BUILDER.static_validate_notebook(notebook)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["code_cells"], 4)
        self.assertEqual([cell["execution_count"] for cell in notebook["cells"] if cell["cell_type"] == "code"], [None] * 4)
        metadata = notebook["metadata"]["rogii"]
        self.assertTrue(metadata["user_execution_required"])
        self.assertFalse(metadata["assistant_execution_authorized"])
        self.assertFalse(metadata["submission_authorized"])
        self.assertFalse(metadata["internet_required"])
        self.assertEqual(metadata["maximum_threads"], 2)
        self.assertEqual(
            metadata["expected_outputs"],
            [
                BUILDER.RUN_RECEIPT_FILENAME,
                BUILDER.OUTPUT_MANIFEST_FILENAME,
                BUILDER.RESULT_ARCHIVE_FILENAME,
                BUILDER.SUBMISSION_FILENAME,
                BUILDER.WELL_DIAGNOSTICS_FILENAME,
            ],
        )

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
        self.assertIn("false", source)

    def test_bundle_contract_uses_neutral_archive_and_committed_sources(self):
        self.assertTrue(BUILDER.BUNDLE_FILENAME.endswith(".zip.bin"))
        paths = BUILDER.bundle_paths(ROOT)
        self.assertIn("src/rogii_validation/e011_inference.py", paths)
        self.assertIn("experiments/E011/deployment/model.json", paths)
        self.assertIn("experiments/E011/deployment/feature_parity.json", paths)
        self.assertIn("experiments/E011/deployment/edge_case_receipt.json", paths)
        self.assertIn("experiments/E011/deployment/pseudo_hidden_benchmark.json", paths)
        self.assertEqual(len(paths), len(set(paths)))

    def test_archive_member_validation_fails_closed(self):
        for value in ("", "/absolute", "../escape", "a/../b", "a//b", "./a"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BUILDER.safe_member_name(value)
        self.assertEqual(BUILDER.safe_member_name("src/rogii_validation/e011_inference.py"), "src/rogii_validation/e011_inference.py")

    def test_current_model_and_receipt_are_self_consistent(self):
        model_path = ROOT / "experiments/E011/deployment/model.json"
        receipt_path = ROOT / "experiments/E011/deployment/model_fit_receipt.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["model_sha256"], BUILDER.sha256_file(model_path))
        self.assertEqual(model["model_name"], "spline4_ridge_equal_s075")
        self.assertEqual(model["source_identities"]["e011_inference_sha256"], BUILDER.sha256_file(ROOT / "src/rogii_validation/e011_inference.py"))


if __name__ == "__main__":
    unittest.main()
