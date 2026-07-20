from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/build_e010_kaggle.py"
SPEC = importlib.util.spec_from_file_location("build_e010_kaggle", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class E010KaggleBundleTests(unittest.TestCase):
    def test_current_v2_contract_constants(self):
        self.assertEqual(MODULE.DEFAULT_SOURCE_COMMIT, "49ca3995269638026561c5593ad2628f8f08a4e9")
        self.assertEqual(MODULE.DEFAULT_DATASET_VERSION, 2)
        self.assertEqual(MODULE.BUNDLE_FILENAME, "rogii-e010-inputs-v2.zip.bin")
        self.assertEqual(MODULE.INPUT_RECEIPT_FILENAME, "e010-input-receipt-v2.json")
        self.assertEqual(MODULE.RESULT_ARCHIVE_FILENAME, "rogii-e010-results-v2.zip")

    def test_safe_member_contract(self):
        self.assertEqual(MODULE.safe_member_name("src/rogii_validation/harness.py"), "src/rogii_validation/harness.py")
        for bad in ("", "/absolute", "../escape", "a/../../b", "a/./b"):
            with self.assertRaises(ValueError):
                MODULE.safe_member_name(bad)

    def test_deterministic_small_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "a/one.txt").write_text("one\n", encoding="utf-8")
            (root / "two.bin").write_bytes(b"two\x00")
            files = []
            for relative in ("a/one.txt", "two.bin"):
                path = root / relative
                files.append({"path": relative, "bytes": path.stat().st_size, "sha256": MODULE.sha256_file(path)})
            manifest = {"schema_version": 1, "files": files}
            first = root / "first.zip.bin"
            second = root / "second.zip.bin"
            MODULE.write_deterministic_bundle(root, first, manifest)
            MODULE.write_deterministic_bundle(root, second, manifest)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.namelist(), ["bundle_manifest.json", "a/one.txt", "two.bin"])
                self.assertEqual(json.loads(archive.read("bundle_manifest.json")), manifest)
                self.assertEqual(archive.read("a/one.txt"), b"one\n")

    def test_notebook_contract_is_private_cpu_two_thread_and_fail_closed(self):
        payload = MODULE.notebook_payload(
            source_commit="a" * 40,
            config_sha="b" * 64,
            bundle_sha="c" * 64,
            bundle_bytes=123,
            receipt_sha="d" * 64,
            dataset_ref="owner/private-inputs",
            dataset_version=7,
            data_signature="e" * 64,
            expected_wells=773,
        )
        metadata = payload["metadata"]["rogii"]
        self.assertFalse(metadata["internet_required"])
        self.assertEqual(metadata["accelerator"], "CPU")
        self.assertEqual(metadata["maximum_threads"], 2)
        self.assertEqual(metadata["expected_outputs"], [
            MODULE.RUN_RECEIPT_FILENAME,
            MODULE.OUTPUT_MANIFEST_FILENAME,
            MODULE.RESULT_ARCHIVE_FILENAME,
        ])
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"] if cell["cell_type"] == "code")
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        ):
            self.assertIn(name, source)
        self.assertIn("threadpool_limits", source)
        self.assertIn("scan_profiles", source)
        self.assertIn("run_e010", source)
        self.assertIn("raise RuntimeError", source)
        self.assertIn("submission_made", source)

    def test_official_source_list_contains_frozen_parents_and_package(self):
        config = json.loads((ROOT / "experiments/E010/config.json").read_text(encoding="utf-8"))
        paths = MODULE.source_paths(ROOT, config)
        self.assertIn("src/rogii_validation/nonlinear_selector.py", paths)
        self.assertIn("src/rogii_validation/e010_paths.py", paths)
        self.assertIn("experiments/E010/config.json", paths)
        self.assertEqual({str(item["path"]) for item in config["parents"].values()} - set(paths), set())
        self.assertEqual(set(config["fold_files"]) - set(paths), set())


if __name__ == "__main__":
    unittest.main()
