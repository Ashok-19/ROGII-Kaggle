import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation import DataValidationError, generate_fold_maps, run_e001, scan_profiles


FIELDS = ["MD", "Z", "TVT", "TVT_input"]


def write_well(path: Path, well_index: int, *, reappear: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        base = 100.0 + well_index * 10.0
        for row_index in range(8):
            target = base + float(max(0, row_index - 2))
            visible = f"{target:.1f}" if row_index < 3 else ""
            if reappear and row_index == 6:
                visible = f"{target:.1f}"
            writer.writerow(
                {
                    "MD": 1000.0 + row_index,
                    "Z": -900.0 - well_index - row_index * 0.1,
                    "TVT": target,
                    "TVT_input": visible,
                }
            )


class E001ValidationTests(unittest.TestCase):
    def make_dataset(self, root: Path, wells: int = 5) -> Path:
        train = root / "data" / "train"
        train.mkdir(parents=True)
        for index in range(wells):
            write_well(train / f"{index:08x}__horizontal_well.csv", index)
        return train

    def test_profile_and_fold_maps_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = self.make_dataset(Path(tmp))
            profiles, data = scan_profiles(train)
            first = generate_fold_maps(profiles, data["data_signature"])
            second = generate_fold_maps(profiles, data["data_signature"])
            self.assertEqual(first, second)
            self.assertEqual(len({item["fingerprint"] for item in first}), 5)
            for fold_map in first:
                self.assertEqual(set(fold_map["assignments"]), {profile.well_id for profile in profiles})

    def test_full_fixture_run_passes_and_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self.make_dataset(root)
            first_folds = root / "folds_a"
            second_folds = root / "folds_b"
            first_output = root / "out_a"
            second_output = root / "out_b"
            first = run_e001(
                train_dir=train,
                fold_dir=first_folds,
                output_dir=first_output,
                expected_wells=5,
                expected_baseline_rmse=None,
            )
            second = run_e001(
                train_dir=train,
                fold_dir=second_folds,
                output_dir=second_output,
                expected_wells=5,
                expected_baseline_rmse=None,
            )
            self.assertEqual(first["status"], "pass")
            self.assertEqual(first, second)
            self.assertAlmostEqual(first["actual_target_metrics"]["last_known_tvt"]["rmse"], math.sqrt(11.0))
            self.assertLess(first["actual_target_metrics"]["last_known_tvt"]["datum_trend_removed_rmse"], 1e-12)
            for name in ("summary.json", "fold_metrics.csv", "data_profile.csv", "well_metrics.csv", "control_metrics.csv", "artifact_manifest.json"):
                self.assertEqual((first_output / name).read_bytes(), (second_output / name).read_bytes(), name)
            for version in range(1, 6):
                self.assertEqual((first_folds / f"v{version}.json").read_bytes(), (second_folds / f"v{version}.json").read_bytes())
            manifest = json.loads((first_output / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["fold_files"]), 5)

    def test_official_artifacts_match_manifest(self):
        manifest_path = ROOT / "experiments" / "E001" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "promoted")
        self.assertEqual(len(manifest["runs"]), 1)
        run = manifest["runs"][0]
        self.assertEqual(run["git_sha"], "d2b0066c36988c5f8dcb77c54350bc641e167f82")
        for artifact in run["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertTrue(path.exists(), artifact["path"])
            self.assertEqual(path.stat().st_size, artifact["bytes"], artifact["path"])
            import hashlib

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, artifact["sha256"], artifact["path"])
        summary = json.loads((ROOT / "experiments" / "E001" / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["data"]["data_signature"], "6ebe65b403f80fefd97dcd7bbfce7314252e779a1837c97364cf55761590fe77")

    def test_noncontiguous_visibility_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "data" / "train"
            train.mkdir(parents=True)
            write_well(train / "bad00000__horizontal_well.csv", 0, reappear=True)
            with self.assertRaises(DataValidationError):
                scan_profiles(train)


if __name__ == "__main__":
    unittest.main()
