from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.gr_path import (
    ALL_CANDIDATES,
    DataValidationError,
    TypewellCurve,
    candidate_paths,
    read_well,
    validate_e005_config,
    validate_submission_ids,
)


class E005SyntheticCase(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "experiments/E005/config.json").read_text(encoding="utf-8"))

    def _write_case(
        self,
        root: Path,
        *,
        well_id: str = "deadbeef",
        known: int = 40,
        hidden: int = 80,
        md_values: list[float] | None = None,
        visible_gr_missing: bool = False,
        hidden_missing: set[int] | None = None,
        all_gr_missing: bool = False,
        periodic: bool = False,
        flat_typewell: bool = False,
        typewell_min: float = 0.0,
        typewell_max: float = 300.0,
        typewell_rows: int = 601,
        noncontiguous_input: bool = False,
    ) -> tuple[Path, Path, list[float]]:
        total = known + hidden
        md = md_values or [float(index) for index in range(total)]
        self.assertEqual(len(md), total)
        truth = [100.0 + 0.08 * index + 2.0 * math.sin(index / 29.0) for index in range(total)]

        def signal(tvt: float) -> float:
            if flat_typewell:
                return 50.0
            if periodic:
                return 50.0 + 12.0 * math.sin(tvt / 3.5)
            return 50.0 + 10.0 * math.sin(tvt / 7.0) + 3.0 * math.cos(tvt / 17.0)

        horizontal = root / f"{well_id}__horizontal_well.csv"
        typewell = root / f"{well_id}__typewell.csv"
        with typewell.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["TVT", "GR", "Geology"], lineterminator="\n")
            writer.writeheader()
            for index in range(typewell_rows):
                fraction = index / max(1, typewell_rows - 1)
                tvt = typewell_min + fraction * (typewell_max - typewell_min)
                writer.writerow({"TVT": tvt, "GR": signal(tvt), "Geology": "forbidden-sentinel"})
        hidden_missing = hidden_missing or set()
        with horizontal.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input", "ANCC"], lineterminator="\n")
            writer.writeheader()
            for index in range(total):
                hidden_index = index - known
                missing_gr = all_gr_missing or (index < known and visible_gr_missing) or (index >= known and hidden_index in hidden_missing)
                tvt_input = truth[index] if index < known else ""
                if noncontiguous_input and index == known + 2:
                    tvt_input = truth[index]
                writer.writerow({
                    "MD": md[index],
                    "X": 1000.0 + index,
                    "Y": 2000.0 + 0.5 * index,
                    "Z": -9000.0 + 0.2 * index,
                    "GR": "" if missing_gr else signal(truth[index]),
                    "TVT": truth[index],
                    "TVT_input": tvt_input,
                    "ANCC": 999999.0,
                })
        return horizontal, typewell, truth

    def _paths(self, horizontal: Path, typewell: Path, truth: list[float]):
        well = read_well(horizontal, typewell, require_truth=True)
        base = [value + 4.0 for value in truth[well.known_rows :]]
        return well, base, candidate_paths(well, base, well.typewell, self.config)

    def test_frozen_config_validation_rejects_malformed_candidate_arrays(self) -> None:
        validate_e005_config(self.config)
        cases = []
        bad = copy.deepcopy(self.config); bad["candidate_order"] = list(reversed(bad["candidate_order"])); cases.append(bad)
        bad = copy.deepcopy(self.config); bad["eligible_candidates"] = ["align_affine"]; cases.append(bad)
        bad = copy.deepcopy(self.config); bad["alignment"]["datum_offsets_ft"] = []; cases.append(bad)
        bad = copy.deepcopy(self.config); bad["alignment"]["toe_offsets_ft"] = [0.0, 0.0]; cases.append(bad)
        bad = copy.deepcopy(self.config); bad["trellis"]["offset_states_ft"] = [0.0, -5.0]; cases.append(bad)
        bad = copy.deepcopy(self.config); bad["diagnostic_blend_weights"] = [0.0]; cases.append(bad)
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(DataValidationError):
                    validate_e005_config(config)

    def test_all_missing_gr_and_missing_visible_prefix_abstain_exactly_to_e004(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kwargs, expected_reason in (({"all_gr_missing": True}, "low_hidden_gr_coverage"), ({"visible_gr_missing": True}, "invalid_visible_calibration")):
                horizontal, typewell, truth = self._write_case(root, well_id=("aaaabbbb" if kwargs.get("all_gr_missing") else "ccccdddd"), **kwargs)
                well, base, (paths, diagnostics) = self._paths(horizontal, typewell, truth)
                self.assertEqual(diagnostics["fallback_reason"], expected_reason)
                for candidate in ("align_affine", "align_visible_path", "pf_gr_path", "trellis_gr_path", "no_gr_geometry_prefix"):
                    self.assertEqual(paths[candidate], base)
                self.assertEqual(len(paths["last_known_tvt"]), well.hidden_rows)

    def test_active_periodic_duplicate_pattern_candidates_are_deterministic_finite_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            horizontal, typewell, truth = self._write_case(root, periodic=True, hidden=140)
            well, base, first = self._paths(horizontal, typewell, truth)
            second = candidate_paths(well, base, TypewellCurve.read(typewell), self.config)
            self.assertEqual(first, second)
            paths, diagnostics = first
            self.assertFalse(diagnostics["fallback_reason"])
            self.assertEqual(paths["duplicate_align"], paths["align_visible_path"])
            lower = well.typewell.minimum_tvt - self.config["alignment"]["typewell_margin_ft"]
            upper = well.typewell.maximum_tvt + self.config["alignment"]["typewell_margin_ft"]
            for candidate in ALL_CANDIDATES:
                if candidate == "oracle_target":
                    continue
                self.assertEqual(len(paths[candidate]), well.hidden_rows)
                self.assertTrue(all(math.isfinite(value) for value in paths[candidate]))
            for candidate in ("align_affine", "align_visible_path", "pf_gr_path", "trellis_gr_path"):
                self.assertTrue(all(lower <= value <= upper for value in paths[candidate]))

    def test_flat_typewell_and_no_overlap_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for well_id, kwargs, reason in (
                ("11111111", {"flat_typewell": True}, "flat_typewell_gr"),
                ("22222222", {"typewell_min": 500.0, "typewell_max": 600.0}, "invalid_visible_calibration"),
            ):
                horizontal, typewell, truth = self._write_case(root, well_id=well_id, **kwargs)
                _, base, (paths, diagnostics) = self._paths(horizontal, typewell, truth)
                self.assertEqual(diagnostics["fallback_reason"], reason)
                self.assertEqual(paths["trellis_gr_path"], base)

    def test_long_missing_interval_short_single_and_long_suffixes_remain_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = [
                ("33333333", 1, set()),
                ("44444444", 3, set()),
                ("55555555", 240, set(range(40, 190))),
            ]
            for well_id, hidden, missing in cases:
                horizontal, typewell, truth = self._write_case(root, well_id=well_id, hidden=hidden, hidden_missing=missing)
                well, _, (paths, _) = self._paths(horizontal, typewell, truth)
                for candidate, values in paths.items():
                    self.assertEqual(len(values), well.hidden_rows, candidate)
                    self.assertTrue(all(math.isfinite(value) for value in values), candidate)

    def test_nonuniform_md_is_supported_but_duplicate_md_and_noncontiguous_prefix_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            total = 120
            md = [0.0]
            for index in range(1, total):
                md.append(md[-1] + (0.5 if index % 3 else 1.75))
            horizontal, typewell, truth = self._write_case(root, well_id="66666666", md_values=md)
            well, _, (paths, _) = self._paths(horizontal, typewell, truth)
            self.assertEqual(len(paths["no_gr_typewell_affine"]), well.hidden_rows)
            duplicate = list(md); duplicate[20] = duplicate[19]
            horizontal, typewell, _ = self._write_case(root, well_id="77777777", md_values=duplicate)
            with self.assertRaises(DataValidationError):
                read_well(horizontal, typewell, require_truth=True)
            horizontal, typewell, _ = self._write_case(root, well_id="88888888", noncontiguous_input=True)
            with self.assertRaises(DataValidationError):
                read_well(horizontal, typewell, require_truth=True)

    def test_very_short_typewell_boundary_clipping_and_nonfinite_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            horizontal, typewell, truth = self._write_case(root, typewell_min=95.0, typewell_max=120.0, typewell_rows=2)
            well = read_well(horizontal, typewell, require_truth=True)
            base = [10000.0] * well.hidden_rows
            paths, _ = candidate_paths(well, base, well.typewell, self.config)
            upper = well.typewell.maximum_tvt + self.config["alignment"]["typewell_margin_ft"]
            for candidate in ("align_affine", "align_visible_path", "pf_gr_path", "trellis_gr_path"):
                self.assertTrue(all(value <= upper for value in paths[candidate]))
            bad = list(base); bad[-1] = float("nan")
            with self.assertRaises(DataValidationError):
                candidate_paths(well, bad, well.typewell, self.config)

    def test_missing_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            horizontal, typewell, _ = self._write_case(root)
            typewell.unlink()
            with self.assertRaises(DataValidationError):
                read_well(horizontal, typewell, require_truth=True)

    def test_sample_submission_contract_detects_duplicate_missing_extra_wrong_order_visible_and_nonfinite_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = root / "sample_submission.csv"
            with sample.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "tvt"], lineterminator="\n")
                writer.writeheader()
                writer.writerow({"id": "a_4", "tvt": 0})
                writer.writerow({"id": "a_5", "tvt": 0})
            predictions = {"a_4": 1.0, "a_5": 2.0}
            self.assertEqual(validate_submission_ids(sample, predictions, ["a_4", "a_5"]), ["a_4", "a_5"])
            with self.assertRaises(DataValidationError):
                validate_submission_ids(sample, predictions, ["a_5", "a_4"])
            with self.assertRaises(DataValidationError):
                validate_submission_ids(sample, {"a_4": 1.0})
            with self.assertRaises(DataValidationError):
                validate_submission_ids(sample, {"a_4": 1.0, "a_5": 2.0, "a_3": 3.0})
            with self.assertRaises(DataValidationError):
                validate_submission_ids(sample, {"a_4": 1.0, "a_5": float("inf")})
            duplicate = root / "duplicate.csv"
            duplicate.write_text("id,tvt\na_4,0\na_4,0\n", encoding="utf-8")
            with self.assertRaises(DataValidationError):
                validate_submission_ids(duplicate, {"a_4": 1.0})


if __name__ == "__main__":
    unittest.main()
