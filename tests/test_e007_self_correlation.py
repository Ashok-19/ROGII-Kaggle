from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.fusion import BlendSufficient
from rogii_validation.harness import DataValidationError
from rogii_validation.self_correlation import (
    ALL_CANDIDATES,
    ELIGIBLE,
    MatchDiagnostic,
    SelfCorrDiagnostic,
    _bounded_path,
    _effective_weight,
    _fingerprint_distance,
    _fingerprints,
    _local_u_slope,
    _make_reference,
    _match_slopes,
    _parent_wells,
    _pseudo_metrics,
    _select_weights,
    read_horizontal_selfcorr,
    selfcorr_path,
    validate_e007_config,
)


class E007SelfCorrelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "experiments/E007/config.json").read_text(encoding="utf-8"))

    def _write_well(
        self,
        root: Path,
        *,
        well_id: str = "deadbeef",
        known: int = 520,
        hidden: int = 120,
        all_gr_missing: bool = False,
        hidden_gr_missing: bool = False,
        constant_gr: bool = False,
        noncontiguous: bool = False,
        duplicate_md: bool = False,
    ) -> tuple[Path, list[float]]:
        total = known + hidden
        u = [100.0]
        for index in range(1, total):
            gr_phase = math.sin(index / 17.0) + 0.4 * math.cos(index / 43.0)
            u.append(u[-1] + 0.018 + 0.015 * gr_phase)
        z = [-9000.0 + 0.01 * index for index in range(total)]
        tvt = [u_value - z_value for u_value, z_value in zip(u, z)]
        path = root / f"{well_id}__horizontal_well.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT", "ANCC"],
                lineterminator="\n",
            )
            writer.writeheader()
            for index in range(total):
                md = float(index)
                if duplicate_md and index == 25:
                    md = 24.0
                missing = all_gr_missing or (hidden_gr_missing and index >= known)
                gr = 50.0 if constant_gr else 50.0 + 12.0 * math.sin(index / 17.0) + 4.0 * math.cos(index / 43.0)
                tvt_input: float | str = tvt[index] if index < known else ""
                if noncontiguous and index == known + 3:
                    tvt_input = tvt[index]
                writer.writerow({
                    "MD": md,
                    "X": 1000.0 + index,
                    "Y": 2000.0 + 0.5 * index,
                    "Z": z[index],
                    "GR": "" if missing else gr,
                    "TVT_input": tvt_input,
                    "TVT": tvt[index],
                    "ANCC": 999999.0,
                })
        return path, tvt

    def test_frozen_config_and_candidate_order(self) -> None:
        validate_e007_config(self.config)
        self.assertEqual(self.config["candidate_order"], list(ALL_CANDIDATES))
        self.assertEqual(self.config["eligible_candidates"], list(ELIGIBLE))
        mutations = (
            lambda item: item["placement"].update(weight_grid=[0.0, 0.5, 0.5, 1.0]),
            lambda item: item["self_correlation"].update(nearest_neighbors=0),
            lambda item: item["self_correlation"].update(correction_soft_cap_ft=math.nan),
            lambda item: item["visible_only_design"].update(pseudo_boundary_fractions=[1.0]),
            lambda item: item["selection"].update(no_post_score_candidate_additions=False),
            lambda item: item["deployment"].update(internet=True),
            lambda item: item.update(fold_files=["folds/v1.json"] * 5),
        )
        for mutate in mutations:
            broken = copy.deepcopy(self.config)
            mutate(broken)
            with self.subTest(mutate=mutate):
                with self.assertRaises(DataValidationError):
                    validate_e007_config(broken)

    def test_horizontal_reader_rejects_duplicate_md_and_noncontiguous_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self._write_well(root, well_id="aaaaaaaa")
            well = read_horizontal_selfcorr(path)
            self.assertEqual(well.known_rows, 520)
            self.assertEqual(well.hidden_rows, 120)
            duplicate, _ = self._write_well(root, well_id="bbbbbbbb", duplicate_md=True)
            with self.assertRaises(DataValidationError):
                read_horizontal_selfcorr(duplicate)
            noncontiguous, _ = self._write_well(root, well_id="cccccccc", noncontiguous=True)
            with self.assertRaises(DataValidationError):
                read_horizontal_selfcorr(noncontiguous)

    def test_fingerprints_are_deterministic_and_distance_requires_common_features(self) -> None:
        values = [None if index % 11 == 0 else math.sin(index / 5.0) for index in range(120)]
        first = _fingerprints(values, self.config["self_correlation"])
        second = _fingerprints(values, self.config["self_correlation"])
        self.assertEqual(first, second)
        distance = _fingerprint_distance(first[50], first[51], 3)
        self.assertTrue(math.isfinite(distance) and distance >= 0.0)
        self.assertTrue(math.isinf(_fingerprint_distance((1.0, None), (1.0, None), 2)))
        constant = _fingerprints([5.0] * 100, self.config["self_correlation"])
        self.assertTrue(all(item == tuple() for item in constant))

    def test_local_slope_window_must_stay_inside_visible_boundary(self) -> None:
        values = [float(index) for index in range(50)]
        self.assertAlmostEqual(_local_u_slope(values, 20, 5), 1.0)
        with self.assertRaises(DataValidationError):
            _local_u_slope(values, 3, 5)
        with self.assertRaises(DataValidationError):
            _local_u_slope(values, 48, 5)

    def test_match_slopes_does_not_read_pseudo_hidden_tvt(self) -> None:
        total = 600
        template = 420
        gr = [50.0 + 10.0 * math.sin(index / 19.0) for index in range(total)]
        z = [-9000.0 + 0.01 * index for index in range(total)]
        tvt = [9100.0 + 0.02 * index + 0.1 * math.sin(index / 31.0) for index in range(total)]
        altered = list(tvt)
        for index in range(template, total):
            altered[index] += 1_000_000.0 + index
        first, first_detail = _match_slopes(
            gr=gr,
            visible_tvt=tvt,
            z=z,
            template_rows=template,
            total_rows=total,
            config=self.config["self_correlation"],
        )
        second, second_detail = _match_slopes(
            gr=gr,
            visible_tvt=altered,
            z=z,
            template_rows=template,
            total_rows=total,
            config=self.config["self_correlation"],
        )
        self.assertEqual(first, second)
        self.assertEqual(first_detail, second_detail)

    def test_active_selfcorr_path_is_deterministic_finite_and_soft_capped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, tvt = self._write_well(Path(directory))
            well = read_horizontal_selfcorr(path)
            base = [tvt[well.known_rows - 1]] * well.hidden_rows
            first = selfcorr_path(well, base, self.config["self_correlation"])
            second = selfcorr_path(well, base, self.config["self_correlation"])
            self.assertEqual(first, second)
            predicted, detail = first
            self.assertEqual(detail.fallback_reason, "")
            self.assertEqual(len(predicted), well.hidden_rows)
            self.assertTrue(all(math.isfinite(value) for value in predicted))
            self.assertLessEqual(
                max(abs(value - anchor) for value, anchor in zip(predicted, base)),
                self.config["self_correlation"]["correction_soft_cap_ft"] + 1e-12,
            )

    def test_missing_or_constant_gr_abstains_exactly_to_e006(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                ("aaaaaaaa", {"hidden_gr_missing": True}, "low_hidden_gr_coverage"),
                ("bbbbbbbb", {"constant_gr": True}, "too_few_template_states"),
            )
            for well_id, kwargs, expected in cases:
                path, tvt = self._write_well(root, well_id=well_id, **kwargs)
                well = read_horizontal_selfcorr(path)
                base = [tvt[well.known_rows - 1] + index * 0.01 for index in range(well.hidden_rows)]
                predicted, detail = selfcorr_path(well, base, self.config["self_correlation"])
                self.assertEqual(detail.fallback_reason, expected)
                self.assertEqual(predicted, base)

    def test_short_hidden_suffix_falls_back_when_query_anchor_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, tvt = self._write_well(Path(directory), hidden=1)
            well = read_horizontal_selfcorr(path)
            base = [tvt[well.known_rows - 1]]
            predicted, detail = selfcorr_path(well, base, self.config["self_correlation"])
            self.assertEqual(predicted, base)
            self.assertEqual(detail.fallback_reason, "too_few_query_anchors")

    def test_bounded_path_saturates_and_rejects_bad_contracts(self) -> None:
        base = [0.0, 1.0, 2.0]
        bounded = _bounded_path(base, [1e9, -1e9, 2.0], 40.0)
        self.assertLessEqual(max(abs(value - anchor) for value, anchor in zip(bounded, base)), 40.0)
        with self.assertRaises(DataValidationError):
            _bounded_path([1.0], [1.0, 2.0], 40.0)
        with self.assertRaises(DataValidationError):
            _bounded_path([1.0], [2.0], 0.0)

    def test_visible_pseudo_metric_is_finite_and_uses_only_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._write_well(Path(directory))
            well = read_horizontal_selfcorr(path)
            for fraction in self.config["visible_only_design"]["pseudo_boundary_fractions"]:
                detail = _pseudo_metrics(well, fraction, self.config["self_correlation"])
                self.assertGreater(detail["rows"], 0)
                self.assertTrue(math.isfinite(float(detail["base_rmse"])))
                self.assertTrue(math.isfinite(float(detail["fixed_rmse"])))

    def test_reliability_is_deterministic_bounded_and_positive_gate_abstains(self) -> None:
        diagnostics = {
            "aaaaaaaa": SelfCorrDiagnostic(0.8, 0.8, 60, 10, 0.2, 0.4, 0.3, 3.0, 2.7, "", 5.0),
            "bbbbbbbb": SelfCorrDiagnostic(0.6, 0.5, 55, 8, 0.8, 0.05, -0.2, 3.0, 3.2, "", 8.0),
            "cccccccc": SelfCorrDiagnostic(0.9, 0.9, 70, 12, 0.1, 0.8, 0.5, 3.0, 2.5, "", 4.0),
        }
        reference = _make_reference(diagnostics)
        first = _effective_weight(
            "crossfit_reliability_shrink", 0.5, diagnostics["aaaaaaaa"], reference, "aaaaaaaa", self.config
        )
        second = _effective_weight(
            "crossfit_reliability_shrink", 0.5, diagnostics["aaaaaaaa"], reference, "aaaaaaaa", self.config
        )
        self.assertEqual(first, second)
        self.assertTrue(0.0 <= first <= 0.5)
        gated = _effective_weight(
            "crossfit_positive_gate", 0.5, diagnostics["bbbbbbbb"], reference, "bbbbbbbb", self.config
        )
        self.assertEqual(gated, 0.0)
        with self.assertRaises(DataValidationError):
            _make_reference({})
        with self.assertRaises(DataValidationError):
            _effective_weight(
                "crossfit_conservative", math.inf, diagnostics["aaaaaaaa"], reference, "aaaaaaaa", self.config
            )

    def test_conservative_weight_selection_prefers_smallest_near_best_or_zero(self) -> None:
        improving = {
            f"{index:08x}": BlendSufficient.from_paths(
                f"{index:08x}", [0.0] * 30, [1.0 + 0.01 * index] * 30, [-0.1] * 30
            )
            for index in range(40)
        }
        rmse_weight, conservative, rows = _select_weights(improving, self.config)
        self.assertGreater(rmse_weight, 0.0)
        self.assertGreater(conservative, 0.0)
        passing = [row for row in rows if row["passes_conservative_constraints"]]
        best = min(float(row["rmse"]) for row in passing)
        chosen = next(float(row["rmse"]) for row in rows if row["weight"] == conservative)
        self.assertLessEqual(chosen, best + self.config["placement"]["conservative_rmse_slack"] + 1e-12)
        harmful = {
            "aaaaaaaa": BlendSufficient.from_paths("aaaaaaaa", [0.0] * 20, [0.1] * 20, [5.0] * 20),
            "bbbbbbbb": BlendSufficient.from_paths("bbbbbbbb", [0.0] * 20, [0.2] * 20, [6.0] * 20),
        }
        _, fallback, _ = _select_weights(harmful, self.config)
        self.assertEqual(fallback, 0.0)

    def test_parent_oof_reader_validates_finiteness_and_contiguity(self) -> None:
        header = [
            "id", "well_id", "row_index", "hidden_index", "target", "last_known_tvt",
            "e004_geometry_prefix", "nested_conservative_grid",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.csv.gz"
            with gzip.open(valid, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
                writer.writeheader()
                writer.writerow({"id": "aaaaaaaa_3", "well_id": "aaaaaaaa", "row_index": 3, "hidden_index": 0, "target": 1, "last_known_tvt": 0, "e004_geometry_prefix": 0.5, "nested_conservative_grid": 0.7})
                writer.writerow({"id": "aaaaaaaa_4", "well_id": "aaaaaaaa", "row_index": 4, "hidden_index": 1, "target": 2, "last_known_tvt": 0, "e004_geometry_prefix": 1.5, "nested_conservative_grid": 1.7})
            chunks = list(_parent_wells(valid))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["hidden_index"], [0, 1])
            nonfinite = root / "nonfinite.csv.gz"
            with gzip.open(nonfinite, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
                writer.writeheader()
                writer.writerow({"id": "aaaaaaaa_3", "well_id": "aaaaaaaa", "row_index": 3, "hidden_index": 0, "target": "nan", "last_known_tvt": 0, "e004_geometry_prefix": 0.5, "nested_conservative_grid": 0.7})
            with self.assertRaises(DataValidationError):
                list(_parent_wells(nonfinite))
            gap = root / "gap.csv.gz"
            with gzip.open(gap, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
                writer.writeheader()
                writer.writerow({"id": "aaaaaaaa_3", "well_id": "aaaaaaaa", "row_index": 3, "hidden_index": 1, "target": 1, "last_known_tvt": 0, "e004_geometry_prefix": 0.5, "nested_conservative_grid": 0.7})
            with self.assertRaises(DataValidationError):
                list(_parent_wells(gap))


if __name__ == "__main__":
    unittest.main()
