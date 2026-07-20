from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if np is not None:
    from rogii_validation.coefficient_learning import (
        E011_OOF_FILENAME,
        E011_RESULT_FILENAMES,
        BasisSufficient,
        CandidateSpec,
        CoefficientWell,
        Context,
        FallbackSelection,
        _StatsBuilder,
        _advance_candidates,
        _apply_fallback,
        _build_contexts,
        _candidate_order,
        _context_audit,
        _direct_scoring_control,
        _fallback_selection,
        _fit_prepared_ridge_many,
        _peak_basis_matrix,
        _peak_correction,
        _read_records,
        _prepare_features,
        _safe_relative,
        basis_matrix,
        basis_vector,
        bound_coefficients,
        candidate_specs,
        finalize_e011_outputs,
        run_e011,
        validate_e011_config,
    )
    from rogii_validation.harness import DataValidationError, ErrorAccumulator, scan_profiles


@unittest.skipIf(np is None, "E011 scientific-runtime tests require NumPy")
class E011Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "experiments/E011/config.json").read_text())

    def test_frozen_config_candidate_count_and_capacity_contract(self):
        validate_e011_config(self.config)
        specs = candidate_specs(self.config)
        self.assertEqual(len(specs), 61)
        self.assertEqual(len({item.name for item in specs}), 61)
        self.assertEqual(sum(item.eligible for item in specs), 59)
        self.assertEqual(sum(item.negative_control for item in specs), 2)
        self.assertGreater(self.config["representations"]["linear1"]["oracle_rmse"], 5.0)
        self.assertGreater(self.config["representations"]["quadratic2"]["oracle_rmse"], 5.0)
        self.assertLess(self.config["representations"]["spline4"]["oracle_rmse"], 4.1)
        broken = copy.deepcopy(self.config)
        broken["resource_design"]["maximum_threads"] = 4
        with self.assertRaises(DataValidationError):
            validate_e011_config(broken)
        broken = copy.deepcopy(self.config)
        broken["shrinkages"] = [0.5, 1.0]
        with self.assertRaises(DataValidationError):
            validate_e011_config(broken)
        broken = copy.deepcopy(self.config)
        broken["submission_authorized"] = True
        with self.assertRaises(DataValidationError):
            validate_e011_config(broken)

    def test_basis_one_two_rows_partition_and_invalid_inputs(self):
        for name in self.config["representations"]:
            dimensions = self.config["representations"][name]["dimensions"]
            one = basis_matrix(1, name, self.config)
            self.assertEqual(one.shape, (1, dimensions))
            self.assertTrue(np.all(np.isfinite(one)))
            np.testing.assert_array_equal(one[0], basis_vector(0.0, name, self.config))
            two = basis_matrix(2, name, self.config)
            self.assertEqual(two.shape, (2, dimensions))
            np.testing.assert_array_equal(two[0], basis_vector(0.0, name, self.config))
            np.testing.assert_array_equal(two[1], basis_vector(1.0, name, self.config))
        self.assertTrue(np.allclose(basis_matrix(1, "spline4", self.config), np.zeros((1, 4))))
        self.assertTrue(np.allclose(basis_matrix(2, "spline4", self.config)[1], [0.0, 0.0, 0.0, 1.0]))
        grid = basis_matrix(101, "spline5", self.config)
        self.assertTrue(np.all(grid >= -1e-12))
        self.assertTrue(np.all(grid.sum(axis=1) <= 1.0 + 1e-12))
        self.assertAlmostEqual(float(grid[-1].sum()), 1.0, places=12)
        with self.assertRaises(DataValidationError):
            basis_matrix(0, "spline4", self.config)
        with self.assertRaises(DataValidationError):
            basis_vector(-0.1, "spline4", self.config)
        with self.assertRaises(DataValidationError):
            basis_vector(0.5, "unknown", self.config)

    def _legacy_peak_correction(self, coefficients, name):
        values = np.asarray(coefficients, dtype=np.float64)
        if name.startswith("spline"):
            return float(np.max(np.abs(values))) if values.size else 0.0
        samples = np.linspace(0.0, 1.0, 257)
        matrix = np.vstack([basis_vector(float(value), name, self.config) for value in samples])
        return float(np.max(np.abs(matrix @ values)))

    def _legacy_bound_coefficients(self, coefficients, name):
        values = np.asarray(coefficients, dtype=np.float64)
        bound_key = "spline_knot_absolute_ft" if name.startswith("spline") else "shape3_coefficient_absolute_ft"
        limit = float(self.config["coefficient_bounds"][bound_key])
        values = np.clip(values, -limit, limit)
        peak = self._legacy_peak_correction(values, name)
        maximum = float(self.config["coefficient_bounds"]["emitted_correction_absolute_ft"])
        if peak > maximum and peak > 0.0:
            values = values * (maximum / peak)
        return values

    def test_peak_basis_cache_is_immutable_and_matches_direct_grid(self):
        samples = np.linspace(0.0, 1.0, 257, dtype=np.float64)
        for name in ("shape3", "spline3", "spline4", "spline5", "spline7"):
            first = _peak_basis_matrix(name, self.config)
            second = _peak_basis_matrix(name, self.config)
            self.assertIs(first, second)
            self.assertFalse(first.flags.writeable)
            snapshot = first.copy()
            with self.assertRaises(ValueError):
                first[0, 0] = 1.0
            np.testing.assert_array_equal(first, snapshot)
            knots = self.config["representations"][name]["knots"] or []
            positions = np.unique(np.concatenate((samples, np.asarray(knots, dtype=np.float64))))
            direct = np.vstack([basis_vector(float(value), name, self.config) for value in positions])
            np.testing.assert_allclose(first, direct, rtol=0.0, atol=0.0)

    def test_peak_and_bounds_match_legacy_for_zero_extremes_random_and_outside_vectors(self):
        rng = np.random.default_rng(20260720)
        maximum = float(self.config["coefficient_bounds"]["emitted_correction_absolute_ft"])
        for name in ("shape3", "spline3", "spline4", "spline5", "spline7"):
            dimensions = int(self.config["representations"][name]["dimensions"])
            bound_key = "spline_knot_absolute_ft" if name.startswith("spline") else "shape3_coefficient_absolute_ft"
            limit = float(self.config["coefficient_bounds"][bound_key])
            vectors = (
                np.zeros(dimensions),
                np.full(dimensions, limit),
                np.full(dimensions, -limit),
                rng.uniform(-0.5 * limit, 0.5 * limit, size=dimensions),
                rng.uniform(-3.0 * limit, 3.0 * limit, size=dimensions),
            )
            self.assertTrue(np.all(np.abs(vectors[3]) <= limit))
            self.assertTrue(np.any(np.abs(vectors[4]) > limit))
            for vector in vectors:
                self.assertAlmostEqual(
                    _peak_correction(vector, name, self.config),
                    self._legacy_peak_correction(vector, name),
                    places=12,
                )
                expected = self._legacy_bound_coefficients(vector, name)
                actual = bound_coefficients(vector, name, self.config)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
                np.testing.assert_array_equal(actual, bound_coefficients(vector, name, self.config))
                self.assertLessEqual(_peak_correction(actual, name, self.config), maximum + 1e-8)
            for nonfinite in (math.nan, math.inf, -math.inf):
                vector = np.zeros(dimensions)
                vector[dimensions // 2] = nonfinite
                with self.assertRaises(DataValidationError):
                    bound_coefficients(vector, name, self.config)
        with self.assertRaises(DataValidationError):
            bound_coefficients([1, 2], "spline4", self.config)

    def test_sufficient_statistics_match_direct_rows_singular_and_zero(self):
        rng = np.random.default_rng(123)
        for rows in (1, 2, 17, 100):
            for name in ("shape3", "spline3", "spline4", "spline5", "spline7"):
                dimensions = self.config["representations"][name]["dimensions"]
                builder = _StatsBuilder(name, dimensions)
                direct = []
                coefficients = bound_coefficients(rng.normal(0, 3, dimensions), name, self.config)
                for index in range(rows):
                    position = index / max(1, rows - 1)
                    basis = basis_vector(position, name, self.config)
                    base_error = float(rng.normal())
                    builder.add(float(index), base_error, basis)
                    direct.append(base_error + float(basis @ coefficients))
                sufficient = builder.finalize()
                metric = sufficient.metric("well", coefficients)
                accumulator = ErrorAccumulator()
                for index, error in enumerate(direct):
                    accumulator.add(error, float(index))
                expected = accumulator.finalize("well")
                self.assertAlmostEqual(metric.sse, expected.sse, places=9)
                self.assertAlmostEqual(metric.mean_error, expected.mean_error, places=9)
                self.assertAlmostEqual(metric.trend_per_row, expected.trend_per_row, places=9)
                zero = sufficient.metric("well", np.zeros(dimensions))
                self.assertAlmostEqual(zero.sse, builder.base_sse, places=9)
                self.assertTrue(np.all(np.isfinite(sufficient.target_coefficients)))

    @staticmethod
    def _metric(well_id: str, errors):
        acc = ErrorAccumulator()
        for index, error in enumerate(errors):
            acc.add(float(error), float(index))
        return acc.finalize(well_id)

    def _small_record(self, well_id: str, feature: float, target=(1.0, 2.0, 3.0, 4.0), rows=8):
        builder = _StatsBuilder("spline4", 4)
        for index in range(rows):
            basis = basis_vector(index / max(1, rows - 1), "spline4", self.config)
            correction = float(basis @ np.asarray(target))
            builder.add(float(index), -correction, basis)
        sufficient = builder.finalize()
        zero = sufficient.metric(well_id, np.zeros(4))
        return CoefficientWell(
            well_id=well_id,
            rows=rows,
            features={"backtest_0p5_signal": feature, "backtest_0p7_signal": feature * 2, "backtest_0p85_signal": feature * 3, "constant": 1.0, "missing": None},
            spatial=(feature, 0.0),
            typewell=(feature, 1.0, 10.0),
            hidden_gr_missing_fraction=0.0,
            last_metric=zero,
            e006_metric=zero,
            statistics={"spline4": sufficient},
        )

    def test_feature_preparation_is_training_only_handles_missing_constant(self):
        records = {f"w{i}": self._small_record(f"w{i}", float(i), target=(i, i / 2, -i / 3, i / 4)) for i in range(12)}
        names = tuple(records["w0"].features)
        train = tuple(f"w{i}" for i in range(8)); test = tuple(f"w{i}" for i in range(8, 12))
        y = np.vstack([records[well].statistics["spline4"].target_coefficients for well in train])
        first = _prepare_features(records, train, test, names, y, 3)
        second = _prepare_features(records, train, test, names, y, 3)
        self.assertEqual(first.names, second.names)
        self.assertTrue(np.array_equal(first.train_x, second.train_x))
        self.assertTrue(np.all(np.isfinite(first.test_x)))
        changed = copy.deepcopy(records)
        for well in test:
            changed[well].features["backtest_0p5_signal"] = 1e9
        third = _prepare_features(changed, train, test, names, y, 3)
        self.assertEqual(first.names, third.names)
        with self.assertRaises(DataValidationError):
            _prepare_features(records, (), test, names, y[:0], 3)

    def test_batched_ridge_matches_sklearn_weighted_and_unweighted(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(20260720)
        for weighted in (False, True):
            train_x = rng.normal(size=(91, 27))
            test_x = rng.normal(size=(13, 27))
            train_y = rng.normal(size=(91, 5))
            weights = rng.uniform(0.2, 3.0, size=91) if weighted else None
            prepared = type("Prepared", (), {"train_x": train_x, "test_x": test_x})()
            alphas = (1.0, 10.0, 100.0, 1000.0)
            actual = _fit_prepared_ridge_many(prepared, train_y, alphas, weights)
            y_mean = train_y.mean(axis=0)
            y_scale = train_y.std(axis=0)
            y_scale[y_scale < 1e-9] = 1.0
            scaled = (train_y - y_mean) / y_scale
            for alpha in alphas:
                expected = Ridge(alpha=alpha, fit_intercept=True).fit(train_x, scaled, sample_weight=weights).predict(test_x) * y_scale + y_mean
                self.assertLess(float(np.max(np.abs(actual[alpha] - expected))), 1e-10)

    @unittest.skipUnless((ROOT / "artifacts/E011/compact_stats_v1.npz").is_file(), "registered compact E011 artifact is not mounted")
    def test_registered_compact_loader_reproduces_frozen_metrics(self):
        records, names, audit = _read_records(ROOT, ROOT / "data/train", self.config)
        self.assertEqual(audit["input_mode"], "compact_sufficient_statistics")
        self.assertEqual(len(records), 773)
        self.assertEqual(len(names), 142)
        self.assertEqual(sum(record.rows for record in records.values()), 3783989)
        e006_sse = sum(record.e006_metric.sse for record in records.values())
        last_sse = sum(record.last_metric.sse for record in records.values())
        self.assertAlmostEqual(math.sqrt(e006_sse / 3783989), 14.933140787237672, places=10)
        self.assertAlmostEqual(math.sqrt(last_sse / 3783989), 15.90985287073455, places=10)
        for representation in self.config["learned_representations"]:
            sse = sum(record.statistics[representation].metric(well_id, record.statistics[representation].target_coefficients).sse for well_id, record in records.items())
            self.assertAlmostEqual(math.sqrt(sse / 3783989), self.config["representations"][representation]["oracle_rmse"], places=10)
        basis_delta, sufficient_delta = _direct_scoring_control(ROOT, ROOT / self.config["parent_artifacts"]["e010_oof"]["path"], records, self.config)
        self.assertEqual(basis_delta, 0.0)
        self.assertLess(sufficient_delta, 1e-10)

    def test_context_audit_and_group_building(self):
        records = {f"w{i}": self._small_record(f"w{i}", float(i)) for i in range(10)}
        folds = []
        for version in range(5):
            folds.append({"version": f"v{version+1}", "n_folds": 5, "assignments": {well: (index + version) % 5 for index, well in enumerate(sorted(records))}})
        contexts, audits, spatial, typewell = _build_contexts(records, folds, self.config)
        self.assertEqual(len(contexts), 35)
        self.assertEqual(len(audits), 35)
        self.assertTrue(all(row["pass"] for row in audits))
        self.assertEqual(set(spatial), set(records))
        self.assertEqual(set(typewell), set(records))
        bad = Context("bad", "repeated", "v1", 0, ("a",), ("a",), {"a": 0})
        self.assertFalse(_context_audit(bad)["pass"])

    def test_fallback_all_none_and_tie_determinism(self):
        records = {f"w{i}": self._small_record(f"w{i}", float(i), target=(i + 1, 0, 0, 0)) for i in range(10)}
        ridge = {well: np.zeros(4) for well in records}
        tree = {well: np.zeros(4) for well in records}
        selection = _fallback_selection(records, "spline4", ridge, tree, self.config)
        self.assertFalse(selection.passed)
        predictions, details = _apply_fallback(selection, "spline4", ridge, tree, self.config)
        self.assertTrue(all(np.array_equal(value, np.zeros(4)) for value in predictions.values()))
        self.assertTrue(all(item["fallback"] for item in details.values()))
        manual = FallbackSelection(1.0, 1.0, 1.0, math.inf, math.inf, 0.0, 1.0, True)
        tree = {well: records[well].statistics["spline4"].target_coefficients * 2 for well in records}
        predictions, details = _apply_fallback(manual, "spline4", ridge, tree, self.config)
        self.assertTrue(all(item["action"] for item in details.values()))
        self.assertTrue(all(np.any(np.abs(value) > 0) for value in predictions.values()))

    def test_advancement_preserves_diversity_and_frozen_order(self):
        specs = [CandidateSpec(f"r{i}_m{j}", f"r{i}", f"m{j}", 1.0) for i in range(5) for j in range(4)]
        summaries = {spec.name: {"rmse": 10.0 + index * 0.01} for index, spec in enumerate(specs)}
        gates = {spec.name: {"a": True, "b": True} for spec in specs}
        order = {spec.name: index for index, spec in enumerate(specs)}
        # The helper's frozen representations are E011 names, so generic names fill by order.
        selected = _advance_candidates(specs, summaries, gates, order, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, [spec.name for spec in specs[:8]])
        tied = {spec.name: {"rmse": 10.0} for spec in specs}
        tied_first = _advance_candidates(specs, tied, gates, order, 8)
        tied_second = _advance_candidates(specs, tied, gates, order, 8)
        self.assertEqual(tied_first, tied_second)
        self.assertEqual(tied_first, [spec.name for spec in specs[:8]])
        gates[specs[0].name]["a"] = False
        self.assertNotIn(specs[0].name, _advance_candidates(specs, summaries, gates, order, 8))

    def test_safe_relative_and_finalize_rejects_partial_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = root / "results"; artifact = root / "artifacts"
            output.mkdir(); artifact.mkdir()
            with self.assertRaises(DataValidationError):
                _safe_relative(root / "outside", artifact)
            self.assertEqual(_safe_relative(artifact / "x.csv", artifact), "x.csv")
            (output / "summary.json").write_text(json.dumps({"experiment_id": "E011", "code_sha": "abc"}))
            with self.assertRaises(DataValidationError):
                finalize_e011_outputs(root=root, output_dir=output, artifact_dir=artifact, config={"expected_hidden_rows": 1}, code_sha="abc")

    @staticmethod
    def _gzip(path: Path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                    writer.writeheader(); writer.writerows(rows)

    def _fixture(self, root: Path, *, wells=30, hidden=12):
        train_dir = root / "data/train"; train_dir.mkdir(parents=True)
        (root / "folds").mkdir()
        (root / "artifacts/E006").mkdir(parents=True)
        (root / "artifacts/E010").mkdir(parents=True)
        (root / "experiments/E008/results").mkdir(parents=True)
        (root / "experiments/E010/results").mkdir(parents=True)
        well_ids=[]; oof_rows=[]; feature_rows=[]; e006_rows=[]
        for index in range(wells):
            well_id=f"{index+1:08x}"; well_ids.append(well_id)
            known=20; total=known+hidden
            signal=(index-(wells-1)/2)/max(1,(wells-1)/2)
            controls=np.asarray([18*signal, -10*signal + 2*math.sin(index), 8*math.cos(index/3), 14*signal*signal-5])
            horizontal=[]
            for row_index in range(total):
                md=1000+row_index
                x=index*100+row_index
                y=(index%5)*1000+row_index*.1
                z=2000+row_index*.05
                last_visible=1000+index*.4+0.08*min(row_index,known-1)
                if row_index<known:
                    tvt=last_visible; tvt_input=f"{tvt:.8f}"
                else:
                    h=row_index-known
                    e006=1000+index*.4+0.08*(known-1)+0.10*(h+1)
                    basis=basis_vector(h/max(1,hidden-1),"spline4",self.config)
                    tvt=e006+float(basis@controls)
                    tvt_input=""
                gr=55+8*math.sin(tvt/7)+2*math.cos(tvt/13)
                gr_value="" if index%2 and row_index>=known and (row_index-known)%3==0 else f"{gr:.8f}"
                horizontal.append([md,x,y,z,tvt,gr_value,tvt_input])
            with (train_dir/f"{well_id}__horizontal_well.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["MD","X","Y","Z","TVT","GR","TVT_input"]); w.writerows(horizontal)
            with (train_dir/f"{well_id}__typewell.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["TVT","GR"])
                for j in range(200):
                    tvt=900+j*.5; w.writerow([tvt,55+8*math.sin(tvt/7)])
            for h in range(hidden):
                row_index=known+h
                target=float(horizontal[row_index][4])
                e006=1000+index*.4+0.08*(known-1)+0.10*(h+1)
                last=1000+index*.4+0.08*(known-1)
                common={"id":f"{well_id}_{row_index}","well_id":well_id,"row_index":row_index,"hidden_index":h,"target":target,"last_known_tvt":last,"e006_nested_fusion":e006}
                oof_rows.append({**common,"bank_oracle":target})
                e006_rows.append({**common,"nested_conservative_grid":e006})
            feature_rows.append({
                "well_id":well_id,
                "backtest_0p5_signal":signal,
                "backtest_0p7_signal":signal*signal,
                "backtest_0p85_signal":math.sin(index),
                "hidden_rows":hidden,
                "known_rows":known,
                "hidden_fraction":hidden/total,
                "hidden_gr_missing_fraction":1/3 if index%2 else 0.0,
                "signal_0":controls[0],
                "signal_1":controls[1],
                "signal_2":controls[2],
                "signal_3":controls[3],
            })
        _, profile=scan_profiles(train_dir)
        for version in range(5):
            fold={"schema_version":1,"version":f"v{version+1}","seed":1000+version,"strategy":"fixture","data_signature":profile["data_signature"],"fingerprint":hashlib.sha256(f"e011-{version}".encode()).hexdigest(),"n_folds":5,"well_count":wells,"assignments":{well:(i+version)%5 for i,well in enumerate(well_ids)},"fold_summary":[]}
            path=root/f"folds/v{version+1}.json"; path.write_text(json.dumps(fold))
        self._gzip(root/"artifacts/E010/oof_predictions.csv.gz",list(oof_rows[0]),oof_rows)
        self._gzip(root/"artifacts/E006/oof_predictions.csv.gz",list(e006_rows[0]),e006_rows)
        with (root/"experiments/E008/results/legal_features.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(feature_rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(feature_rows)
        # E011 verifies but does not consume these two E010 diagnostics.
        for name in ("oracle_targets.csv","selector_predictions.csv"):
            path=root/f"experiments/E010/results/{name}"
            path.write_text("well_id,value\n00000001,0\n")
        config=copy.deepcopy(self.config)
        config.pop("compact_input", None)
        config["expected_wells"]=wells; config["expected_hidden_rows"]=wells*hidden; config["data_signature"]=profile["data_signature"]
        config["legal_feature_contract"]["expected_rows"]=wells; config["legal_feature_contract"]["expected_columns"]=len(feature_rows[0])
        fixture_paths={
            "e006_oof":root/"artifacts/E006/oof_predictions.csv.gz",
            "e008_legal_features":root/"experiments/E008/results/legal_features.csv",
            "e010_oof":root/"artifacts/E010/oof_predictions.csv.gz",
            "e010_oracle_targets":root/"experiments/E010/results/oracle_targets.csv",
            "e010_selector_predictions":root/"experiments/E010/results/selector_predictions.csv",
        }
        for key,path in fixture_paths.items():
            config["parent_artifacts"][key]={"path":path.relative_to(root).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
        config["fold_files"]=[]
        for version in range(5):
            path=root/f"folds/v{version+1}.json"
            config["fold_files"].append({"path":path.relative_to(root).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size})
        for name in ("ridge_equal","ridge_row_weighted"):
            config["model_families"][name]["feature_counts"]=[4,8]
            config["model_families"][name]["alphas"]=[1.0,10.0]
        config["model_families"]["extra_trees"].update({"n_estimators":8,"max_depth":4,"min_samples_leaf":2})
        config["promotion"].update({"minimum_gain_vs_e006":0.01,"minimum_gain_vs_last_known":0.01,"minimum_representation_oracle_gain_retention":0.0,"minimum_map_wins":1,"minimum_outer_cell_wins":1,"maximum_p90_deterioration_vs_e006":100.0,"require_positive_every_spatial_group":False,"require_positive_every_typewell_group":False,"require_positive_long_suffix":False,"require_positive_high_gr_missingness":False,"require_positive_e006_catastrophic_slice":False,"maximum_rss_mb":4096})
        config["evaluation"]["preliminary_advancement"].update({"minimum_gain_vs_e006":0.0,"minimum_map_wins":0,"minimum_outer_cell_wins":0,"minimum_representation_oracle_gain_retention":-1.0})
        config["controls"].update({"maximum_rss_mb":4096,"maximum_runtime_minutes":30,"shuffled_target_maximum_gain_vs_e006":100.0,"sign_flipped_target_maximum_gain_vs_e006":100.0})
        fixture_config = root / "experiments/E011/config.json"
        fixture_config.parent.mkdir(parents=True, exist_ok=True)
        fixture_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        return train_dir,config

    def test_end_to_end_fixture_all_branches_outputs_and_determinism(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); train,config=self._fixture(root)
            # validate_e011_config requires the frozen 2 GB promotion cap; fixture only relaxes after validation.
            config["promotion"]["maximum_rss_mb"]=2048
            first=run_e011(root=root,train_dir=train,output_dir=root/"out1",artifact_dir=root/"art1",config=config,code_sha="fixture")
            second=run_e011(root=root,train_dir=train,output_dir=root/"out2",artifact_dir=root/"art2",config=config,code_sha="fixture")
            self.assertEqual(first["candidate_count"],61)
            self.assertEqual(set(first["candidate_metrics"]),set(second["candidate_metrics"]))
            self.assertEqual(first["reported_candidate"],second["reported_candidate"])
            self.assertEqual(hashlib.sha256((root/"art1"/E011_OOF_FILENAME).read_bytes()).hexdigest(),hashlib.sha256((root/"art2"/E011_OOF_FILENAME).read_bytes()).hexdigest())
            for name in E011_RESULT_FILENAMES:
                self.assertTrue((root/"out1"/name).is_file(),name)
            self.assertTrue((root/"out1/artifact_manifest.json").is_file())
            self.assertEqual(sum(1 for _ in gzip.open(root/"art1"/E011_OOF_FILENAME,"rt"))-1,config["expected_hidden_rows"])
            self.assertTrue(first["controls"]["basis_reconstruction"]["pass"])
            self.assertTrue(first["controls"]["sufficient_statistic_direct"]["pass"])
            self.assertTrue(first["controls"]["duplicate_model"]["pass"])
            self.assertLess(first["runtime"]["max_rss_kb"],2048*1024)

    def test_parent_hash_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); train,config=self._fixture(root,wells=10,hidden=4)
            config["promotion"]["maximum_rss_mb"]=2048
            config["parent_artifacts"]["e010_oof"]["sha256"]="0"*64
            with self.assertRaises(DataValidationError):
                run_e011(root=root,train_dir=train,output_dir=root/"out",artifact_dir=root/"art",config=config,code_sha="fixture")


if __name__ == "__main__":
    unittest.main()
