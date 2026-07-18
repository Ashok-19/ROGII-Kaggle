# Experiment Manifests

The dashboard is populated automatically from `experiments/**/manifest.json`.

## Required manifest shape

```json
{
  "experiment_id": "E001",
  "title": "Robust U baselines",
  "hypothesis": "A robust smooth U target beats last-known TVT.",
  "status": "designed",
  "track": "surface",
  "priority": 1,
  "owner": "NNMax",
  "parent_id": null,
  "created_at": "2026-07-18T00:00:00Z",
  "updated_at": "2026-07-18T00:00:00Z",
  "design": {
    "baseline": "last_known_tvt",
    "legal_inputs": ["MD", "Z", "known TVT_input prefix"],
    "fold_map": "folds/v1.json",
    "controls": ["positive", "noop", "duplicate", "shuffle"],
    "promotion": "Repeated-fold pooled RMSE improves >=0.15 with no p90-well regression >0.25",
    "runtime_budget_minutes": 30
  },
  "runs": []
}
```

## Run entry

```json
{
  "run_id": "R20260718-1200-u-spline",
  "status": "complete",
  "started_at": "2026-07-18T12:00:00Z",
  "ended_at": "2026-07-18T12:20:00Z",
  "git_sha": "...",
  "fold_map": "folds/v1.json",
  "config_path": "experiments/E001/configs/u_spline.json",
  "artifact_dir": "artifacts/E001/R20260718-1200-u-spline",
  "controls": {"positive": "pass", "noop": "pass", "duplicate": "pass", "shuffle": "pass"},
  "metrics": [
    {"split": "cv", "name": "rmse", "value": 10.2},
    {"split": "cv", "name": "p90_well_rmse", "value": 15.1}
  ],
  "well_metrics_path": "artifacts/E001/R20260718-1200-u-spline/well_metrics.csv",
  "notes": ""
}
```

Manifests are the source of truth. Generated SQLite state is disposable and can be rebuilt with `python tools/rogii.py init --reset && python tools/rogii.py sync`.
