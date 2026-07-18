#!/usr/bin/env python3
"""Run the E001 metric, fold, and control harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation import run_e001  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/E001/config.json"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--fold-dir", type=Path, default=Path("folds"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E001/results"))
    parser.add_argument("--skip-reference-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_dir = args.train_dir if args.train_dir.is_absolute() else root / args.train_dir
    fold_dir = args.fold_dir if args.fold_dir.is_absolute() else root / args.fold_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    summary = run_e001(
        train_dir=train_dir,
        fold_dir=fold_dir,
        output_dir=output_dir,
        seeds=tuple(config["fold_seeds"]),
        n_folds=int(config["n_folds"]),
        expected_wells=None if args.skip_reference_check else int(config["expected_wells"]),
        expected_baseline_rmse=None if args.skip_reference_check else float(config["expected_baseline_rmse"]),
        baseline_tolerance=float(config["baseline_tolerance"]),
    )
    compact = {
        "status": summary["status"],
        "data_signature": summary["data"]["data_signature"],
        "wells": summary["data"]["well_count"],
        "hidden_rows": summary["data"]["hidden_rows"],
        "baseline_rmse": summary["actual_target_metrics"]["last_known_tvt"]["rmse"],
        "controls": {name: detail["pass"] for name, detail in summary["controls"].items()},
        "output_dir": str(output_dir.relative_to(root)),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
