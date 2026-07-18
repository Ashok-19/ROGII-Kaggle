#!/usr/bin/env python3
"""Run E002: the structural U baseline ladder."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation import run_e002  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/E002/config.json"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--fold-dir", type=Path, default=Path("folds"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E002/results"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/E002/structural-ladder"))
    return parser


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config_path = resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    summary = run_e002(
        root=root,
        train_dir=resolve(root, args.train_dir),
        fold_dir=resolve(root, args.fold_dir),
        output_dir=resolve(root, args.output_dir),
        artifact_dir=resolve(root, args.artifact_dir),
        config=config,
    )
    wall_seconds = time.perf_counter() - started
    compact = {
        "status": summary["status"],
        "selected_candidate": summary["selected_candidate"],
        "selected_rmse": summary["candidate_metrics"][summary["selected_candidate"]]["rmse"],
        "best_challenger": summary["best_challenger"],
        "challenger_rmse": summary["candidate_metrics"][summary["best_challenger"]]["rmse"],
        "baseline_rmse": summary["candidate_metrics"]["last_known_tvt"]["rmse"],
        "pooled_gain": summary["selection"]["pooled_rmse_gain"],
        "improved_fold_cells": summary["improved_fold_cells"][summary["best_challenger"]],
        "sign_ratio": summary["sign_verification"]["plus_to_minus_ratio"],
        "controls": {name: detail["pass"] for name, detail in summary["controls"].items()},
        "wall_seconds": wall_seconds,
        "output_dir": str(resolve(root, args.output_dir).relative_to(root)),
        "artifact_dir": str(resolve(root, args.artifact_dir).relative_to(root)),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    controls_pass = all(bool(detail["pass"]) for detail in summary["controls"].values())
    return 0 if controls_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
