#!/usr/bin/env python3
"""Run E003: datum, trend, and risk learnability."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation import run_e003  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/E003/config.json"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E003/results"))
    return parser


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = json.loads(resolve(root, args.config).read_text(encoding="utf-8"))
    started = time.perf_counter()
    summary = run_e003(
        root=root,
        train_dir=resolve(root, args.train_dir),
        output_dir=resolve(root, args.output_dir),
        config=config,
    )
    wall_seconds = time.perf_counter() - started
    compact = {
        "status": summary["status"],
        "baseline_rmse": summary["baseline_metrics"]["rmse"],
        "oracle_datum_rmse": summary["oracle_headroom"]["datum_only"]["rmse"],
        "oracle_datum_trend_rmse": summary["oracle_headroom"]["datum_and_trend"]["rmse"],
        "selected_action": summary["selected_action"],
        "selected_action_rmse": summary["selected_action_metrics"]["rmse"],
        "selected_action_gain": summary["selected_action_gain"],
        "signed_action_authorized": summary["decision"]["signed_action_authorized"],
        "selected_risk_model": summary["selected_risk_model"],
        "risk_spearman": summary["selected_risk_metrics"]["spearman"],
        "risk_auc": summary["selected_risk_metrics"]["auc_top_fraction"],
        "risk_detection_authorized": summary["decision"]["risk_detection_authorized"],
        "controls": {name: detail["pass"] for name, detail in summary["controls"].items()},
        "wall_seconds": wall_seconds,
        "output_dir": str(resolve(root, args.output_dir).relative_to(root)),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if all(bool(detail["pass"]) for detail in summary["controls"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
