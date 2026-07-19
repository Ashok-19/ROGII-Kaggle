#!/usr/bin/env python3
"""Run the frozen E005 GR alignment, particle-filter, and trellis bank."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.gr_path import run_e005  # noqa: E402


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/E005/config.json"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E005/results"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/E005"))
    parser.add_argument("--code-sha", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = json.loads(resolve(root, args.config).read_text(encoding="utf-8"))
    code_sha = args.code_sha.strip() or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    started = time.perf_counter()
    summary = run_e005(
        root=root,
        train_dir=resolve(root, args.train_dir),
        output_dir=resolve(root, args.output_dir),
        artifact_dir=resolve(root, args.artifact_dir),
        config=config,
        code_sha=code_sha,
    )
    print(json.dumps({
        "status": summary["status"],
        "selected_candidate": summary["selected_candidate"],
        "baseline_rmse": summary["baseline_metrics"]["rmse"],
        "e004_rmse": summary["e004_metrics"]["rmse"],
        "selected_rmse": summary["selected_candidate_metrics"]["rmse"],
        "gates": summary["promotion"]["gates"],
        "wall_seconds": time.perf_counter() - started,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
