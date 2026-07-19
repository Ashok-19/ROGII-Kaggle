#!/usr/bin/env python3
"""Run E004 surface-free ablations and deployment packaging."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.deployment import run_e004  # noqa: E402


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/E004/config.json"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E004/results"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/E004"))
    parser.add_argument("--code-sha", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = json.loads(resolve(root, args.config).read_text(encoding="utf-8"))
    code_sha = args.code_sha.strip() or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    started = time.perf_counter()
    summary = run_e004(
        root=root,
        train_dir=resolve(root, args.train_dir),
        test_dir=resolve(root, args.test_dir),
        sample_submission=resolve(root, args.sample_submission),
        output_dir=resolve(root, args.output_dir),
        artifact_dir=resolve(root, args.artifact_dir),
        config=config,
        code_sha=code_sha,
    )
    compact = {
        "status": summary["status"],
        "selected_candidate": summary["selected_candidate"],
        "selected_families": summary["selected_families"],
        "rmse": summary["selected_candidate_metrics"]["rmse"],
        "gain": summary["selected_candidate_metrics"]["gain_vs_baseline"],
        "spatial_rmse": summary["selected_spatial_metrics"]["rmse"],
        "eligible_candidates": summary["eligible_candidates"],
        "local_model_ready": summary["deployment"]["local_model_ready"],
        "deployment_ready": summary["deployment"]["deployment_ready"],
        "controls": {name: detail["pass"] for name, detail in summary["controls"].items()},
        "wall_seconds": time.perf_counter() - started,
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if summary["deployment"]["local_model_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
