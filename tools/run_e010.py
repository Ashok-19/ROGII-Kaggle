#!/usr/bin/env python3
"""Run the frozen E010 nonlinear candidate-coverage and selector-regret experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.nonlinear_selector import run_e010  # noqa: E402


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--config", type=Path, default=Path("experiments/E010/config.json"))
    command.add_argument("--train-dir", type=Path, default=Path("data/train"))
    command.add_argument("--output-dir", type=Path, default=Path("experiments/E010/results"))
    command.add_argument("--artifact-dir", type=Path, default=Path("artifacts/E010"))
    command.add_argument("--code-sha", default="")
    return command


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    root = arguments.root.resolve()
    config = json.loads(resolve(root, arguments.config).read_text(encoding="utf-8"))
    code_sha = arguments.code_sha.strip() or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    started = time.perf_counter()
    summary = run_e010(
        root=root,
        train_dir=resolve(root, arguments.train_dir),
        output_dir=resolve(root, arguments.output_dir),
        artifact_dir=resolve(root, arguments.artifact_dir),
        config=config,
        code_sha=code_sha,
    )
    reported = summary["reported_candidate"]
    print(json.dumps({
        "status": summary["status"],
        "selected_candidate": summary["selected_candidate"],
        "reported_candidate": reported,
        "last_known_rmse": summary["baseline_metrics"]["rmse"],
        "e006_rmse": summary["e006_metrics"]["rmse"],
        "bank_oracle_rmse": summary["bank_oracle_metrics"]["rmse"],
        "reported_rmse": summary["candidate_metrics"][reported]["rmse"],
        "candidate_count": summary["candidate_count"],
        "advanced_families": summary["advanced_families"],
        "eligible_candidates": summary["eligible_candidates"],
        "wall_seconds": time.perf_counter() - started,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
