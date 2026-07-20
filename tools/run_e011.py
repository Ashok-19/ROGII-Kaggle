#!/usr/bin/env python3
"""Run the frozen E011 memory-safe nonlinear coefficient-learning experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.coefficient_learning import run_e011  # noqa: E402

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--config", type=Path, default=Path("experiments/E011/config.json"))
    command.add_argument("--train-dir", type=Path, default=Path("data/train"))
    command.add_argument("--output-dir", type=Path, default=Path("experiments/E011/results"))
    command.add_argument("--artifact-dir", type=Path, default=Path("artifacts/E011"))
    command.add_argument("--code-sha", default="")
    return command


def main(argv: list[str] | None = None) -> int:
    for name in THREAD_ENV:
        os.environ[name] = "2"
    os.environ["JOBLIB_MULTIPROCESSING"] = "0"
    arguments = parser().parse_args(argv)
    root = arguments.root.resolve()
    config = json.loads(resolve(root, arguments.config).read_text(encoding="utf-8"))
    code_sha = arguments.code_sha.strip() or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    started = time.perf_counter()
    summary = run_e011(
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
        "reported_rmse": summary["candidate_metrics"][reported]["rmse"],
        "advanced_candidates": summary["advanced_candidates"],
        "eligible_candidates": summary["eligible_candidates"],
        "wall_seconds": time.perf_counter() - started,
        "max_rss_kb": summary["runtime"]["max_rss_kb"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
