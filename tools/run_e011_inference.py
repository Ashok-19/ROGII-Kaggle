#!/usr/bin/env python3
"""Run the frozen E011 deployment model on a competition-layout test directory."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.e011_inference import load_e011_model, write_e011_submission  # noqa: E402

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--model", type=Path, default=Path("experiments/E011/deployment/model.json"))
    command.add_argument("--test-dir", type=Path, default=Path("data/test"))
    command.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    command.add_argument("--output", type=Path, default=Path("artifacts/E011/submission.csv"))
    return command


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    for name in THREAD_ENV:
        os.environ[name] = "2"
    os.environ["JOBLIB_MULTIPROCESSING"] = "0"
    arguments = parser().parse_args(argv)
    root = arguments.root.resolve()
    started = time.perf_counter()
    model = load_e011_model(resolve(root, arguments.model))
    result = write_e011_submission(
        model,
        resolve(root, arguments.test_dir),
        resolve(root, arguments.sample_submission),
        resolve(root, arguments.output),
    )
    payload = {
        **{key: value for key, value in result.items() if key != "well_predictions"},
        "wall_seconds": time.perf_counter() - started,
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "thread_env": {name: os.environ[name] for name in THREAD_ENV},
        "joblib_multiprocessing": os.environ["JOBLIB_MULTIPROCESSING"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
