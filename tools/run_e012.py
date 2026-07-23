#!/usr/bin/env python3
"""Run E012 nested regime-specialist confirmation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.regime_specialists import run_e012  # noqa: E402


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=ROOT)
    command.add_argument("--train-dir", type=Path, default=Path("data/train"))
    command.add_argument("--config", type=Path, default=Path("experiments/E012/config.json"))
    command.add_argument("--e011-config", type=Path, default=Path("experiments/E011/config.json"))
    command.add_argument("--output-dir", type=Path, default=Path("experiments/E012/results"))
    command.add_argument("--code-sha", default="")
    return command


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.root.resolve()
    config = json.loads(resolve(root, args.config).read_text(encoding="utf-8"))
    e011_config = json.loads(resolve(root, args.e011_config).read_text(encoding="utf-8"))
    code_sha = args.code_sha.strip() or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    summary = run_e012(
        root=root,
        train_dir=resolve(root, args.train_dir),
        output_dir=resolve(root, args.output_dir),
        config=config,
        e011_config=e011_config,
        code_sha=code_sha,
    )
    candidates = sorted(
        ((float(metrics["rmse"]), name) for name, metrics in summary["candidate_metrics"].items() if name != "e011"),
    )
    print(json.dumps({
        "status": summary["status"],
        "selected_candidate": summary["selected_candidate"],
        "e011_rmse": summary["e011_metrics"]["rmse"],
        "best_candidate": candidates[0][1],
        "best_candidate_rmse": candidates[0][0],
        "passing_candidates": summary["passing_candidates"],
        "all_controls_pass": summary["all_controls_pass"],
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
