#!/usr/bin/env python3
"""Fit the promoted E006 deployment rule, build its notebook, and test local parity."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_validation.e006_inference import write_e006_submission  # noqa: E402
from rogii_validation.fusion import BlendSufficient, _select_inner_weights  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def fit_full_weight(oof_path: Path, config: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    sufficient: dict[str, BlendSufficient] = {}
    current: str | None = None
    truth: list[float] = []
    base: list[float] = []
    particle: list[float] = []

    def flush() -> None:
        if current is not None:
            sufficient[current] = BlendSufficient.from_paths(current, truth, base, particle)

    with gzip.open(oof_path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"well_id", "target", "e004_geometry_prefix", "pf_gr_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{oof_path}: missing parent OOF columns {sorted(missing)}")
        for row in reader:
            well_id = str(row["well_id"])
            if current is None:
                current = well_id
            if well_id != current:
                flush()
                current = well_id
                truth = []
                base = []
                particle = []
            truth.append(float(row["target"]))
            base.append(float(row["e004_geometry_prefix"]))
            particle.append(float(row["pf_gr_path"]))
    flush()
    if len(sufficient) != int(config["expected_wells"]):
        raise ValueError(f"full deployment fit found {len(sufficient)} wells")
    _, conservative, rows = _select_inner_weights(sufficient, config)
    return conservative, rows


def build_model(root: Path, config: dict[str, Any], code_sha: str) -> dict[str, Any]:
    e004_model_path = root / str(config["parents"]["e004_model"])
    e005_config_path = root / str(config["parents"]["e005_config"])
    oof_path = root / str(config["parents"]["e005_oof_path"])
    if sha256(oof_path) != str(config["parents"]["e005_oof_sha256"]):
        raise ValueError("E006 parent OOF hash mismatch")
    e004_model = json.loads(e004_model_path.read_text(encoding="utf-8"))
    e005 = json.loads(e005_config_path.read_text(encoding="utf-8"))
    weight, grid_rows = fit_full_weight(oof_path, config)
    grid_payload = json.dumps(grid_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "experiment_id": "E006",
        "model_name": "nested_conservative_grid",
        "code_sha": code_sha,
        "data_signature": config["data_signature"],
        "training_wells": config["expected_wells"],
        "fusion_weight": weight,
        "fusion_weight_rule": "full-training application of frozen conservative inner-grid rule",
        "full_fit_grid_sha256": hashlib.sha256(grid_payload).hexdigest(),
        "parent_oof_sha256": sha256(oof_path),
        "e004_model_sha256": sha256(e004_model_path),
        "e005_config_sha256": sha256(e005_config_path),
        "e004_model": e004_model,
        "alignment": e005["alignment"],
        "particle_filter": e005["particle_filter"],
        "surfaces_required": False,
        "external_artifacts_required": False,
        "internet_required": False,
    }


def notebook_e006_source(source: str) -> str:
    marker = "from .e004_inference import (  # removed by the notebook builder\n"
    start = source.find(marker)
    if start < 0:
        raise ValueError("could not locate E006 relative import block")
    end = source.find(")\n", start)
    if end < 0:
        raise ValueError("could not terminate E006 relative import block")
    return source[:start] + source[end + 2 :]


def build_notebook(root: Path, model_path: Path, notebook_path: Path) -> dict[str, Any]:
    e004_source_path = root / "src/rogii_validation/e004_inference.py"
    e006_source_path = root / "src/rogii_validation/e006_inference.py"
    e004_source = e004_source_path.read_text(encoding="utf-8")
    e006_source = notebook_e006_source(e006_source_path.read_text(encoding="utf-8"))
    model_text = model_path.read_text(encoding="utf-8")
    execution = """# Locate the official competition input without assuming its Kaggle mount slug.\nimport os\nfrom pathlib import Path\n\ninput_override = os.environ.get('ROGII_INPUT_ROOT', '').strip()\nif input_override:\n    competition_root = Path(input_override)\nelse:\n    candidates = sorted(Path('/kaggle/input').glob('**/sample_submission.csv'))\n    if not candidates:\n        raise FileNotFoundError('Could not locate sample_submission.csv under /kaggle/input')\n    competition_root = candidates[0].parent\n\ntest_dir = competition_root / 'test'\nsample_submission = competition_root / 'sample_submission.csv'\noutput_path = Path(os.environ.get('ROGII_OUTPUT_PATH', '/kaggle/working/submission.csv'))\nresult = write_e006_submission(MODEL, test_dir, sample_submission, output_path)\nprint(json.dumps({k: v for k, v in result.items() if k != 'well_predictions'}, indent=2, sort_keys=True))\n"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "e006-intro",
                "metadata": {},
                "source": [
                    "# E006 nested PF–E004 fusion\n",
                    "Self-contained standard-library inference. Internet and external model datasets are not required.\n",
                    "The notebook locates competition data dynamically and writes `/kaggle/working/submission.csv`.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "e006-e004-runtime",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": e004_source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": "e006-fusion-runtime",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": e006_source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "id": "e006-model",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["MODEL = json.loads(r'''", model_text, "''')\n"],
            },
            {
                "cell_type": "code",
                "id": "e006-launch",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": execution.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "rogii": {
                "experiment_id": "E006",
                "e004_inference_sha256": sha256(e004_source_path),
                "e006_inference_sha256": sha256(e006_source_path),
                "model_sha256": sha256(model_path),
                "internet_required": False,
                "external_artifacts": [],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(notebook_path, notebook)
    return notebook


def execute_notebook_locally(notebook_path: Path, input_root: Path, output_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__main__"}
    old_input = os.environ.get("ROGII_INPUT_ROOT")
    old_output = os.environ.get("ROGII_OUTPUT_PATH")
    os.environ["ROGII_INPUT_ROOT"] = str(input_root)
    os.environ["ROGII_OUTPUT_PATH"] = str(output_path)
    try:
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            exec(compile(text, str(notebook_path), "exec"), namespace, namespace)
    finally:
        if old_input is None:
            os.environ.pop("ROGII_INPUT_ROOT", None)
        else:
            os.environ["ROGII_INPUT_ROOT"] = old_input
        if old_output is None:
            os.environ.pop("ROGII_OUTPUT_PATH", None)
        else:
            os.environ["ROGII_OUTPUT_PATH"] = old_output
    return namespace.get("result", {})


def package_local(
    root: Path,
    config_path: Path,
    output_dir: Path,
    artifact_dir: Path,
    notebook_path: Path,
    code_sha: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_model(root, config, code_sha)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    write_json(model_path, model)
    direct_path = artifact_dir / "submission.csv"
    direct = write_e006_submission(model, root / "data/test", root / "data/sample_submission.csv", direct_path)
    build_notebook(root, model_path, notebook_path)
    notebook_path_output = artifact_dir / "notebook_submission.csv"
    notebook_result = execute_notebook_locally(notebook_path, root / "data", notebook_path_output)
    byte_identical = direct_path.read_bytes() == notebook_path_output.read_bytes()
    return {
        "model_path": str(model_path.relative_to(root)),
        "model_sha256": sha256(model_path),
        "notebook_path": str(notebook_path.relative_to(root)),
        "notebook_sha256": sha256(notebook_path),
        "direct_submission_path": str(direct_path.relative_to(root)),
        "notebook_submission_path": str(notebook_path_output.relative_to(root)),
        "submission_sha256": sha256(direct_path),
        "notebook_submission_sha256": sha256(notebook_path_output),
        "byte_identical": byte_identical,
        "fusion_weight": model["fusion_weight"],
        "direct": {key: value for key, value in direct.items() if key != "well_predictions"},
        "notebook": {key: value for key, value in notebook_result.items() if key != "well_predictions"},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=Path("experiments/E006/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E006/results"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/E006"))
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/e006_deployment.ipynb"))
    parser.add_argument("--code-sha", required=True)
    return parser


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    result = package_local(
        root,
        resolve(root, args.config),
        resolve(root, args.output_dir),
        resolve(root, args.artifact_dir),
        resolve(root, args.notebook),
        args.code_sha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
