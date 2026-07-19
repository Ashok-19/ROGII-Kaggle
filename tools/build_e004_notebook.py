#!/usr/bin/env python3
"""Build and locally execute the self-contained E004 deployment notebook."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def build_notebook(root: Path, model_path: Path, notebook_path: Path) -> dict[str, Any]:
    inference_path = root / "src/rogii_validation/e004_inference.py"
    if not inference_path.exists():
        inference_path = ROOT / "src/rogii_validation/e004_inference.py"
    source = inference_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    execution = """# Locate the competition input without assuming its Kaggle mount slug.\nimport os\nfrom pathlib import Path\n\ninput_override = os.environ.get('ROGII_INPUT_ROOT', '').strip()\nif input_override:\n    competition_root = Path(input_override)\nelse:\n    candidates = sorted(Path('/kaggle/input').glob('**/sample_submission.csv'))\n    if not candidates:\n        raise FileNotFoundError('Could not locate sample_submission.csv under /kaggle/input')\n    competition_root = candidates[0].parent\n\ntest_dir = competition_root / 'test'\nsample_submission = competition_root / 'sample_submission.csv'\noutput_path = Path(os.environ.get('ROGII_OUTPUT_PATH', '/kaggle/working/submission.csv'))\nresult = write_submission(MODEL, test_dir, sample_submission, output_path)\nprint(json.dumps({k: v for k, v in result.items() if k != 'well_predictions'}, indent=2, sort_keys=True))\n"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "e004-intro",
                "metadata": {},
                "source": [
                    "# E004 surface-free deployment candidate\n",
                    "Pure standard-library inference. Internet and external model datasets are not required.\n",
                    "The notebook locates competition data dynamically and writes `/kaggle/working/submission.csv`.\n",
                ],
            },
            {
                "id": "e004-runtime",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
            {
                "id": "e004-model",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["MODEL = json.loads(r'''", model_text, "''')\n"],
            },
            {
                "id": "e004-launch",
                "cell_type": "code",
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
                "experiment_id": "E004",
                "inference_sha256": sha256(inference_path),
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


def execute_notebook_locally(notebook_path: Path, input_root: Path, output_path: Path) -> None:
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


def rewrite_controls(output_dir: Path, controls: dict[str, Any]) -> None:
    path = output_dir / "control_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["control", "status", "detail_json"], lineterminator="\n")
        writer.writeheader()
        for name, detail in sorted(controls.items()):
            status = "pass" if detail["pass"] else "blocked" if name == "remote_kaggle_mcp_parity" else "fail"
            writer.writerow({
                "control": name,
                "status": status,
                "detail_json": json.dumps({key: value for key, value in detail.items() if key != "pass"}, sort_keys=True, separators=(",", ":")),
            })


def regenerate_manifest(root: Path, output_dir: Path, artifact_dir: Path, notebook_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    runtime = [artifact_dir / "submission.csv", artifact_dir / "notebook_submission.csv"]
    manifest = {
        "schema_version": 1,
        "files": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(files, key=lambda item: item.name)
        ],
        "notebook": {
            "path": str(notebook_path.relative_to(root)),
            "sha256": sha256(notebook_path),
            "bytes": notebook_path.stat().st_size,
        },
        "runtime_artifacts": [
            {"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in runtime
        ],
        "fold_files": [
            {"path": relative, "sha256": sha256(root / relative), "bytes": (root / relative).stat().st_size}
            for relative in config["fold_files"]
        ],
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def finalize_local_parity(root: Path, output_dir: Path, artifact_dir: Path, notebook_path: Path, config_path: Path) -> dict[str, Any]:
    model_path = output_dir / "model.json"
    build_notebook(root, model_path, notebook_path)
    notebook_output = artifact_dir / "notebook_submission.csv"
    execute_notebook_locally(notebook_path, root / "data", notebook_output)
    direct_output = artifact_dir / "submission.csv"
    byte_identical = direct_output.read_bytes() == notebook_output.read_bytes()
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["controls"]["local_notebook_parity"] = {
        "pass": byte_identical,
        "direct_sha256": sha256(direct_output),
        "notebook_sha256": sha256(notebook_output),
        "rows_byte_identical": byte_identical,
    }
    summary["deployment"]["local_notebook_parity"] = byte_identical
    local_controls = [
        detail["pass"]
        for name, detail in summary["controls"].items()
        if name != "remote_kaggle_mcp_parity"
    ]
    summary["deployment"]["local_model_ready"] = all(local_controls)
    summary["deployment"]["deployment_ready"] = bool(
        summary["deployment"]["local_model_ready"]
        and summary["controls"]["remote_kaggle_mcp_parity"]["pass"]
    )
    summary["status"] = (
        "promoted" if summary["deployment"]["deployment_ready"]
        else "blocked" if summary["deployment"]["local_model_ready"]
        else "rejected"
    )
    write_json(summary_path, summary)
    rewrite_controls(output_dir, summary["controls"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = regenerate_manifest(root, output_dir, artifact_dir, notebook_path, config)
    return {
        "byte_identical": byte_identical,
        "direct_sha256": sha256(direct_output),
        "notebook_sha256": sha256(notebook_output),
        "notebook_sha256_file": sha256(notebook_path),
        "manifest_files": len(manifest["files"]),
        "status": summary["status"],
        "local_model_ready": summary["deployment"]["local_model_ready"],
        "deployment_ready": summary["deployment"]["deployment_ready"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E004/results"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/E004"))
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/e004_deployment.ipynb"))
    parser.add_argument("--config", type=Path, default=Path("experiments/E004/config.json"))
    return parser


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    result = finalize_local_parity(
        root,
        resolve(root, args.output_dir),
        resolve(root, args.artifact_dir),
        resolve(root, args.notebook),
        resolve(root, args.config),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["byte_identical"] and result["local_model_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
