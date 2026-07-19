"""Deterministic validation utilities for the ROGII competition."""

from .harness import (
    DEFAULT_FOLD_SEEDS,
    DataValidationError,
    generate_fold_maps,
    run_e001,
    scan_profiles,
)
from .structural import huber_line, run_e002
from .learnability import run_e003
from .deployment import run_e004
from .gr_path import run_e005
from .fusion import run_e006
from .self_correlation import run_e007
from .residual_action import run_e008
from .consensus_abstention import run_e009

try:
    from .nonlinear_selector import run_e010
except ModuleNotFoundError as exc:  # Optional scientific runtime for E010 only.
    _E010_IMPORT_ERROR = exc

    def run_e010(*args, **kwargs):
        raise RuntimeError("E010 requires NumPy and scikit-learn in the active Python environment") from _E010_IMPORT_ERROR

__all__ = [
    "DEFAULT_FOLD_SEEDS",
    "DataValidationError",
    "generate_fold_maps",
    "run_e001",
    "huber_line",
    "run_e002",
    "run_e003",
    "run_e004",
    "run_e005",
    "run_e006",
    "run_e007",
    "run_e008",
    "run_e009",
    "run_e010",
    "scan_profiles",
]
