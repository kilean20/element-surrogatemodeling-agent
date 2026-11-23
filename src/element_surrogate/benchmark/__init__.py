"""Bounded scientific model studies with explicit coordinates and provenance."""

from .protocol import load_protocol, protocol_identity
from .report import build_tradeoff_report
from .scoring import score_predictions
from .study import create_study, evaluate_test, lock_selection, run_study

__all__ = [
    "build_tradeoff_report",
    "create_study",
    "evaluate_test",
    "load_protocol",
    "lock_selection",
    "protocol_identity",
    "run_study",
    "score_predictions",
]
