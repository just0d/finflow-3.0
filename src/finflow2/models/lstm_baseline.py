"""
FinFlow 2.0 — Baseline LSTM (re-export, NO modification)
This file is a thin import wrapper around the FinFlow 1.0 module
`phase3_models.LSTMClassifier`.  It exists ONLY so the new modular
structure (/models/lstm_baseline.py) can co-exist with FinFlow 1.0
without duplicating code.
  Trained weights at models/lstm_best.pt remain valid.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Make the FinFlow 1.0 root importable regardless of cwd
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase3_models import LSTMClassifier  # noqa: E402

__all__ = ["LSTMClassifier"]
