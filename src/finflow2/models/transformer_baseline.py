"""
FinFlow 2.0 — Baseline Transformer (re-export, NO modification)
Thin wrapper around `phase3_models.TransformerClassifier`.
The original Phase-3 file is preserved as-is.
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase3_models import TransformerClassifier, SinusoidalPositionalEncoding  # noqa: E402

__all__ = ["TransformerClassifier", "SinusoidalPositionalEncoding"]
