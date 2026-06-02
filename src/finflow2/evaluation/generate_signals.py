"""
FinFlow 2.0 — Generate signal CSVs for the new models
Produces files in the SAME format used by phase5_backtester.py:

    outputs/fused_signals_crossmodal.csv
    outputs/fused_signals_sentlstm.csv

Required columns (same as fused_signals_lstm.csv):
    symbol, timestamp, close, raw_pred, confidence, sentiment_score,
    fused_signal, was_overridden, override_reason, model

For deep-fusion models (CrossModalTransformer, SentimentLSTM) the
sentiment is *already* baked into the raw_pred via cross-attention or
the sentiment gate, so we set:

    fused_signal    = raw_pred
    was_overridden  = False
    override_reason = "deep_fusion"

This is the conceptual difference between FinFlow 1.0 (rule-based late
fusion) and FinFlow 2.0 (learned deep fusion): no post-hoc rule fires.

The script also produces RAW (no-deep-fusion) variants so the
backtester can compare deep-fusion vs raw price-only — analogous to
the LSTM_NLP / LSTM_Raw split it already does.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finflow2.evaluation.predict import (
    predict_cross_modal, predict_sentiment_lstm, predict_baseline
)

OUT = ROOT / "outputs"
PROC = ROOT / "data" / "processed"
ALIGNED = ROOT / "data" / "aligned"

def _load_sentiment_test() -> pd.DataFrame:
    for p in [PROC / "test_sentiment.csv", ALIGNED / "test_aligned.csv"]:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
            return df[["symbol", "timestamp", "open", "high", "low", "close",
                        "volume", "news_count", "sentiment_score"]].copy()
    raise FileNotFoundError("No test sentiment file")

def _attach_market_data(preds: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        preds, sent, on=["symbol", "timestamp"], how="left"
    )

def build_signals_deep(model_short: str, predictor) -> pd.DataFrame:
    sent = _load_sentiment_test()
    preds = predictor()
    merged = _attach_market_data(preds, sent)

    merged["fused_signal"]    = merged["raw_pred"]
    merged["was_overridden"]  = False
    merged["override_reason"] = "deep_fusion"
    merged["model"]           = model_short.upper()
    return merged

def main():
    OUT.mkdir(exist_ok=True, parents=True)

    print("\n[1/2] CrossModalTransformer signals…")
    cm = build_signals_deep("CrossModal", predict_cross_modal)
    p = OUT / "fused_signals_crossmodal.csv"
    cm.to_csv(p, index=False)
    print(f"   {len(cm):,} rows  →  {p}")
    print("   Class dist:", cm["fused_signal"].value_counts().to_dict())

    print("\n[2/2] SentimentLSTM signals…")
    sl = build_signals_deep("SentimentLSTM", predict_sentiment_lstm)
    p = OUT / "fused_signals_sentlstm.csv"
    sl.to_csv(p, index=False)
    print(f"   {len(sl):,} rows  →  {p}")
    print("   Class dist:", sl["fused_signal"].value_counts().to_dict())

if __name__ == "__main__":
    main()
