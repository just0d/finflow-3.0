"""
FinFlow 2.0 — Inference helpers for the new models
Loads a trained checkpoint, runs it on the *test* split, and returns a
DataFrame with columns:

    symbol, timestamp,
    raw_pred,            # SELL / HOLD / BUY  (string)
    raw_pred_num,        # 0 / 1 / 2
    confidence,          # max softmax probability
    prob_sell, prob_hold, prob_buy

This is the SAME contract as `phase4_late_fusion.generate_nn_predictions`.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finflow2.training.data_loader import _build_aligned_tensors
from finflow2.models.cross_modal_transformer import CrossModalTransformer
from finflow2.models.sentiment_lstm          import SentimentLSTMModel

CKPT = ROOT / "finflow2" / "checkpoints"
LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _to_pred_df(meta: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    pred_class = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    out = meta.copy()
    out["raw_pred_num"] = pred_class
    out["raw_pred"]     = [LABEL_MAP[c] for c in pred_class]
    out["confidence"]   = confidence
    out["prob_sell"]    = probs[:, 0]
    out["prob_hold"]    = probs[:, 1]
    out["prob_buy"]     = probs[:, 2]
    return out

# Cross-Modal

def predict_cross_modal(split: str = "test") -> pd.DataFrame:
    ckpt_path = CKPT / "cross_modal_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — train first.")

    blob   = torch.load(ckpt_path, map_location="cpu")
    cfg    = blob.get("config", {})
    model  = CrossModalTransformer(**cfg)
    model.load_state_dict(blob["state_dict"]); model.eval()
    dev = _device(); model.to(dev)

    data = _build_aligned_tensors(split)
    p = torch.from_numpy(data["X_price"]).to(dev)
    t = torch.from_numpy(data["X_text"] ).to(dev)

    probs_all = []
    BS = 512
    with torch.no_grad():
        for i in range(0, len(p), BS):
            logits = model(p[i:i+BS], t[i:i+BS])
            probs_all.append(torch.softmax(logits, -1).cpu().numpy())
    probs = np.concatenate(probs_all)
    return _to_pred_df(data["meta"], probs)

# Sentiment LSTM

def predict_sentiment_lstm(split: str = "test") -> pd.DataFrame:
    ckpt_path = CKPT / "sentiment_lstm_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — train first.")

    blob   = torch.load(ckpt_path, map_location="cpu")
    cfg    = blob.get("config", {})
    model  = SentimentLSTMModel(**cfg)
    model.load_state_dict(blob["state_dict"]); model.eval()
    dev = _device(); model.to(dev)

    data = _build_aligned_tensors(split)
    p = torch.from_numpy(data["X_price"]).to(dev)
    s = torch.from_numpy(data["X_sent"] ).to(dev)
    n = torch.from_numpy(data["X_news"] ).to(dev)

    probs_all = []
    BS = 512
    with torch.no_grad():
        for i in range(0, len(p), BS):
            logits = model(p[i:i+BS], s[i:i+BS], n[i:i+BS])
            probs_all.append(torch.softmax(logits, -1).cpu().numpy())
    probs = np.concatenate(probs_all)
    return _to_pred_df(data["meta"], probs)

# Generic baseline predictor (LSTM / Transformer baselines)

def predict_baseline(model_name: str, split: str = "test") -> pd.DataFrame:
    """
    Reuses the FinFlow 1.0 trained weights at models/{lstm,transformer}_best.pt
    so we can score them with the *same* evaluation framework.
    """
    from finflow2.models.lstm_baseline        import LSTMClassifier
    from finflow2.models.transformer_baseline import TransformerClassifier

    weights = ROOT / "models" / f"{model_name.lower()}_best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found")

    data = _build_aligned_tensors(split)
    n_feat = data["X_price"].shape[2]

    if model_name.lower() == "lstm":
        model = LSTMClassifier(input_size=n_feat)
    else:
        model = TransformerClassifier(input_size=n_feat)

    state = torch.load(weights, map_location="cpu")
    if model_name.lower() == "transformer":
        # Phase-3 Colab key remap (see phase4_late_fusion.py)
        remapped = {}
        for k, v in state.items():
            if k.startswith("proj."):       remapped["input_proj." + k[5:]] = v
            elif k == "pe.pe":              remapped["pos_enc.pe"]          = v
            elif k.startswith("encoder."):  remapped["transformer." + k[8:]] = v
            else:                           remapped[k] = v
        state = remapped

    model.load_state_dict(state); model.eval()
    dev = _device(); model.to(dev)

    p = torch.from_numpy(data["X_price"]).to(dev)
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(p), 512):
            logits = model(p[i:i+512])
            probs_all.append(torch.softmax(logits, -1).cpu().numpy())
    probs = np.concatenate(probs_all)
    return _to_pred_df(data["meta"], probs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["cross_modal", "sentiment_lstm",
                                            "lstm_baseline", "transformer_baseline"])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    if args.model == "cross_modal":
        df = predict_cross_modal(args.split)
    elif args.model == "sentiment_lstm":
        df = predict_sentiment_lstm(args.split)
    elif args.model == "lstm_baseline":
        df = predict_baseline("lstm", args.split)
    else:
        df = predict_baseline("transformer", args.split)

    print(df.head())
    print(f"\nN={len(df):,}")
    print("Class dist:\n", df["raw_pred"].value_counts())
