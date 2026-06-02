"""
Late Fusion Pipeline: Combines independent predictions from the technical model 
and the FinBERT sentiment model. Applies rule-based logic to veto trades 
when strong negative sentiment contradicts technical buy signals.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# Paths
BASE    = Path(__file__).parent
PREP    = BASE / "data" / "preprocessed"
ALIGNED = BASE / "data" / "aligned"
PROC    = BASE / "data" / "processed"
MODELS  = BASE / "models"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

TARGET_TICKERS = ["NVDA", "TSLA", "JPM", "SPY"]
LABEL_MAP  = {0: "SELL", 1: "HOLD", 2: "BUY"}   # numeric → string
LABEL_INV  = {"SELL": 0, "HOLD": 1, "BUY": 2}    # string → numeric

# Fusion Thresholds
SENTIMENT_BRAKE      = -0.50   # FinBERT score ≤ this → veto BUY → HOLD
SENTIMENT_CONVICTION =  0.50   # FinBERT score ≥ this → confirm BUY (no change, logged)
CONFIDENCE_MIN       =  0.40   # model must be ≥ this confident to enter a trade
                                # (if max softmax prob < 0.40 → HOLD regardless)

# STEP 1 — Load sentiment scores from the test set

def load_sentiment_test() -> pd.DataFrame:
    """
    Load sentiment scores. Priority order:
      1. data/processed/test_sentiment.csv   ← real FinBERT scores (Phase 2 done)
      2. data/aligned/test_aligned.csv       ← no scores yet; will impute neutral
    """
    for path in [PROC / "test_sentiment.csv", ALIGNED / "test_aligned.csv"]:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            source = path.name
            break

    # If sentiment columns are all NaN (Phase 2 not run yet), impute neutral
    if df["sentiment_score"].isna().all():
        print(f"  [WARN] No FinBERT scores in {source}.")
        print(f"         Imputing neutral (0.0) for all hours.")
        print(f"         Run phase2_finbert.py to get real sentiment!")
        df["sentiment_score"]    = 0.0
        df["sentiment_positive"] = 0.0
        df["sentiment_negative"] = 0.0
        df["sentiment_neutral"]  = 1.0
    else:
        n_scored = df["sentiment_score"].notna().sum()
        print(f"  Loaded real FinBERT scores ({n_scored:,} rows scored) from {source}")

    return df[["symbol", "timestamp", "open", "high", "low", "close",
               "volume", "news_count", "sentiment_score",
               "sentiment_positive", "sentiment_negative", "sentiment_neutral",
               "target_direction"]].copy()

# STEP 2 — Generate raw predictions from trained neural nets

def generate_nn_predictions(model_name: str) -> pd.DataFrame | None:
    """
    Load saved model weights → run inference on X_test → return probabilities.
    Returns None if model weights not found (Phase 3 not run yet).
    """
    model_path = MODELS / f"{model_name.lower()}_best.pt"
    if not model_path.exists():
        print(f"  [SKIP] {model_path} not found. Run phase3_train_evaluate.py first.")
        return None

    try:
        import torch
        from phase3_models import LSTMClassifier, TransformerClassifier
        from torch.utils.data import TensorDataset, DataLoader

        X_test = np.load(PREP / "X_test.npy")
        meta   = pd.read_csv(PREP / "meta_test.csv", parse_dates=["timestamp"])
        if meta["timestamp"].dt.tz is None:
            meta["timestamp"] = meta["timestamp"].dt.tz_localize("UTC")

        n_features = X_test.shape[2]
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )

        if model_name.lower() == "lstm":
            model = LSTMClassifier(input_size=n_features)
        else:
            from phase3_models import TransformerClassifier
            model = TransformerClassifier(input_size=n_features)

        state_dict = torch.load(model_path, map_location=device)

        # The Colab notebook saved weights with slightly different attribute names
        # (proj / pe / encoder) vs the local phase3_models.py (input_proj / pos_enc / transformer).
        # Use startswith checks (not str.replace) to avoid collisions like out_proj → out_input_proj.
        if model_name.lower() == "transformer":
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("proj."):          # proj.weight / proj.bias
                    new_k = "input_proj." + k[5:]
                elif k == "pe.pe":                 # positional encoding buffer
                    new_k = "pos_enc.pe"
                elif k.startswith("encoder."):     # encoder.layers.X...
                    new_k = "transformer." + k[8:]
                else:
                    new_k = k                      # norm.*, head.* — unchanged
                remapped[new_k] = v
            state_dict = remapped

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test)),
            batch_size=512, shuffle=False
        )
        all_probs = []
        with torch.no_grad():
            for (Xb,) in loader:
                logits = model(Xb.to(device))
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        probs_arr = np.concatenate(all_probs)          # (N, 3)
        pred_class = probs_arr.argmax(axis=1)           # (N,)
        confidence = probs_arr.max(axis=1)              # (N,)

        result = meta.copy()
        result["raw_pred_num"]    = pred_class
        result["raw_pred"]        = [LABEL_MAP[c] for c in pred_class]
        result["confidence"]      = confidence
        result["prob_sell"]       = probs_arr[:, 0]
        result["prob_hold"]       = probs_arr[:, 1]
        result["prob_buy"]        = probs_arr[:, 2]

        return result

    except ImportError as e:
        print(f"  [WARN] Could not load PyTorch model: {e}")
        return None

def load_prophet_predictions() -> pd.DataFrame | None:
    """
    Prophet predictions were generated inside phase3_train_evaluate.py.
    Check if a saved version exists; otherwise return None.
    """
    prophet_path = OUTPUTS / "prophet_test_preds.csv"
    if prophet_path.exists():
        df = pd.read_csv(prophet_path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df
    return None

# STEP 3 — Apply the Late-Fusion Meta-Rules

def apply_late_fusion(
    predictions: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Merges model predictions with FinBERT sentiment on (symbol, timestamp).
    Applies three override rules:
      1. Low-confidence filter: if confidence < CONFIDENCE_MIN → HOLD
      2. Emergency brake:       if pred == BUY and sentiment ≤ -0.50 → HOLD
      3. Bearish amplification: if pred == BUY and sentiment ≤ -0.30 → HOLD (soft brake)
    """
    # Merge on (symbol, timestamp)
    merged = pd.merge(
        predictions,
        sentiment_df[["symbol", "timestamp", "close", "open", "high", "low",
                       "volume", "news_count", "sentiment_score",
                       "sentiment_positive", "sentiment_negative"]],
        on=["symbol", "timestamp"],
        how="left"
    )
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)

    # Apply fusion rules
    fused_signals    = merged["raw_pred"].copy()
    override_reasons = pd.Series(["none"] * len(merged), index=merged.index)

    for idx, row in merged.iterrows():
        raw       = row["raw_pred"]
        conf      = row.get("confidence", 1.0)
        sentiment = row["sentiment_score"]

        reason = "none"

        # Rule 0 — low confidence filter (neural nets only)
        if "confidence" in merged.columns and conf < CONFIDENCE_MIN:
            fused_signals.iloc[idx] = "HOLD"
            reason = f"low_confidence({conf:.2f})"

        # Rule 1 — hard emergency brake
        elif raw == "BUY" and sentiment <= SENTIMENT_BRAKE:
            fused_signals.iloc[idx] = "HOLD"
            reason = f"emergency_brake(sentiment={sentiment:.2f})"

        # Rule 2 — soft brake (bearish but not extreme)
        elif raw == "BUY" and sentiment <= -0.30:
            fused_signals.iloc[idx] = "HOLD"
            reason = f"soft_brake(sentiment={sentiment:.2f})"

        # Rule 3 — conviction confirmation (no change, just log)
        elif raw == "BUY" and sentiment >= SENTIMENT_CONVICTION:
            reason = f"conviction_boost(sentiment={sentiment:.2f})"

    merged["fused_signal"]    = fused_signals
    merged["override_reason"] = override_reasons
    merged["was_overridden"]  = merged["fused_signal"] != merged["raw_pred"]
    merged["model"]           = model_name

    return merged

# STEP 4 — Save & Report

def print_fusion_report(df: pd.DataFrame, model_name: str):
    print(f"\n  ── {model_name} Fusion Report ──────────────────────")

    total = len(df)
    overrides = df["was_overridden"].sum()
    print(f"  Total signals    : {total:,}")
    print(f"  Overridden       : {overrides:,}  ({overrides/total:.1%})")

    # Breakdown of overrides
    reasons = df[df["was_overridden"]]["override_reason"].value_counts()
    for reason, count in reasons.items():
        print(f"    {reason:40s}: {count:,}")

    # Signal distribution before/after fusion
    raw_counts   = df["raw_pred"].value_counts()
    fused_counts = df["fused_signal"].value_counts()
    print(f"\n  Signal distribution:")
    print(f"  {'Signal':6s}  {'Before Fusion':>15s}  {'After Fusion':>13s}  {'Change':>8s}")
    for sig in ["BUY", "HOLD", "SELL"]:
        before = raw_counts.get(sig, 0)
        after  = fused_counts.get(sig, 0)
        print(f"  {sig:6s}  {before:>12,} ({before/total:.1%})  {after:>10,} ({after/total:.1%})  {after-before:>+8,}")

    # Per-ticker override rate
    print(f"\n  Override rate per ticker:")
    for sym in TARGET_TICKERS:
        sym_df = df[df["symbol"] == sym]
        rate   = sym_df["was_overridden"].mean()
        n_news = (sym_df["news_count"] > 0).mean()
        print(f"    {sym:5s}: {rate:.1%} overridden  |  {n_news:.1%} hours had news")

    # Hours where brake fired on a high-confidence BUY (highest value overrides)
    high_conf_brakes = df[
        df["was_overridden"] &
        df["raw_pred"].eq("BUY") &
        df.get("confidence", pd.Series([1.0]*len(df))).ge(0.60)
    ]
    if len(high_conf_brakes) > 0:
        print(f"\n  High-confidence BUY overrides (conf ≥ 0.60): {len(high_conf_brakes):,}")
        print(f"  (These are the most valuable NLP interventions)")

# MAIN

if __name__ == "__main__":
    print("  FinFlow 1.0 — Phase 4: Late-Fusion Meta-Rule Engine")

    # Load sentiment ground truth
    print("\n[1/4] Loading sentiment scores...")
    sentiment_df = load_sentiment_test()

    # Generate predictions for all models
    print("\n[2/4] Loading model predictions...")
    model_preds = {}

    for mname in ["lstm", "transformer"]:
        print(f"\n  {mname.upper()}:")
        preds = generate_nn_predictions(mname)
        if preds is not None:
            model_preds[mname] = preds
            n = len(preds)
            buy_pct  = (preds["raw_pred"] == "BUY").mean()
            sell_pct = (preds["raw_pred"] == "SELL").mean()
            print(f"  {n:,} sequences | BUY={buy_pct:.1%} SELL={sell_pct:.1%} HOLD={(1-buy_pct-sell_pct):.1%}")

    # Apply late fusion
    print("\n[3/4] Applying Late-Fusion meta-rules...")
    fused_all = {}

    for mname, preds in model_preds.items():
        print(f"\n  {mname.upper()}:")
        fused = apply_late_fusion(preds, sentiment_df, mname.upper())
        fused_all[mname] = fused
        print_fusion_report(fused, mname.upper())

        # Save
        out_path = OUTPUTS / f"fused_signals_{mname}.csv"
        fused.to_csv(out_path, index=False)
        print(f"\n  Saved → {out_path}")

    # Build a buy-and-hold "signals" file for the backtester baseline
    print("\n[4/4] Building Buy-and-Hold baseline signals...")
    bah_rows = []
    for _, row in sentiment_df.iterrows():
        bah_rows.append({
            "symbol":          row["symbol"],
            "timestamp":       row["timestamp"],
            "close":           row["close"],
            "open":            row["open"],
            "high":            row["high"],
            "low":             row["low"],
            "volume":          row["volume"],
            "raw_pred":        "BUY",
            "confidence":      1.0,
            "sentiment_score": row["sentiment_score"],
            "fused_signal":    "BUY",
            "was_overridden":  False,
            "override_reason": "none",
            "model":           "BUY_AND_HOLD",
        })
    bah_df = pd.DataFrame(bah_rows)
    bah_df.to_csv(OUTPUTS / "fused_signals_buyandhold.csv", index=False)

    # Summary
    print("\n" + "="*60)
    print("  PHASE 4 COMPLETE")
    if fused_all:
        print(f"\n  Files saved:")
        for mname in list(fused_all.keys()) + ["buyandhold"]:
            p = OUTPUTS / f"fused_signals_{mname}.csv"
            if p.exists():
                print(f"    {p.name:45s} {p.stat().st_size/1024:.0f} KB")
        print(f"\n  Next: python3 phase5_backtester.py")
    else:
        print("\n  [NOTE] No model predictions were loaded.")
        print("  Run phase3_train_evaluate.py (or Colab) first, then re-run this script.")
        print("  The Buy-and-Hold baseline is ready for Phase 5.")
