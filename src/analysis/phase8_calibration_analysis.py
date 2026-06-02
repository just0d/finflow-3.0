"""
phase8_calibration_analysis.py — confidence calibration audit for LSTM vs Transformer.

We test the claim that the LSTM's NLP-gated underperformance is a calibration
phenomenon, not a representation one. Specifically:

  (1) Reliability diagram: bin predictions by confidence; check whether empirical
      accuracy in each bin matches the bin's nominal confidence.
  (2) Expected Calibration Error (ECE) per model.
  (3) Confidence histogram: how peaked is each model?
  (4) Sentiment-overlap: at high confidence, is sentiment_score concentrated
      in zones that *contradict* the prediction?
  (5) Vetoed-trade forensics: of the trades the NLP gate blocked, what would
      they have made? (Positive ⇒ gate killed profitable trades.)

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FINAL = ROOT / "final_results"
CHARTS = FINAL / "charts"
FINAL.mkdir(exist_ok=True); CHARTS.mkdir(exist_ok=True)

EXCLUDE = {"NVDA"}
N_BINS = 10
BAND_PCT = 0.0005   # ±5 bp dead-band defining HOLD label

#  Ground-truth labels from forward returns 
def add_truth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["fwd_ret"] = df.groupby("symbol")["close"].pct_change().shift(-1)
    df["truth"] = np.where(df["fwd_ret"] > BAND_PCT, "BUY",
                  np.where(df["fwd_ret"] < -BAND_PCT, "SELL", "HOLD"))
    df = df[~df["symbol"].isin(EXCLUDE)].dropna(subset=["fwd_ret"])
    return df

# Reliability diagram 
def reliability(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, float, float]:
    df = df.copy()
    df["correct"] = (df["raw_pred"] == df["truth"]).astype(int)
    bins = np.linspace(0.30, 1.0, N_BINS + 1)
    df["bin"] = pd.cut(df["confidence"], bins, include_lowest=True)
    g = df.groupby("bin", observed=True).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        conf=("confidence", "mean"),
    ).reset_index()
    g["model"] = label
    # ECE = sum( n_i/N · |acc_i − conf_i| )
    N = g["n"].sum()
    ece = float((g["n"] / N * (g["acc"] - g["conf"]).abs()).sum())
    overconf = float(((g["conf"] - g["acc"]) * g["n"] / N).sum())  # >0 ⇒ overconfident
    return g, ece, overconf

#  Sentiment-overlap 
def sentiment_overlap(df: pd.DataFrame) -> dict:
    """Among top-decile confidence trades, what fraction face *contrarian* sentiment?"""
    df = df[df["raw_pred"].isin(["BUY", "SELL"])].copy()
    if df.empty: return {}
    q = df["confidence"].quantile(0.90)
    top = df[df["confidence"] >= q]
    contrarian = ((top["raw_pred"] == "BUY") & (top["sentiment_score"] <= -0.10)) | \
                 ((top["raw_pred"] == "SELL") & (top["sentiment_score"] >= 0.05))
    return {
        "n_top_decile": int(len(top)),
        "frac_contrarian": float(contrarian.mean()),
        "frac_in_news":   float((top["news_count"] > 0).mean()),
        "mean_abs_sent":  float(top["sentiment_score"].abs().mean()),
    }

#  Vetoed-trade forensics 
def vetoed_audit(df: pd.DataFrame) -> dict:
    """Trades the NLP gate would block under Protocol-B thresholds."""
    df = df[df["raw_pred"].isin(["BUY", "SELL"])].copy()
    long_blocked  = (df["raw_pred"] == "BUY")  & (df["sentiment_score"] < -0.10)
    short_blocked = (df["raw_pred"] == "SELL") & (df["sentiment_score"] >  0.05)
    blocked = df[long_blocked | short_blocked].copy()
    blocked["signed_pnl"] = np.where(blocked["raw_pred"] == "BUY",
                                      blocked["fwd_ret"], -blocked["fwd_ret"])
    if blocked.empty:
        return {"n_blocked": 0}
    return {
        "n_blocked":      int(len(blocked)),
        "mean_pnl_pct":   float(blocked["signed_pnl"].mean() * 100),
        "median_pnl_pct": float(blocked["signed_pnl"].median() * 100),
        "win_rate_pct":   float((blocked["signed_pnl"] > 0).mean() * 100),
        "total_pnl_pct":  float(blocked["signed_pnl"].sum() * 100),
    }

def main():
    pairs = [
        ("LSTM",        "fused_signals_lstm.csv"),
        ("Transformer", "fused_signals_transformer.csv"),
    ]

    rel_all, summary_rows, audit_rows = [], [], []
    for label, fname in pairs:
        df = pd.read_csv(OUT / fname)
        df = add_truth(df)
        g, ece, oc = reliability(df, label)
        rel_all.append(g)
        s = sentiment_overlap(df)
        a = vetoed_audit(df)
        summary_rows.append({
            "model": label, "ece": round(ece, 4), "overconfidence": round(oc, 4),
            "max_class_acc": round((df["raw_pred"] == df["truth"]).mean(), 4),
            "mean_conf": round(df["confidence"].mean(), 4),
            **{f"top10_{k}": round(v, 4) for k, v in s.items()},
        })
        audit_rows.append({"model": label, **{k: round(v, 4) for k, v in a.items()}})

    rel = pd.concat(rel_all, ignore_index=True)
    rel.to_csv(FINAL / "v3_calibration.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(FINAL / "v3_calibration_summary.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(FINAL / "v3_vetoed_trade_audit.csv", index=False)

    #  Plots 
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"LSTM": "#3b82f6", "Transformer": "#f97316"}

    # (1) Reliability
    ax[0].plot([0.3, 1], [0.3, 1], "--", color="#888", lw=1, label="perfect calibration")
    for label, fname in pairs:
        g = rel[rel["model"] == label]
        ax[0].plot(g["conf"], g["acc"], "o-", color=colors[label], label=label, lw=2, markersize=6)
    ax[0].set_xlabel("predicted confidence")
    ax[0].set_ylabel("empirical accuracy")
    ax[0].set_title("Reliability diagram")
    ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_xlim(0.30, 1.0); ax[0].set_ylim(0.20, 0.60)

    # (2) Confidence histogram
    for label, fname in pairs:
        df = pd.read_csv(OUT / fname)
        df = add_truth(df)
        ax[1].hist(df["confidence"], bins=30, alpha=0.55, color=colors[label],
                   label=label, density=True)
    ax[1].set_xlabel("confidence (max softmax)")
    ax[1].set_ylabel("density")
    ax[1].set_title("Confidence distribution")
    ax[1].grid(alpha=0.3); ax[1].legend()

    # (3) Sentiment vs confidence (top-decile cloud)
    for label, fname in pairs:
        df = pd.read_csv(OUT / fname)
        df = add_truth(df)
        df = df[df["raw_pred"].isin(["BUY", "SELL"])]
        q = df["confidence"].quantile(0.90)
        top = df[df["confidence"] >= q]
        top = top.sample(min(800, len(top)), random_state=0)
        ax[2].scatter(top["sentiment_score"], top["confidence"],
                      s=14, alpha=0.45, color=colors[label], label=label)
    ax[2].axvline(-0.10, color="#888", ls=":", lw=1)
    ax[2].axvline( 0.05, color="#888", ls=":", lw=1)
    ax[2].set_xlabel("FinBERT sentiment_score")
    ax[2].set_ylabel("model confidence")
    ax[2].set_title("Top-decile predictions vs sentiment\n(dotted lines = NLP gate)")
    ax[2].grid(alpha=0.3); ax[2].legend()

    plt.tight_layout()
    plt.savefig(CHARTS / "v3_calibration.png", dpi=130)
    plt.close()

    # Console
    print(" Calibration audit — LSTM vs Transformer")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()
    print("Vetoed-trade forensics (trades NLP gate would block):")
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print()
    print("→ saved final_results/v3_calibration.csv")
    print("→ saved final_results/v3_calibration_summary.csv")
    print("→ saved final_results/v3_vetoed_trade_audit.csv")
    print("→ saved final_results/charts/v3_calibration.png")

if __name__ == "__main__":
    main()
