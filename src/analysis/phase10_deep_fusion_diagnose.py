"""
phase10_deep_fusion_diagnose.py — diagnose & repair the deep-fusion models.

Two-part script:

  PART A — Calibration audit (mirror of phase8 for the deep models)
        confidence histograms, ECE, overconfidence,
        signal-mix (BUY/SELL/HOLD ratios), top-decile sentiment overlap.

  PART B — Repair sweep (mirror of phase9 for the deep models)
        Try fixes that do NOT require retraining:
          R1  default              top-40% conf floor, strict gate (current)
          R2  raw + default cf     no gate, top-40% conf
          R3  high-conf only       top-20% conf floor, strict gate
          R4  top-decile           top-10% conf floor, strict gate
          R5  extreme gate (V8)    top-40% conf, |s|>0.30 AND high conf
          R6  ensemble w/ Tx       agree-with-Transformer veto
          R7  temperature-scaled   T=2 on softmax → recompute conf, default gate

"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase6_backtester_v2 as bt   # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FINAL = ROOT / "final_results"
CHARTS = FINAL / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

DEEP_MODELS = {
    "CrossModalTransformer": "fused_signals_crossmodal.csv",
    "SentimentLSTM":         "fused_signals_sentlstm.csv",
}
TX_FILE = "fused_signals_transformer.csv"
EXCLUDE = {"NVDA"}
BAND = 0.0005

# PART A — CALIBRATION AUDIT 
def add_truth(df):
    df = df.sort_values(["symbol","timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["fwd_ret"] = df.groupby("symbol")["close"].pct_change().shift(-1)
    df["truth"] = np.where(df["fwd_ret"] >  BAND, "BUY",
                  np.where(df["fwd_ret"] < -BAND, "SELL", "HOLD"))
    df = df[~df["symbol"].isin(EXCLUDE)].dropna(subset=["fwd_ret"])
    return df

def calibration(df, label):
    df = df.copy()
    df["correct"] = (df["raw_pred"] == df["truth"]).astype(int)
    bins = np.linspace(0.30, 1.0, 11)
    df["bin"] = pd.cut(df["confidence"], bins, include_lowest=True)
    g = df.groupby("bin", observed=True).agg(
        n=("correct","size"), acc=("correct","mean"), conf=("confidence","mean")
    ).reset_index()
    g["model"] = label
    N = g["n"].sum()
    ece = float((g["n"]/N * (g["acc"]-g["conf"]).abs()).sum())
    oc  = float(((g["conf"]-g["acc"]) * g["n"]/N).sum())
    return g, {
        "model": label, "ece": round(ece,4), "overconf": round(oc,4),
        "accuracy": round(df["correct"].mean(),4),
        "mean_conf": round(df["confidence"].mean(),4),
        "buy%":  round((df["raw_pred"]=="BUY").mean()*100, 1),
        "sell%": round((df["raw_pred"]=="SELL").mean()*100, 1),
        "hold%": round((df["raw_pred"]=="HOLD").mean()*100, 1),
    }

#  PART B — REPAIR SWEEP ════
def signed_size(row, gate_kind, conf_floor):
    sig = row.get("fused_signal")
    conf = float(row.get("confidence", 0) or 0)
    sent = float(row.get("sentiment_score", 0) or 0)
    if conf < conf_floor or sig == "HOLD":
        return 0.0
    size = min(1.0, max(0.0, (conf - conf_floor) / max(1e-6, 1 - conf_floor)))
    if sig == "BUY":
        if   gate_kind == "strict":  block = sent < -0.10
        elif gate_kind == "extreme": block = (conf > 0.55) and (sent < -0.30)
        elif gate_kind in ("raw", "ensemble", "temp"): block = False
        else: block = False
        return 0.0 if block else +size
    if sig == "SELL":
        if   gate_kind == "strict":  block = sent > +0.05
        elif gate_kind == "extreme": block = (conf > 0.55) and (sent > +0.30)
        elif gate_kind in ("raw", "ensemble", "temp"): block = False
        else: block = False
        return 0.0 if block else -size
    return 0.0

def temperature_scale(df, T=2.0):
    """Softer softmax (T>1 ⇒ flatter); recompute confidence & raw_pred."""
    p = df[["prob_sell","prob_hold","prob_buy"]].values
    eps = 1e-9
    logits = np.log(np.clip(p, eps, 1.0))
    z = logits / T
    z -= z.max(axis=1, keepdims=True)
    sm = np.exp(z); sm /= sm.sum(axis=1, keepdims=True)
    df = df.copy()
    df["prob_sell"], df["prob_hold"], df["prob_buy"] = sm[:,0], sm[:,1], sm[:,2]
    idx = sm.argmax(axis=1)
    df["confidence"] = sm.max(axis=1)
    label_map = {0:"SELL", 1:"HOLD", 2:"BUY"}
    df["raw_pred"]      = [label_map[i] for i in idx]
    df["fused_signal"]  = df["raw_pred"]   # treat as raw for gating
    return df

def apply_ensemble(df_deep, df_tx):
    """Veto deep prediction unless it agrees with Transformer's raw_pred."""
    key = ["symbol","timestamp"]
    tx = df_tx[key + ["raw_pred"]].rename(columns={"raw_pred":"tx_pred"})
    df = df_deep.merge(tx, on=key, how="left")
    df["fused_signal"] = np.where(df["raw_pred"] == df["tx_pred"],
                                   df["raw_pred"], "HOLD")
    return df

def run_variant(fname, name, gate, conf_q=None, *, df_override=None):
    df = bt._load(OUT / fname).copy() if df_override is None else df_override.copy()
    if "fused_signal" not in df.columns or gate in ("raw","temp","ensemble"):
        if "raw_pred" in df.columns and gate != "ensemble":
            df["fused_signal"] = df["raw_pred"]
    cq = conf_q if conf_q is not None else bt.CONF_QUANTILE
    non_hold = df.loc[df["fused_signal"] != "HOLD", "confidence"]
    cf = float(non_hold.quantile(cq)) if len(non_hold) else 0.5
    df["pos_signed"] = df.apply(lambda r: signed_size(r, gate, cf), axis=1)

    pivot_close = df.pivot_table(index="timestamp", columns="symbol",
                                 values="close", aggfunc="last").ffill()
    pivot_pos = df.pivot_table(index="timestamp", columns="symbol",
                               values="pos_signed", aggfunc="last").reindex(
                                   pivot_close.index).ffill().fillna(0)
    pos = pivot_pos.shift(1).fillna(0)
    rets = pivot_close.pct_change().fillna(0)
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    weights = pos.div(gross, axis=0).fillna(0)
    bar_ret = (weights*rets).sum(axis=1)
    flips = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
    bar_ret = bar_ret - flips * (bt.COMMISSION + bt.SLIPPAGE)
    eq = (1 + bar_ret).cumprod() * bt.INITIAL_CAPITAL
    n = int((weights.diff().abs().sum(axis=1) > 1e-6).sum())
    res = bt._metrics(eq, bar_ret, n, name)
    res["equity"] = eq
    return res

def main():
    print("="*72); print(" Deep-fusion diagnosis & repair"); print("="*72)

    # PART A 
    cal_rows = []
    cal_bins = []
    print("\n[A] Calibration audit:")
    for label, fname in DEEP_MODELS.items():
        df = add_truth(pd.read_csv(OUT / fname))
        g, summary = calibration(df, label)
        cal_bins.append(g); cal_rows.append(summary)
        print(f"  {label:24s} acc={summary['accuracy']:.3f}  conf={summary['mean_conf']:.3f}  "
              f"ECE={summary['ece']:.3f}  overconf={summary['overconf']:+.3f}  "
              f"signal mix BUY={summary['buy%']}% SELL={summary['sell%']}% HOLD={summary['hold%']}%")
    pd.DataFrame(cal_rows).to_csv(FINAL / "v3_deep_fusion_calibration.csv", index=False)

    # PART B 
    print("\n[B] Repair sweep:")
    repair_rows, curves = [], {}

    for label, fname in DEEP_MODELS.items():
        df_tx_loaded = bt._load(OUT / TX_FILE)
        for tag, descr, gate, cq, df_override in [
            ("R1", "default (current)",         "strict",  None, None),
            ("R2", "raw (no gate)",             "raw",     None, None),
            ("R3", "stricter conf (top-20%)",   "strict",  0.80, None),
            ("R4", "top-decile (top-10%)",      "strict",  0.90, None),
            ("R5", "extreme gate (V8)",         "extreme", None, None),
            ("R6", "ensemble w/ Transformer",   "ensemble",None,
                   apply_ensemble(bt._load(OUT/fname), df_tx_loaded)),
            ("R7", "temperature-scaled (T=2)",  "temp",    None,
                   temperature_scale(bt._load(OUT/fname), T=2.0)),
        ]:
            r = run_variant(fname, f"{label}/{tag}", gate, conf_q=cq,
                            df_override=df_override)
            curves[f"{label} {tag}"] = r["equity"]
            repair_rows.append({
                "model": label, "tag": tag, "config": descr,
                "return_%":   r["total_return_%"],
                "sharpe":     r["sharpe"],
                "max_dd_%":   r["max_drawdown_%"],
                "profit_factor": r["profit_factor"],
                "n_trades":   r["n_trades"],
                "win_rate_%": r["win_rate_%"],
            })
            print(f"  {label:24s} {tag} {descr:30s}  ret={r['total_return_%']:+6.2f}%  "
                  f"Sharpe={r['sharpe']:+.2f}  trades={r['n_trades']}")

    pd.DataFrame(repair_rows).to_csv(FINAL / "v3_deep_fusion_repair.csv", index=False)

    # PLOTS
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))

    # confidence histograms
    cols = {"CrossModalTransformer":"#a78bfa", "SentimentLSTM":"#22d3ee",
            "Transformer (anchor)":"#f97316"}
    for label, fname in DEEP_MODELS.items():
        df = pd.read_csv(OUT/fname)
        ax[0,0].hist(df["confidence"], bins=30, alpha=0.55, color=cols[label],
                     label=label, density=True)
    df_tx = pd.read_csv(OUT/TX_FILE)
    ax[0,0].hist(df_tx["confidence"], bins=30, alpha=0.4,
                 color=cols["Transformer (anchor)"], label="Transformer (anchor)", density=True)
    ax[0,0].set_title("Confidence distributions")
    ax[0,0].set_xlabel("max softmax probability"); ax[0,0].set_ylabel("density")
    ax[0,0].grid(alpha=.3); ax[0,0].legend()

    # reliability
    ax[0,1].plot([0.3,1],[0.3,1], "--", color="#888", lw=1, label="perfect")
    for g in cal_bins:
        lab = g["model"].iloc[0]
        ax[0,1].plot(g["conf"], g["acc"], "o-", color=cols[lab], lw=2, label=lab)
    ax[0,1].set_title("Reliability (deep fusion)")
    ax[0,1].set_xlabel("predicted confidence"); ax[0,1].set_ylabel("empirical accuracy")
    ax[0,1].set_xlim(0.30,1.0); ax[0,1].set_ylim(0.20,0.60)
    ax[0,1].grid(alpha=.3); ax[0,1].legend()

    # CrossModal repairs
    for k, eq in curves.items():
        if k.startswith("CrossModalTransformer"):
            ax[1,0].plot(eq.index, eq.values, label=k.split(" ",1)[1], lw=1.5, alpha=0.9)
    ax[1,0].axhline(bt.INITIAL_CAPITAL, color="grey", lw=0.7, ls=":")
    ax[1,0].set_title("CrossModalTransformer — repair sweep")
    ax[1,0].set_ylabel("equity (USD)"); ax[1,0].grid(alpha=.3)
    ax[1,0].legend(loc="best", fontsize=8)

    # SentimentLSTM repairs
    for k, eq in curves.items():
        if k.startswith("SentimentLSTM"):
            ax[1,1].plot(eq.index, eq.values, label=k.split(" ",1)[1], lw=1.5, alpha=0.9)
    ax[1,1].axhline(bt.INITIAL_CAPITAL, color="grey", lw=0.7, ls=":")
    ax[1,1].set_title("SentimentLSTM — repair sweep")
    ax[1,1].set_ylabel("equity (USD)"); ax[1,1].grid(alpha=.3)
    ax[1,1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(CHARTS / "v3_deep_fusion_diagnose.png", dpi=130)
    plt.close()
    print("\n→ saved final_results/v3_deep_fusion_calibration.csv")
    print("→ saved final_results/v3_deep_fusion_repair.csv")
    print("→ saved final_results/charts/v3_deep_fusion_diagnose.png")

if __name__ == "__main__":
    main()
