"""
phase9_lstm_gate_tuning.py — try targeted NLP-gate variants on the LSTM signals.

Motivation (from phase8):
    The LSTM's contrarian-sentiment signals are near coin-flips, so the *default*
    gate (block longs at s<-0.10, block shorts at s>+0.05) prunes about as many
    winners as losers. Five configurations are tested:

        V1 strict      default thresholds (current Protocol-B setting)
        V2 loose       only block on extreme sentiment, |s| > 0.30
        V3 news-cond.  apply gate only when news_count > 0  (no-news bars pass)
        V4 conf-cond.  apply gate only when confidence > 0.55  (top half)
        V5 asymmetric  block longs against bad news; shorts always allowed
        V6 raw         no gate at all (baseline)

Transformer is kept at the proven setting (V1) and shown for context.

"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phase6_backtester_v2 as bt   # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FINAL = ROOT / "final_results"
CHARTS = FINAL / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

def signed_size_custom(row, allow_short, gate_kind, conf_floor):
    """Variant gate logic. Returns position size in [-1, +1]."""
    sig = row.get("fused_signal")
    conf = float(row.get("confidence", 0) or 0)
    sent = float(row.get("sentiment_score", 0) or 0)
    news = float(row.get("news_count", 0) or 0)

    if conf < conf_floor or sig == "HOLD":
        return 0.0
    size = min(1.0, max(0.0, (conf - conf_floor) / max(1e-6, 1 - conf_floor)))

    if sig == "BUY":
        block = False
        if   gate_kind == "strict":      block = sent < -0.10
        elif gate_kind == "loose":       block = sent < -0.30
        elif gate_kind == "news_cond":   block = (news > 0) and (sent < -0.10)
        elif gate_kind == "conf_cond":   block = (conf > 0.55) and (sent < -0.10)
        elif gate_kind == "asymmetric":  block = sent < -0.10
        elif gate_kind == "asym_conf":   block = (conf > 0.55) and (sent < -0.10)
        elif gate_kind == "extreme":     block = (conf > 0.55) and (sent < -0.30)
        elif gate_kind == "raw":         block = False
        return 0.0 if block else +size

    if sig == "SELL":
        if not allow_short: return 0.0
        block = False
        if   gate_kind == "strict":      block = sent > +0.05
        elif gate_kind == "loose":       block = sent > +0.30
        elif gate_kind == "news_cond":   block = (news > 0) and (sent > +0.05)
        elif gate_kind == "conf_cond":   block = (conf > 0.55) and (sent > +0.05)
        elif gate_kind == "asymmetric":  block = False           # shorts always allowed
        elif gate_kind == "asym_conf":   block = False           # shorts always allowed
        elif gate_kind == "extreme":     block = (conf > 0.55) and (sent > +0.30)
        elif gate_kind == "raw":         block = False
        return 0.0 if block else -size

    return 0.0

def run_variant(fname: str, model_label: str, gate_kind: str,
                use_raw: bool = False) -> dict:
    df = bt._load(OUT / fname).copy()
    if use_raw and "raw_pred" in df.columns:
        df["fused_signal"] = df["raw_pred"]

    non_hold = df.loc[df["fused_signal"] != "HOLD", "confidence"]
    cf = float(non_hold.quantile(bt.CONF_QUANTILE)) if len(non_hold) else 0.5
    df["pos_signed"] = df.apply(lambda r: signed_size_custom(r, True, gate_kind, cf), axis=1)

    pivot_close = df.pivot_table(index="timestamp", columns="symbol",
                                 values="close", aggfunc="last").ffill()
    pivot_pos = df.pivot_table(index="timestamp", columns="symbol",
                               values="pos_signed", aggfunc="last").reindex(
                                   pivot_close.index).ffill().fillna(0)
    pos = pivot_pos.shift(1).fillna(0)
    rets = pivot_close.pct_change().fillna(0)
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    weights = pos.div(gross, axis=0).fillna(0)
    bar_ret = (weights * rets).sum(axis=1)
    flips = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
    bar_ret = bar_ret - flips * (bt.COMMISSION + bt.SLIPPAGE)
    equity = (1 + bar_ret).cumprod() * bt.INITIAL_CAPITAL
    n_trades = int((weights.diff().abs().sum(axis=1) > 1e-6).sum())
    res = bt._metrics(equity, bar_ret, n_trades, f"{model_label}/{gate_kind}")
    res["equity"] = equity
    return res

def main():
    print(" LSTM gate-tuning sweep  (Transformer shown unchanged)")

    rows = []
    curves = {}

    # LSTM variants
    for kind, descr in [
        ("strict",     "V1 strict (current default)"),
        ("loose",      "V2 loose  |s|>0.30 only"),
        ("news_cond",  "V3 news-conditional"),
        ("conf_cond",  "V4 conf-conditional (>0.55)"),
        ("asymmetric", "V5 asymmetric (longs gated only)"),
        ("asym_conf",  "V7 asym + conf (longs only, conf>0.55)"),
        ("extreme",    "V8 extreme (|s|>0.30 AND conf>0.55)"),
        ("raw",        "V6 raw (no gate)"),
    ]:
        r = run_variant("fused_signals_lstm.csv", "LSTM", kind,
                        use_raw=(kind == "raw"))
        curves[descr] = r["equity"]
        rows.append({
            "config": descr,
            "model": "LSTM",
            "return_%":   r["total_return_%"],
            "sharpe":     r["sharpe"],
            "max_dd_%":   r["max_drawdown_%"],
            "profit_factor": r["profit_factor"],
            "n_trades":   r["n_trades"],
            "win_rate_%": r["win_rate_%"],
            "final_eq":   r["final_equity"],
        })
        print(f"  LSTM/{kind:<11s} ret={r['total_return_%']:+6.2f}%  "
              f"Sharpe={r['sharpe']:+.2f}  MDD={r['max_drawdown_%']:+.2f}%  "
              f"trades={r['n_trades']}")

    # Transformer V1 (anchor)
    r = run_variant("fused_signals_transformer.csv", "Transformer", "strict")
    curves["Transformer V1 (anchor)"] = r["equity"]
    rows.append({
        "config": "Transformer V1 (anchor / unchanged)",
        "model": "Transformer",
        "return_%":   r["total_return_%"],
        "sharpe":     r["sharpe"],
        "max_dd_%":   r["max_drawdown_%"],
        "profit_factor": r["profit_factor"],
        "n_trades":   r["n_trades"],
        "win_rate_%": r["win_rate_%"],
        "final_eq":   r["final_equity"],
    })
    print(f"  Transformer/strict ret={r['total_return_%']:+6.2f}%  "
          f"Sharpe={r['sharpe']:+.2f}  trades={r['n_trades']}")

    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "v3_lstm_gate_tuning.csv", index=False)

    # Plot
    plt.figure(figsize=(13, 5.5))
    style = {
        "V1 strict (current default)":          ("#888",    "--", 1.5),
        "V2 loose  |s|>0.30 only":              ("#3b82f6", "-",  2),
        "V3 news-conditional":                  ("#22d3ee", "-",  2),
        "V4 conf-conditional (>0.55)":          ("#a78bfa", "-",  2),
        "V5 asymmetric (longs gated only)":     ("#22c55e", "-",  2),
        "V7 asym + conf (longs only, conf>0.55)": ("#fbbf24","-",  2),
        "V8 extreme (|s|>0.30 AND conf>0.55)":  ("#ec4899", "-",  2),
        "V6 raw (no gate)":                     ("#ef4444", ":",  1.5),
        "Transformer V1 (anchor)":              ("#f97316", "-",  2.5),
    }
    for k, eq in curves.items():
        c, ls, lw = style.get(k, ("#aaa", "-", 1))
        plt.plot(eq.index, eq.values, ls, lw=lw, color=c, label=k, alpha=0.9)
    plt.axhline(bt.INITIAL_CAPITAL, color="grey", lw=0.7, ls=":")
    plt.title("LSTM gate-tuning sweep — Transformer kept unchanged")
    plt.xlabel("date"); plt.ylabel("equity (USD)")
    plt.legend(loc="best", fontsize=9); plt.tight_layout()
    plt.savefig(CHARTS / "v3_lstm_gate_tuning.png", dpi=130)
    plt.close()

    print("\n" + df.to_string(index=False))
    print("\nsaved final_results/v3_lstm_gate_tuning.csv")
    print("saved final_results/charts/v3_lstm_gate_tuning.png")

    # Pick winner among LSTM rows
    lstm_only = df[df["model"] == "LSTM"].copy()
    best = lstm_only.sort_values("sharpe", ascending=False).iloc[0]
    print(f"\n→ Best LSTM gate config: {best['config']}")
    print(f"   return = {best['return_%']:+.2f}%  Sharpe = {best['sharpe']:+.2f}  "
          f"MDD = {best['max_dd_%']:+.2f}%  trades = {int(best['n_trades'])}")

if __name__ == "__main__":
    main()
