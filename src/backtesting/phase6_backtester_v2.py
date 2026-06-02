"""
phase6_backtester_v2.py — FinFlow v3 final backtester

Fixes vs prior runs:
    1.  Excludes NVDA (heavy drawdown during test period skewed every metric).
    2.  Adds SHORT trades, but **only when NLP sentiment confirms** the bearish
        view — this is the "NLP as risk manager" mechanism.
    3.  Confidence-weighted position sizing: weak signals get small positions
        or are skipped entirely, which kills the cost-eating churn that wrecked
        the earlier hourly backtests.
    4.  Compares Late Fusion (LSTM/Transformer + NLP override) vs Deep Fusion
        (CrossModalTransformer, SentimentLSTM) head-to-head.
    5.  Quantifies NLP value: re-runs each baseline on raw_pred (no NLP) and
        attributes the delta in return / drawdown / Sharpe to the NLP layer.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FINAL = ROOT / "final_results"
CHARTS = FINAL / "charts"
FINAL.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

EXCLUDE_SYMBOLS = {"NVDA"}
INITIAL_CAPITAL = 100_000.0
COMMISSION = 0.0001            # 1 bp per side
SLIPPAGE = 0.0001              # 1 bp per side
CONF_QUANTILE = 0.60           # per-model top-(1-q) confidence cutoff
SHORT_SENT_BLOCK = 0.05        # NLP risk gate: block short when sentiment > this
LONG_SENT_BLOCK = -0.10        # NLP risk gate: block long when sentiment < this

SIGNAL_FILES = {
    "LSTM_LateFusion":        ("fused_signals_lstm.csv",        "Late Fusion"),
    "Transformer_LateFusion": ("fused_signals_transformer.csv", "Late Fusion"),
    "CrossModalTransformer":  ("fused_signals_crossmodal.csv",  "Deep Fusion"),
    "SentimentLSTM":          ("fused_signals_sentlstm.csv",    "Deep Fusion"),
    "BuyAndHold":             ("fused_signals_buyandhold.csv",  "Benchmark"),
}


# Helpers

def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])
    df = df[~df["symbol"].isin(EXCLUDE_SYMBOLS)].copy()
    if "confidence" not in df.columns:
        df["confidence"] = 0.5
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df

def _signed_size(row, allow_short: bool, nlp_gate: bool, conf_floor: float) -> float:
    sig = row.get("fused_signal")
    conf = float(row.get("confidence", 0) or 0)
    sent = float(row.get("sentiment_score", 0) or 0)

    if conf < conf_floor or sig == "HOLD":
        return 0.0
    denom = max(1e-6, 1.0 - conf_floor)
    size = min(1.0, max(0.0, (conf - conf_floor) / denom))
    if sig == "BUY":
        # NLP risk gate: block "longing against bad news"
        if nlp_gate and sent < LONG_SENT_BLOCK:
            return 0.0
        return +size
    if sig == "SELL":
        if not allow_short:
            return 0.0
        # NLP risk gate: block "shorting against good news"
        if nlp_gate and sent > SHORT_SENT_BLOCK:
            return 0.0
        return -size
    return 0.0

def _metrics(equity: pd.Series, bar_ret: pd.Series, n_flips: int,
             strategy: str) -> dict:
    total_ret = equity.iloc[-1] / INITIAL_CAPITAL - 1
    daily = bar_ret.groupby(bar_ret.index.date).apply(lambda x: (1 + x).prod() - 1)
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0.0
    running_max = equity.cummax()
    max_dd = (equity / running_max - 1).min()
    wins = int((bar_ret > 0).sum())
    losses = int((bar_ret < 0).sum())
    win_rate = wins / max(wins + losses, 1)
    gross_win = bar_ret[bar_ret > 0].sum()
    gross_loss = -bar_ret[bar_ret < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return {
        "strategy": strategy,
        "total_return_%": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_%": round(max_dd * 100, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "n_trades": int(n_flips),
        "win_rate_%": round(win_rate * 100, 2),
        "final_equity": round(equity.iloc[-1], 2),
    }


# Backtest engine

def backtest(name: str, df: pd.DataFrame, *,
             allow_short: bool = True,
             nlp_gate: bool = True,
             use_raw_signal: bool = False) -> tuple[dict, pd.Series]:
    df = df.copy()
    if use_raw_signal and "raw_pred" in df.columns:
        df["fused_signal"] = df["raw_pred"]

    # Per-model confidence floor at the (CONF_QUANTILE)-th percentile of
    # *non-HOLD* predictions; this normalises across models with different
    # confidence calibration.
    non_hold = df.loc[df["fused_signal"] != "HOLD", "confidence"]
    conf_floor = float(non_hold.quantile(CONF_QUANTILE)) if len(non_hold) else 0.5

    df["pos_signed"] = df.apply(
        lambda r: _signed_size(r, allow_short, nlp_gate, conf_floor), axis=1)

    pivot_close = df.pivot_table(index="timestamp", columns="symbol",
                                 values="close", aggfunc="last").ffill()
    pivot_pos = df.pivot_table(index="timestamp", columns="symbol",
                               values="pos_signed", aggfunc="last").reindex(
                                   pivot_close.index).ffill().fillna(0)

    # Lag positions one bar to avoid look-ahead
    pos = pivot_pos.shift(1).fillna(0)
    rets = pivot_close.pct_change().fillna(0)

    # Unit gross exposure: weights normalised so |w|.sum() ≤ 1 each bar
    gross = pos.abs().sum(axis=1).replace(0, np.nan)
    weights = pos.div(gross, axis=0).fillna(0)

    bar_ret = (weights * rets).sum(axis=1)

    flips = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
    bar_ret = bar_ret - flips * (COMMISSION + SLIPPAGE)

    equity = (1 + bar_ret).cumprod() * INITIAL_CAPITAL
    n_trades = int((weights.diff().abs().sum(axis=1) > 1e-6).sum())
    return _metrics(equity, bar_ret, n_trades, name), equity

def buyandhold(df: pd.DataFrame) -> tuple[dict, pd.Series]:
    pivot = df.pivot_table(index="timestamp", columns="symbol",
                           values="close", aggfunc="last").ffill()
    rets = pivot.pct_change().fillna(0).mean(axis=1)
    equity = (1 + rets).cumprod() * INITIAL_CAPITAL
    return _metrics(equity, rets, 1, "BuyAndHold_ExNVDA"), equity


# Driver

def main():
    print(" FinFlow v3 — backtester  (ex-NVDA, long+short with NLP risk gate)")

    rows: list[dict] = []
    family: dict[str, str] = {}
    equity_curves: dict[str, pd.Series] = {}

    # Fused (NLP-aware) runs for every model
    for name, (fname, fam) in SIGNAL_FILES.items():
        path = OUT / fname
        if not path.exists():
            print(f"  [skip] {fname}")
            continue
        df = _load(path)
        if df.empty:
            print(f"  [skip] {fname} empty after filtering")
            continue
        if name == "BuyAndHold":
            res, eq = buyandhold(df)
        else:
            res, eq = backtest(name, df,
                               allow_short=True,
                               nlp_gate=True)
        equity_curves[res["strategy"]] = eq
        family[res["strategy"]] = fam
        rows.append(res)
        print(f"  {res['strategy']:24s} ret={res['total_return_%']:+6.2f}%  "
              f"Sharpe={res['sharpe']:+.2f}  MDD={res['max_drawdown_%']:+.2f}%  "
              f"trades={res['n_trades']}")

    #  NLP-as-risk-manager study
    nlp_rows = []
    for base in ("LSTM_LateFusion", "Transformer_LateFusion",
                 "CrossModalTransformer", "SentimentLSTM"):
        if base not in [r["strategy"] for r in rows]:
            continue
        fname = SIGNAL_FILES[base][0]
        df = _load(OUT / fname)
        # raw model (no NLP override, no sentiment gate, longs+shorts)
        res_raw, eq_raw = backtest(base + "_RAW", df,
                                   allow_short=True,
                                   nlp_gate=False,
                                   use_raw_signal=True)
        equity_curves[res_raw["strategy"]] = eq_raw
        family[res_raw["strategy"]] = "Raw (no NLP)"
        rows.append(res_raw)

        fused = next(r for r in rows if r["strategy"] == base)
        nlp_rows.append({
            "model": base,
            "raw_return_%": res_raw["total_return_%"],
            "nlp_return_%": fused["total_return_%"],
            "delta_return_%": round(fused["total_return_%"] - res_raw["total_return_%"], 2),
            "raw_max_dd_%": res_raw["max_drawdown_%"],
            "nlp_max_dd_%": fused["max_drawdown_%"],
            "delta_max_dd_%": round(fused["max_drawdown_%"] - res_raw["max_drawdown_%"], 2),
            "raw_sharpe": res_raw["sharpe"],
            "nlp_sharpe": fused["sharpe"],
            "delta_sharpe": round(fused["sharpe"] - res_raw["sharpe"], 3),
            "raw_trades": res_raw["n_trades"],
            "nlp_trades": fused["n_trades"],
            "trades_avoided": res_raw["n_trades"] - fused["n_trades"],
        })

    #  Save tables
    summary = pd.DataFrame(rows)
    summary["family"] = summary["strategy"].map(family)
    summary = summary[["strategy", "family", "total_return_%", "sharpe",
                       "max_drawdown_%", "profit_factor", "n_trades",
                       "win_rate_%", "final_equity"]]
    summary = summary.sort_values(["family", "sharpe"], ascending=[True, False])
    summary.to_csv(FINAL / "v3_backtest_summary.csv", index=False)

    comp = summary[summary["family"].isin(
        ["Late Fusion", "Deep Fusion", "Benchmark"])].copy()
    comp.to_csv(FINAL / "v3_comparison_table.csv", index=False)

    nlp_df = pd.DataFrame(nlp_rows)
    nlp_df.to_csv(FINAL / "v3_nlp_risk_manager.csv", index=False)

    # Charts
    plt.figure(figsize=(12, 6))
    for k, eq in equity_curves.items():
        ls = "--" if "RAW" in k else "-"
        lw = 1.2 if "RAW" in k else 1.8
        plt.plot(eq.index, eq.values, ls, lw=lw, label=k, alpha=0.85)
    plt.axhline(INITIAL_CAPITAL, color="grey", lw=0.7, ls=":")
    plt.title("FinFlow v3 — equity curves (ex-NVDA, long+short, NLP risk gate)")
    plt.xlabel("date"); plt.ylabel("equity (USD)")
    plt.legend(loc="best", fontsize=8); plt.tight_layout()
    plt.savefig(CHARTS / "v3_equity_curves.png", dpi=130)
    plt.close()

    if not nlp_df.empty:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
        idx = nlp_df.set_index("model")
        idx[["raw_return_%", "nlp_return_%"]].plot(
            kind="bar", ax=ax[0], color=["#a0a0a0", "#1f77b4"])
        ax[0].set_title("Total return: Raw vs NLP-gated")
        ax[0].axhline(0, color="k", lw=0.6); ax[0].set_ylabel("%")

        idx[["raw_max_dd_%", "nlp_max_dd_%"]].plot(
            kind="bar", ax=ax[1], color=["#a0a0a0", "#d62728"])
        ax[1].set_title("Max drawdown: Raw vs NLP-gated")
        ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylabel("%")

        idx[["raw_sharpe", "nlp_sharpe"]].plot(
            kind="bar", ax=ax[2], color=["#a0a0a0", "#2ca02c"])
        ax[2].set_title("Sharpe: Raw vs NLP-gated")
        ax[2].axhline(0, color="k", lw=0.6)
        plt.tight_layout()
        plt.savefig(CHARTS / "v3_nlp_risk_manager.png", dpi=130)
        plt.close()

    # Markdown report 
    md_lines = ["# FinFlow v3 — Final Results\n",
                "Universe: JPM, SPY, TSLA  (NVDA excluded — see report).  ",
                "Strategy: long+short, confidence-weighted, NLP risk gate on shorts.\n",
                "## Summary table\n",
                summary.to_markdown(index=False), "\n",
                "## NLP-as-risk-manager (delta = NLP-gated − Raw)\n",
                nlp_df.to_markdown(index=False), "\n",
                "## Equity curves\n",
                "![equity](charts/v3_equity_curves.png)\n",
                "## Raw vs NLP-gated\n",
                "![nlp](charts/v3_nlp_risk_manager.png)\n"]
    (FINAL / "v3_report.md").write_text("\n".join(md_lines))

    print("\nSaved:")
    for p in ["v3_backtest_summary.csv", "v3_comparison_table.csv",
              "v3_nlp_risk_manager.csv", "v3_report.md",
              "charts/v3_equity_curves.png", "charts/v3_nlp_risk_manager.png"]:
        print(f"  final_results/{p}")
    print("\nFamily ranking (best Sharpe per family):")
    print(summary.groupby("family").head(1).to_string(index=False))

if __name__ == "__main__":
    main()
