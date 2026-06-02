"""
phase11_meta_ensemble.py — meta-ensemble of FinFlow 3.0's best post-fix models.

Members (each at their tuned configuration from phases 6/9/10):
    M1  Transformer + Late-Fusion + NLP risk gate          (strict)        — Sharpe 1.66
    M2  LSTM + V4 conf-conditional gate                                    — Sharpe 0.30
    M3  CrossModalTransformer × Transformer agreement (R6)                 — Sharpe 0.81
    M4  SentimentLSTM         × Transformer agreement (R6)                 — Sharpe 0.64

Ensemble aggregators (over the per-symbol signed weights):
    E1  equal-weight average                  (mean of active member positions)
    E2  Sharpe-weighted average               (weights ∝ individual Sharpe, clipped at 0)
    E3  majority vote                         (sign of sum, only act if ≥⌈M/2⌉ agree)
    E4  strict consensus                      (act only when all members agree on sign)

Sub-grids:
    G2 = {M1, M2}        unimodal pair
    G3 = {M1, M2, M3}    + cross-modal
    G4 = {M1, M2, M3, M4} all four

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

#  Member-position builders 
def position_M1_transformer_lf_nlp() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict NLP gate (Protocol-B default) on the Transformer LF signal."""
    df = bt._load(OUT / "fused_signals_transformer.csv").copy()
    non_hold = df.loc[df["fused_signal"] != "HOLD", "confidence"]
    cf = float(non_hold.quantile(bt.CONF_QUANTILE))
    df["pos_signed"] = df.apply(lambda r: bt._signed_size(r, True, True, cf), axis=1)
    return _to_weights(df)

def position_M2_lstm_v4() -> tuple[pd.DataFrame, pd.DataFrame]:
    """V4 conf-conditional gate (block only when conf>0.55) on the LSTM LF signal."""
    df = bt._load(OUT / "fused_signals_lstm.csv").copy()
    non_hold = df.loc[df["fused_signal"] != "HOLD", "confidence"]
    cf = float(non_hold.quantile(bt.CONF_QUANTILE))

    def sized(r):
        sig = r["fused_signal"]; conf = float(r["confidence"]); sent = float(r["sentiment_score"])
        if conf < cf or sig == "HOLD": return 0.0
        size = min(1, max(0, (conf - cf) / max(1e-6, 1 - cf)))
        if sig == "BUY":
            return 0.0 if (conf > 0.55 and sent < -0.10) else +size
        if sig == "SELL":
            return 0.0 if (conf > 0.55 and sent > +0.05) else -size
        return 0.0
    df["pos_signed"] = df.apply(sized, axis=1)
    return _to_weights(df)

def position_M3_crossmodal_r6() -> tuple[pd.DataFrame, pd.DataFrame]:
    """R6 ensemble: act on CrossModal only when it agrees with Transformer's raw_pred."""
    df = bt._load(OUT / "fused_signals_crossmodal.csv").copy()
    tx = bt._load(OUT / "fused_signals_transformer.csv")[
        ["symbol", "timestamp", "raw_pred"]].rename(columns={"raw_pred": "tx_pred"})
    df = df.merge(tx, on=["symbol", "timestamp"], how="left")
    df["fused_signal"] = np.where(df["raw_pred"] == df["tx_pred"], df["raw_pred"], "HOLD")
    non_hold = df.loc[df["fused_signal"] != "HOLD", "confidence"]
    cf = float(non_hold.quantile(bt.CONF_QUANTILE)) if len(non_hold) else 0.5
    df["pos_signed"] = df.apply(lambda r: bt._signed_size(r, True, False, cf), axis=1)
    return _to_weights(df)

def position_M4_sentlstm_r6() -> tuple[pd.DataFrame, pd.DataFrame]:
    """R6 ensemble for SentimentLSTM."""
    df = bt._load(OUT / "fused_signals_sentlstm.csv").copy()
    tx = bt._load(OUT / "fused_signals_transformer.csv")[
        ["symbol", "timestamp", "raw_pred"]].rename(columns={"raw_pred": "tx_pred"})
    df = df.merge(tx, on=["symbol", "timestamp"], how="left")
    df["fused_signal"] = np.where(df["raw_pred"] == df["tx_pred"], df["raw_pred"], "HOLD")
    non_hold = df.loc[df["fused_signal"] != "HOLD", "confidence"]
    cf = float(non_hold.quantile(bt.CONF_QUANTILE)) if len(non_hold) else 0.5
    df["pos_signed"] = df.apply(lambda r: bt._signed_size(r, True, False, cf), axis=1)
    return _to_weights(df)

def _to_weights(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot to (timestamp × symbol) weights and close-price matrices, with unit-gross
    normalisation applied at the member level (so each member's risk is bounded)."""
    pivot_close = df.pivot_table(index="timestamp", columns="symbol",
                                 values="close", aggfunc="last").ffill()
    pivot_pos = df.pivot_table(index="timestamp", columns="symbol",
                               values="pos_signed", aggfunc="last").reindex(
                                   pivot_close.index).ffill().fillna(0)
    # member-level unit-gross normalisation
    gross = pivot_pos.abs().sum(axis=1).replace(0, np.nan)
    weights = pivot_pos.div(gross, axis=0).fillna(0)
    return weights, pivot_close

#  Backtest from a weights matrix 
def backtest_weights(weights: pd.DataFrame, prices: pd.DataFrame, name: str
                     ) -> tuple[dict, pd.Series]:
    pos = weights.shift(1).fillna(0)
    rets = prices.pct_change().fillna(0)
    bar_ret = (pos * rets).sum(axis=1)
    flips = (pos - pos.shift(1).fillna(0)).abs().sum(axis=1)
    bar_ret = bar_ret - flips * (bt.COMMISSION + bt.SLIPPAGE)
    eq = (1 + bar_ret).cumprod() * bt.INITIAL_CAPITAL
    n_trades = int((pos.diff().abs().sum(axis=1) > 1e-6).sum())
    return bt._metrics(eq, bar_ret, n_trades, name), eq

#  Aggregators
def agg_equal(W_list):                # E1
    return sum(W_list) / len(W_list)

def agg_weighted(W_list, sharpes):    # E2
    w = np.clip(np.array(sharpes, dtype=float), 0, None)
    if w.sum() == 0: w = np.ones_like(w)
    w /= w.sum()
    out = sum(W * a for W, a in zip(W_list, w))
    return out

def agg_majority(W_list):             # E3
    M = len(W_list); thresh = (M // 2) + 1
    signs = sum(np.sign(W) for W in W_list)
    keep = (signs.abs() >= thresh).astype(int)
    out = (sum(W_list) / M) * keep        # average magnitude when majority agree
    return out

def agg_consensus(W_list):            # E4
    out = W_list[0].copy()
    sign0 = np.sign(out)
    agree = sign0 != 0
    for W in W_list[1:]:
        agree &= (np.sign(W) == sign0)
    return (sum(W_list) / len(W_list)) * agree.astype(int)

#  Driver 
SHARPES = {"M1": 1.66, "M2": 0.30, "M3": 0.81, "M4": 0.64}
LABELS  = {"M1": "Transformer+LF+NLP",
           "M2": "LSTM+V4",
           "M3": "CrossModal R6",
           "M4": "SentimentLSTM R6"}

def main():
    print("="*72); print(" Meta-ensemble — best 2/3/4 FinFlow 3.0 members"); print("="*72)

    # 1) Build each member's weights matrix
    M1_w, prices = position_M1_transformer_lf_nlp()
    M2_w, _      = position_M2_lstm_v4()
    M3_w, _      = position_M3_crossmodal_r6()
    M4_w, _      = position_M4_sentlstm_r6()

    # Align indices (use Transformer's timestamp index as canonical)
    idx = M1_w.index
    members = {"M1": M1_w, "M2": M2_w.reindex(idx).fillna(0),
               "M3": M3_w.reindex(idx).fillna(0), "M4": M4_w.reindex(idx).fillna(0)}
    prices = prices.reindex(idx).ffill()

    # 2) Member runs (anchor)
    rows, curves = [], {}
    for tag, W in members.items():
        res, eq = backtest_weights(W, prices, LABELS[tag])
        rows.append({"tier": "member", "tag": tag, "config": LABELS[tag],
                     "return_%":  res["total_return_%"], "sharpe": res["sharpe"],
                     "max_drawdown_%": res["max_drawdown_%"], "n_trades": res["n_trades"],
                     "win_rate_%": res["win_rate_%"], "final_eq": res["final_equity"]})
        curves[f"member · {tag} {LABELS[tag]}"] = eq
        print(f"  {tag} {LABELS[tag]:24s} ret={res['total_return_%']:+6.2f}%  "
              f"Sharpe={res['sharpe']:+.2f}  trades={res['n_trades']}")

    # 3) Ensemble grids
    GRIDS = {
        "G2": ["M1", "M2"],
        "G3": ["M1", "M2", "M3"],
        "G4": ["M1", "M2", "M3", "M4"],
    }
    AGGS = {
        "E1": ("equal",        lambda Ws, ss: agg_equal(Ws)),
        "E2": ("sharpe-wtd",   lambda Ws, ss: agg_weighted(Ws, ss)),
        "E3": ("majority",     lambda Ws, ss: agg_majority(Ws)),
        "E4": ("consensus",    lambda Ws, ss: agg_consensus(Ws)),
    }
    print()
    for gtag, gmems in GRIDS.items():
        Ws  = [members[m] for m in gmems]
        ss  = [SHARPES[m] for m in gmems]
        for atag, (aname, fn) in AGGS.items():
            W = fn(Ws, ss)
            # re-normalise meta weights so |w|.sum() ≤ 1 each bar
            gross = W.abs().sum(axis=1).replace(0, np.nan)
            W = W.div(gross, axis=0).fillna(0)
            label = f"{gtag}·{atag} {aname:<10s} ({'+'.join(gmems)})"
            res, eq = backtest_weights(W, prices, label)
            rows.append({"tier": "ensemble", "tag": f"{gtag}·{atag}", "config": label,
                         "return_%":  res["total_return_%"], "sharpe": res["sharpe"],
                         "max_drawdown_%": res["max_drawdown_%"], "n_trades": res["n_trades"],
                         "win_rate_%": res["win_rate_%"], "final_eq": res["final_equity"]})
            curves[label] = eq
            print(f"  {label:60s} ret={res['total_return_%']:+6.2f}%  "
                  f"Sharpe={res['sharpe']:+.2f}  MDD={res['max_drawdown_%']:+.2f}%  "
                  f"trades={res['n_trades']}")

    # 4) Add Buy & Hold for context
    bh = bt.buyandhold(bt._load(OUT / "fused_signals_buyandhold.csv"))
    rows.append({"tier": "benchmark", "tag": "BH", "config": "Buy&Hold ex-NVDA",
                 "return_%":  bh[0]["total_return_%"], "sharpe": bh[0]["sharpe"],
                 "max_drawdown_%": bh[0]["max_drawdown_%"], "n_trades": 1,
                 "win_rate_%": bh[0]["win_rate_%"], "final_eq": bh[0]["final_equity"]})
    curves["benchmark · Buy&Hold (ex-NVDA)"] = bh[1]

    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "v3_ensemble_summary.csv", index=False)

    # comparison table
    df_sorted = df.sort_values("sharpe", ascending=False)
    df_sorted.to_csv(FINAL / "v3_ensemble_comparison.csv", index=False)

    # 5) Plot — top 6 by Sharpe + members + BH
    plt.figure(figsize=(13, 6.5))
    top = df_sorted.head(6)["config"].tolist()
    show = list(set(top + [v for v in curves if v.startswith("member") or v.startswith("benchmark")]))
    palette = plt.cm.tab10.colors + plt.cm.Set2.colors
    for i, k in enumerate(curves):
        if k not in show: continue
        eq = curves[k]
        ls = "-" if "ensemble" in k.lower() or "G" in k.split()[0] else \
             ":" if "benchmark" in k else "--"
        lw = 2.4 if "G" in k.split()[0] else 1.6
        plt.plot(eq.index, eq.values, ls, lw=lw, color=palette[i % len(palette)],
                 label=k, alpha=0.85)
    plt.axhline(bt.INITIAL_CAPITAL, color="grey", lw=0.7, ls=":")
    plt.title("FinFlow 3.0 — meta-ensemble vs members vs benchmark")
    plt.xlabel("date"); plt.ylabel("equity (USD)")
    plt.legend(loc="best", fontsize=8); plt.tight_layout()
    plt.savefig(CHARTS / "v3_ensemble_equity.png", dpi=130)
    plt.close()

    # 6) Best ensemble announcement
    print()
    best = df_sorted[df_sorted["tier"] == "ensemble"].iloc[0]
    print(f"  ★ Best ensemble: {best['config']}")
    print(f"     return = {best['return_%']:+.2f}%  Sharpe = {best['sharpe']:+.2f}  "
          f"MDD = {best['max_drawdown_%']:+.2f}%  trades = {int(best['n_trades'])}")

def export_ensembles_to_webapp(top_specs, members, prices):
    """Save the top-N ensemble results in webapp JSON format and refresh manifest."""
    import json
    WEBAPP_DATA = ROOT / "webapp" / "data"
    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

    AGGS = {"E1": agg_equal,
            "E2": lambda Ws, ss: agg_weighted(Ws, ss),
            "E3": agg_majority,
            "E4": agg_consensus}

    for spec in top_specs:
        gtag, atag, name, slug = spec["gtag"], spec["atag"], spec["name"], spec["slug"]
        gmems = spec["mems"]
        Ws = [members[m] for m in gmems]
        ss = [SHARPES[m] for m in gmems]
        if atag == "E2":
            W = AGGS[atag](Ws, ss)
        else:
            W = AGGS[atag](Ws)
        gross = W.abs().sum(axis=1).replace(0, np.nan)
        W = W.div(gross, axis=0).fillna(0)
        res, eq = backtest_weights(W, prices, name)

        # downsample equity curve
        if len(eq) > 1500:
            step = int(np.ceil(len(eq) / 1500))
            eq = eq.iloc[::step]
        equity_pts = [{"t": ts.isoformat(), "v": round(float(v), 2)}
                      for ts, v in eq.items()]

        # trade events from W flips
        diff = W.diff().fillna(W.iloc[0])
        trades = []
        for ts, row in diff.iterrows():
            for sym in W.columns:
                d = float(row[sym])
                if abs(d) < 1e-6: continue
                w_now = float(W.loc[ts, sym])
                price = float(prices.loc[ts, sym]) if not pd.isna(prices.loc[ts, sym]) else 0.0
                side = ("OPEN LONG" if w_now > 0 and d > 0 else
                        "OPEN SHORT" if w_now < 0 and d < 0 else
                        "CLOSE" if w_now == 0 else "RESIZE")
                trades.append({"t": ts.isoformat(), "sym": str(sym),
                               "side": side, "price": round(price, 2),
                               "weight": round(w_now, 3)})

        out = {"name": name, "family": "Ensemble",
               "metrics": {"return_pct": res["total_return_%"],
                            "sharpe": res["sharpe"],
                            "max_dd_pct": res["max_drawdown_%"],
                            "profit_factor": res.get("profit_factor"),
                            "n_trades": res["n_trades"],
                            "win_rate_pct": res["win_rate_%"],
                            "final_equity": res["final_equity"]},
               "equity": equity_pts, "trades": trades, "trade_count": len(trades)}
        (WEBAPP_DATA / f"{slug}.json").write_text(json.dumps(out, separators=(",", ":")))

    # Refresh manifest with ensembles included, ranked by Sharpe
    manifest = json.loads((WEBAPP_DATA / "manifest.json").read_text())
    existing = manifest["models"]
    for spec in top_specs:
        slug = spec["slug"]
        # remove if already there
        existing = [m for m in existing if m.get("slug") != slug]
        m_data = json.loads((WEBAPP_DATA / f"{slug}.json").read_text())
        existing.append({
            "slug": slug, "name": m_data["name"], "family": "Ensemble",
            "metrics": m_data["metrics"],
            "trade_count": m_data["trade_count"],
            "equity_points": len(m_data["equity"]),
        })
    existing.sort(key=lambda c: (c["family"] == "Benchmark", -c["metrics"]["sharpe"]))
    for i, c in enumerate(existing, 1):
        c["rank"] = i
    manifest["models"] = existing
    manifest["generated"] = pd.Timestamp.utcnow().isoformat()
    (WEBAPP_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  → exported {len(top_specs)} ensemble(s) to webapp/data/")

if __name__ == "__main__":
    main()
    # Also build & export top ensembles for the dashboard
    print()
    print("Exporting ensembles to webapp...")
    M1_w, prices = position_M1_transformer_lf_nlp()
    M2_w, _      = position_M2_lstm_v4()
    M3_w, _      = position_M3_crossmodal_r6()
    M4_w, _      = position_M4_sentlstm_r6()
    idx = M1_w.index
    members = {"M1": M1_w, "M2": M2_w.reindex(idx).fillna(0),
               "M3": M3_w.reindex(idx).fillna(0), "M4": M4_w.reindex(idx).fillna(0)}
    prices = prices.reindex(idx).ffill()

    TOP_SPECS = [
        {"gtag": "G4", "atag": "E3", "mems": ["M1","M2","M3","M4"],
         "name": "G4 Majority Ensemble (all 4 models)",
         "slug": "ensemble_g4_majority"},
        {"gtag": "G3", "atag": "E3", "mems": ["M1","M2","M3"],
         "name": "G3 Majority Ensemble (Tx + LSTM + CrossModal)",
         "slug": "ensemble_g3_majority"},
        {"gtag": "G3", "atag": "E2", "mems": ["M1","M2","M3"],
         "name": "G3 Sharpe-Weighted Ensemble (Tx + LSTM + CrossModal)",
         "slug": "ensemble_g3_sharpewtd"},
    ]
    export_ensembles_to_webapp(TOP_SPECS, members, prices)
