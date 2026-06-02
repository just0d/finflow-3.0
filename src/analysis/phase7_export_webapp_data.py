"""
phase7_export_webapp_data.py — emit per-model JSON for the dashboard.

For every strategy in phase6_backtester_v2 we re-run the backtest and write
{
    "metrics":     {...},
    "equity":      [{t, v}, ...],     # ALL bars, full year
    "trades":      [{t, sym, side, price, weight, equity}, ...],   # signal flips only
}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import phase6_backtester_v2 as bt   # noqa: E402

WEBAPP_DATA = ROOT / "webapp" / "data"
WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

# strategies + which CSV + which run config
STRATS = [
    ("Transformer + Late Fusion + NLP",  "fused_signals_transformer.csv",  "Late Fusion",  False, True),
    ("Transformer (raw, no NLP)",        "fused_signals_transformer.csv",  "No NLP",       True,  False),
    ("LSTM + Late Fusion + NLP",         "fused_signals_lstm.csv",         "Late Fusion",  False, True),
    ("LSTM (raw, no NLP)",               "fused_signals_lstm.csv",         "No NLP",       True,  False),
    ("CrossModalTransformer (deep)",     "fused_signals_crossmodal.csv",   "Deep Fusion",  False, True),
    ("SentimentLSTM (deep)",             "fused_signals_sentlstm.csv",     "Deep Fusion",  False, True),
    ("Buy & Hold (ex-NVDA)",             "fused_signals_buyandhold.csv",   "Benchmark",    False, False),
]

def run_one(name, fname, family, use_raw, nlp_gate):
    df = bt._load(ROOT / "outputs" / fname)
    if df.empty:
        return None
    if family == "Benchmark":
        res, equity = bt.buyandhold(df)
        weights = None
    else:
        res, equity = bt.backtest(name, df, allow_short=True,
                                  nlp_gate=nlp_gate, use_raw_signal=use_raw)
        # Reproduce the per-symbol weight series so we can extract trade events
        df2 = df.copy()
        if use_raw and "raw_pred" in df2.columns:
            df2["fused_signal"] = df2["raw_pred"]
        non_hold = df2.loc[df2["fused_signal"] != "HOLD", "confidence"]
        cf = float(non_hold.quantile(bt.CONF_QUANTILE)) if len(non_hold) else 0.5
        df2["pos_signed"] = df2.apply(
            lambda r: bt._signed_size(r, True, nlp_gate, cf), axis=1)
        pivot_close = df2.pivot_table(index="timestamp", columns="symbol",
                                       values="close", aggfunc="last").ffill()
        pivot_pos = df2.pivot_table(index="timestamp", columns="symbol",
                                     values="pos_signed", aggfunc="last").reindex(
                                         pivot_close.index).ffill().fillna(0)
        pos = pivot_pos.shift(1).fillna(0)
        gross = pos.abs().sum(axis=1).replace(0, np.nan)
        weights = pos.div(gross, axis=0).fillna(0)

    eq = equity.copy()
    if len(eq) > 1500:
        step = int(np.ceil(len(eq) / 1500))
        eq = eq.iloc[::step]
    equity_pts = [{"t": ts.isoformat(), "v": round(float(v), 2)}
                  for ts, v in eq.items()]

    # trade events (only when weights flip) 
    trades = []
    if weights is not None:
        diff = weights.diff().fillna(weights.iloc[0])
        for ts, row in diff.iterrows():
            for sym in weights.columns:
                d = float(row[sym])
                if abs(d) < 1e-6:
                    continue
                w_now = float(weights.loc[ts, sym])
                price = float(pivot_close.loc[ts, sym]) if not pd.isna(pivot_close.loc[ts, sym]) else 0.0
                if w_now > 0 and d > 0:
                    side = "OPEN LONG"
                elif w_now < 0 and d < 0:
                    side = "OPEN SHORT"
                elif w_now == 0:
                    side = "CLOSE"
                else:
                    side = "RESIZE"
                trades.append({
                    "t": ts.isoformat(),
                    "sym": str(sym),
                    "side": side,
                    "price": round(price, 2),
                    "weight": round(w_now, 3),
                })

    out = {
        "name": name,
        "family": family,
        "metrics": {
            "return_pct": res["total_return_%"],
            "sharpe": res["sharpe"],
            "max_dd_pct": res["max_drawdown_%"],
            "profit_factor": res.get("profit_factor"),
            "n_trades": res["n_trades"],
            "win_rate_pct": res["win_rate_%"],
            "final_equity": res["final_equity"],
        },
        "equity": equity_pts,
        "trades": trades,
        "trade_count": len(trades),
    }
    return out

def main():
    print(f"Exporting webapp data → {WEBAPP_DATA}")
    catalog = []
    for name, fname, family, use_raw, gate in STRATS:
        path = ROOT / "outputs" / fname
        if not path.exists():
            print(f"  [skip] {fname}")
            continue
        out = run_one(name, fname, family, use_raw, gate)
        if not out:
            continue
        slug = (name.lower()
                .replace(" + ", "_").replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace(",", "").replace("&", "and"))
        (WEBAPP_DATA / f"{slug}.json").write_text(json.dumps(out, separators=(",", ":")))
        catalog.append({
            "slug": slug,
            "name": name,
            "family": family,
            "metrics": out["metrics"],
            "trade_count": out["trade_count"],
            "equity_points": len(out["equity"]),
        })
        print(f"  {slug:50s} return={out['metrics']['return_pct']:+6.2f}% "
              f"sharpe={out['metrics']['sharpe']:+.2f} trades={out['trade_count']}")

    # rank by Sharpe (excluding benchmark gets a separate flag)
    catalog.sort(key=lambda c: (c["family"] == "Benchmark", -c["metrics"]["sharpe"]))
    for i, c in enumerate(catalog, 1):
        c["rank"] = i

    (WEBAPP_DATA / "manifest.json").write_text(json.dumps({
        "generated": pd.Timestamp.utcnow().isoformat(),
        "test_window": {"start": "2025-01-07", "end": "2026-01-01"},
        "universe": ["JPM", "SPY", "TSLA"],
        "excluded": ["NVDA"],
        "models": catalog,
    }, indent=2))
    print(f"\nManifest written. Top-3 by Sharpe:")
    for c in [x for x in catalog if x["family"] != "Benchmark"][:3]:
        m = c["metrics"]
        print(f"  #{c['rank']}  {c['name']:38s}  ret={m['return_pct']:+6.2f}%  "
              f"Sharpe={m['sharpe']:+.2f}  MDD={m['max_dd_pct']:+.2f}%")

if __name__ == "__main__":
    main()
