"""Robustness check — run the full meta-ensemble with NVDA INCLUDED.

We monkey-patch phase6_backtester_v2.EXCLUDE_SYMBOLS to be empty, then re-import
phase11 fresh so all member-position builders see the full 4-ticker universe.
Prints a side-by-side comparison vs the headline (ex-NVDA) numbers.
"""
from __future__ import annotations
import importlib, sys
import numpy as np
import pandas as pd

# Force the backtester to include NVDA
import phase6_backtester_v2 as bt
bt.EXCLUDE_SYMBOLS = set()   # ← was {"NVDA"}

# Fresh re-import of phase11 so its module-level constants see the patch
if "phase11_meta_ensemble" in sys.modules:
    del sys.modules["phase11_meta_ensemble"]
import phase11_meta_ensemble as p11

# Build member positions including NVDA
print("Building member positions with NVDA included ...")
M1_w, prices = p11.position_M1_transformer_lf_nlp()
M2_w, _      = p11.position_M2_lstm_v4()
M3_w, _      = p11.position_M3_crossmodal_r6()
M4_w, _      = p11.position_M4_sentlstm_r6()

idx = M1_w.index
members = {"M1": M1_w,
           "M2": M2_w.reindex(idx).fillna(0),
           "M3": M3_w.reindex(idx).fillna(0),
           "M4": M4_w.reindex(idx).fillna(0)}
prices = prices.reindex(idx).ffill()
print(f"Universe in backtest: {sorted(prices.columns.tolist())}")
print(f"Bars: {len(prices):,}")

# Run each member + the four ensemble aggregators
def run(name, W):
    gross = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(gross, axis=0).fillna(0)
    res, eq = p11.backtest_weights(W, prices, name)
    return res, eq

results = {}
for tag, m in members.items():
    res, _ = run(tag, m)
    results[tag] = res

# G2/G3/G4 majority ensembles
G2 = p11.agg_majority([members[m] for m in ("M1","M2")])
G3 = p11.agg_majority([members[m] for m in ("M1","M2","M3")])
G4 = p11.agg_majority([members[m] for m in ("M1","M2","M3","M4")])
results["G2 majority"], _ = run("G2 majority", G2)
results["G3 majority"], _ = run("G3 majority", G3)
results["G4 majority"], _ = run("G4 majority", G4)

# All four aggregators on the full 4-member set for completeness
G4_eq = p11.agg_equal([members[m] for m in ("M1","M2","M3","M4")])
G4_wtd = p11.agg_weighted([members[m] for m in ("M1","M2","M3","M4")],
                          [p11.SHARPES[m] for m in ("M1","M2","M3","M4")])
G4_cons = p11.agg_consensus([members[m] for m in ("M1","M2","M3","M4")])
results["G4 equal"],     _ = run("G4 equal",     G4_eq)
results["G4 sharpe-wtd"],_ = run("G4 sharpe-wtd",G4_wtd)
results["G4 consensus"], _ = run("G4 consensus", G4_cons)

# Buy & Hold WITH NVDA
_, eq_bh = bt.buyandhold(bt._load(p11.OUT / "fused_signals_buyandhold.csv"))
res_bh = p11._metrics if hasattr(p11, "_metrics") else bt._metrics
# Compute BH metrics directly
ret_bh = eq_bh.iloc[-1] / 100000 - 1
daily_bh = eq_bh.pct_change().dropna().groupby(eq_bh.pct_change().dropna().index.date).apply(lambda x: (1+x).prod()-1)
sharpe_bh = (daily_bh.mean()/daily_bh.std()) * np.sqrt(252) if daily_bh.std()>0 else 0
mdd_bh = ((eq_bh / eq_bh.cummax()) - 1).min()
print(f"\nBuy & Hold (4 tickers incl. NVDA): ret={ret_bh*100:+.2f}%  Sharpe={sharpe_bh:+.2f}  MDD={mdd_bh*100:+.2f}%")

# Print comparison table
print()
print(f"{'Strategy':<25s}  {'Return%':>9s}  {'Sharpe':>7s}  {'MaxDD%':>8s}  {'PF':>5s}  {'Trades':>7s}")
for tag, res in results.items():
    pf = res.get('profit_factor', None)
    pf_s = f"{pf:.2f}" if pf is not None else "  —"
    print(f"{tag:<25s}  {res['total_return_%']:+9.2f}  {res['sharpe']:+7.2f}  "
          f"{res['max_drawdown_%']:+8.2f}  {pf_s:>5s}  {res['n_trades']:>7d}")

# Side-by-side vs headline (ex-NVDA) numbers
print()
print("Headline (ex-NVDA, from poster)  vs  Robustness check (with NVDA)")
poster = {
    "M1":             ("+69.22 %", "1.66", "-22.21 %"),
    "M2":             ("+4.94 %",  "0.30", "-33.69 %"),
    "M3":             ("+20.58 %", "0.81", "-14.67 %"),
    "M4":             ("-15.86 %", "-0.47","-24.23 %"),  # SentimentLSTM (R6) from earlier
    "G3 majority":    ("+43.83 %", "1.51", "-14.65 %"),
    "G4 majority":    ("+30.76 %", "1.71", "-7.37 %"),
}
for tag, (ret_e, shp_e, mdd_e) in poster.items():
    if tag in results:
        r = results[tag]
        print(f"{tag:<14s}  ex-NVDA: ret={ret_e:>8s}  Sharpe={shp_e:>5s}  MDD={mdd_e:>8s}")
        print(f"{'':<14s}  w/ NVDA: ret={r['total_return_%']:+8.2f}%  "
              f"Sharpe={r['sharpe']:+5.2f}  MDD={r['max_drawdown_%']:+8.2f}%")
        print()
