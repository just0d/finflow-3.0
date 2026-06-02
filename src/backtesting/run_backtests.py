"""
FinFlow 2.0 — Backtest adapter
Reuses the FinFlow 1.0 phase5_backtester *without modification*.
Imports its `Portfolio`, `run_simulation`, `make_buyandhold`,
`load_price_data` and runs every available signal file in /outputs/
through the same engine.

"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import the EXISTING backtester engine — DO NOT modify it
import phase5_backtester as bt

OUT = ROOT / "outputs"

# Mapping: signal CSV to strategy display name
STRATEGIES_DEEP = {
    "fused_signals_crossmodal.csv": "CrossModalTransformer",
    "fused_signals_sentlstm.csv":   "SentimentLSTM",
}

STRATEGIES_BASE = {
    "fused_signals_lstm.csv":         "LSTM_NLP",
    "fused_signals_transformer.csv":  "Transformer_NLP",
    "fused_signals_buyandhold.csv":   "Buy_Hold_SPY",
}

def main():
    print("  FinFlow 2.0 — Backtest Adapter (uses unchanged phase5 engine)")

    price_df = bt.load_price_data()
    print(f"  Price data: {len(price_df):,} rows")

    portfolios = []
    metrics    = []

    # Run baselines (existing FinFlow 1.0 signals)
    for fname, strat in STRATEGIES_BASE.items():
        df = bt.load_signal_file(fname)
        if df is None:
            print(f"  [SKIP] {fname} not found")
            continue
        if "Buy_Hold" in strat:
            port = bt.make_buyandhold(price_df); port.name = strat
        else:
            port = bt.run_simulation(strat, df, price_df)
        portfolios.append(port)
        metrics.append(port.compute_metrics())
        print(f"  {strat:24s} {len(port.trades):4d} trades")

    # Add raw (no-late-fusion) variants for LSTM / Transformer
    for fname, strat in STRATEGIES_BASE.items():
        if "Buy_Hold" in strat: continue
        df = bt.load_signal_file(fname)
        if df is None: continue
        raw_df = df.copy()
        raw_df["fused_signal"] = raw_df.get("raw_pred", raw_df["fused_signal"])
        raw_strat = strat.replace("_NLP", "_Raw")
        port = bt.run_simulation(raw_strat, raw_df, price_df)
        portfolios.append(port)
        metrics.append(port.compute_metrics())
        print(f"  {raw_strat:24s} {len(port.trades):4d} trades")

    # Run NEW models (FinFlow 2.0)
    for fname, strat in STRATEGIES_DEEP.items():
        df = bt.load_signal_file(fname)
        if df is None:
            print(f"  [SKIP] {fname} not found")
            continue
        port = bt.run_simulation(strat, df, price_df)
        portfolios.append(port)
        metrics.append(port.compute_metrics())
        print(f"  {strat:24s} {len(port.trades):4d} trades")

    # Print + save
    if metrics:
        bt.print_race_results(metrics)
        df_summary = pd.DataFrame(metrics)
        out = OUT / "finflow2_backtest_summary.csv"
        df_summary.to_csv(out, index=False)
        print(f"\n  Saved → {out}")

    return portfolios, metrics

if __name__ == "__main__":
    main()
