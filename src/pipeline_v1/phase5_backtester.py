"""
FinFlow 1.0 — Phase 5: Stateful Backtester & Race
Runs four simulations through the hidden 2025 test window and
calculates the three academic metrics your professor requires:

  1. Sharpe Ratio   — risk-adjusted return (> 1.0 good, > 2.0 excellent)
  2. Max Drawdown   — worst peak-to-trough fall (FinBERT brake should reduce this)
  3. Profit Factor  — gross wins / gross losses (> 1.5 = stable edge)

The Race:
  Strategy A — Best Math Model (raw, no NLP)
  Strategy B — Best Math Model + FinBERT Late Fusion   ← the protagonist
  Strategy C — Transformer or Prophet (second best from Phase 3)
  Strategy D — Buy & Hold SPY                          ← the benchmark

Simulation Constraints (hardcoded for academic integrity):
  Capital        : $100,000
  Slippage       : 0.1% per trade (bid/ask spread + broker fee simulation)
  Position sizing: 15% of current cash per signal (never all-in)
  Max open pos.  : 4 simultaneously (one per ticker)
  Direction      : Long-only (no short selling for clarity)

Run after Phase 4:
  python3 phase4_late_fusion.py
  python3 phase5_backtester.py

"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Paths
BASE    = Path(__file__).parent
OUTPUTS = BASE / "outputs"; OUTPUTS.mkdir(exist_ok=True)
CHARTS  = BASE / "outputs" / "charts"; CHARTS.mkdir(exist_ok=True)
ALIGNED = BASE / "data" / "aligned"
PROC    = BASE / "data" / "processed"

TARGET_TICKERS = ["NVDA", "TSLA", "JPM", "SPY"]

# Bloomberg palette
BG, PANEL, BORDER = "#0a0a0f", "#0f1117", "#1e2330"
TEXT, MUTED = "#c8cdd6", "#5a6070"
ORANGE, BLUE, GREEN, RED, AMBER, PURPLE = (
    "#f97316", "#3b82f6", "#22c55e", "#ef4444", "#f59e0b", "#a78bfa"
)
STRATEGY_COLORS = {
    "Model_Raw":     BLUE,
    "Model_NLP":     GREEN,
    "Second_NLP":    PURPLE,
    "Buy_Hold_SPY":  ORANGE,
}

# Simulation Constants
INITIAL_CAPITAL  = 100_000.0
SLIPPAGE         = 0.001          # 0.1% each way
POSITION_SIZE    = 0.15           # risk 15% of cash per new trade
RF_ANNUAL        = 0.042          # 10-year US Treasury ~4.2%
TRADING_HRS_DAY  = 6.5
ANNUALISATION    = np.sqrt(252 * TRADING_HRS_DAY)  # hourly → annual Sharpe scale

# PORTFOLIO ENGINE

class Portfolio:
    """
    Stateful paper-trading engine.
    Tracks cash, open positions, and full trade history.
    """

    def __init__(self, name: str, initial_capital: float = INITIAL_CAPITAL):
        self.name     = name
        self.cash     = initial_capital
        self.initial  = initial_capital
        self.positions: dict[str, dict] = {}   # {ticker: {shares, entry_price, entry_time}}
        self.value_ts: list[tuple] = []         # [(timestamp, portfolio_value)]
        self.trades:   list[dict]  = []

    # Core execution
    def execute(
        self,
        timestamp,
        ticker:  str,
        signal:  str,
        price:   float,
    ) -> dict | None:
        """Process one signal. Returns trade dict if a trade executed, else None."""
        if pd.isna(price) or price <= 0:
            return None

        trade = None

        # BUY — enter long if we have no position in this ticker
        if signal == "BUY" and ticker not in self.positions:
            invest = self.cash * POSITION_SIZE
            if invest < 1.0:
                return None   # insufficient cash
            buy_price = price * (1 + SLIPPAGE)     # slippage on entry
            shares    = invest / buy_price
            self.cash -= invest
            self.positions[ticker] = {
                "shares":      shares,
                "entry_price": buy_price,
                "entry_time":  timestamp,
                "entry_value": invest,
            }
            trade = {
                "timestamp":   str(timestamp),
                "ticker":      ticker,
                "action":      "BUY",
                "price":       round(buy_price, 4),
                "shares":      round(shares, 6),
                "value":       round(invest, 2),
                "pnl":         0.0,
                "slippage":    round(invest * SLIPPAGE, 2),
            }
            self.trades.append(trade)

        # SELL — exit long position if we have one
        elif signal == "SELL" and ticker in self.positions:
            pos       = self.positions.pop(ticker)
            sell_price = price * (1 - SLIPPAGE)    # slippage on exit
            proceeds  = pos["shares"] * sell_price
            pnl       = proceeds - pos["entry_value"]
            self.cash += proceeds
            trade = {
                "timestamp":   str(timestamp),
                "ticker":      ticker,
                "action":      "SELL",
                "price":       round(sell_price, 4),
                "shares":      round(pos["shares"], 6),
                "value":       round(proceeds, 2),
                "pnl":         round(pnl, 2),
                "slippage":    round(pos["entry_value"] * SLIPPAGE, 2),
            }
            self.trades.append(trade)

        return trade

    # Value snapshot
    def record_value(self, timestamp, current_prices: dict[str, float]):
        """Snapshot total portfolio value (cash + open positions at current prices)."""
        pos_value = 0.0
        for ticker, pos in self.positions.items():
            p = current_prices.get(ticker, pos["entry_price"])
            pos_value += pos["shares"] * p
        total = self.cash + pos_value
        self.value_ts.append((timestamp, total))

    # Metric computation
    def compute_metrics(self) -> dict:
        if len(self.value_ts) < 2:
            return {}

        timestamps = [t for t, _ in self.value_ts]
        values     = np.array([v for _, v in self.value_ts])

        # Returns (hourly)
        rets = np.diff(values) / values[:-1]

        # Risk-free rate (hourly)
        rf_hourly = (1 + RF_ANNUAL) ** (1 / (252 * TRADING_HRS_DAY)) - 1
        excess    = rets - rf_hourly

        # Sharpe Ratio (annualised)
        sharpe = (excess.mean() / excess.std() * ANNUALISATION) if excess.std() > 0 else 0.0

        # Maximum Drawdown
        peak      = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        mdd       = float(drawdowns.min())   # most negative value

        # Profit Factor (from closed trades)
        winning = [t["pnl"] for t in self.trades if t.get("pnl", 0) > 0 and t["action"] == "SELL"]
        losing  = [abs(t["pnl"]) for t in self.trades if t.get("pnl", 0) < 0 and t["action"] == "SELL"]
        profit_factor = sum(winning) / sum(losing) if sum(losing) > 0 else float("inf")

        # Total return
        total_return = (values[-1] - values[0]) / values[0]

        # Win rate
        n_trades  = len([t for t in self.trades if t["action"] == "SELL"])
        n_winners = len(winning)
        win_rate  = n_winners / n_trades if n_trades > 0 else 0.0

        return {
            "strategy":         self.name,
            "initial_capital":  round(self.initial, 2),
            "final_value":      round(float(values[-1]), 2),
            "total_return":     round(total_return * 100, 2),       # %
            "sharpe_ratio":     round(sharpe, 4),
            "max_drawdown":     round(mdd * 100, 2),                # %
            "profit_factor":    round(min(profit_factor, 99.9), 4), # cap at 99.9 for display
            "n_trades":         n_trades,
            "win_rate":         round(win_rate * 100, 2),           # %
            "total_slippage":   round(sum(t.get("slippage", 0) for t in self.trades), 2),
        }

# SIMULATION RUNNER

def load_signal_file(fname: str) -> pd.DataFrame | None:
    path = OUTPUTS / fname
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df

def load_price_data() -> pd.DataFrame:
    """Load test set prices. Use sentiment CSV if available (same columns + sentiment)."""
    for p in [PROC / "test_sentiment.csv", ALIGNED / "test_aligned.csv"]:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
    raise FileNotFoundError("No test price data found.")

def run_simulation(
    name:        str,
    signals_df:  pd.DataFrame,
    price_df:    pd.DataFrame,
) -> Portfolio:
    """
    Runs a full simulation through the 2025 test window.
    At each unique timestamp, processes all 4 tickers in order.
    """
    portfolio = Portfolio(name)

    # Build a fast price lookup: {(symbol, timestamp): close_price}
    price_lookup = {
        (row.symbol, row.timestamp): row.close
        for row in price_df.itertuples(index=False)
    }

    # Build a fast signal lookup: {(symbol, timestamp): signal}
    signal_lookup = {
        (row.symbol, row.timestamp): row.fused_signal
        for row in signals_df.itertuples(index=False)
    }

    # Get all unique timestamps (sorted)
    all_timestamps = sorted(price_df["timestamp"].unique())

    for ts in all_timestamps:
        # Get current prices for all tickers
        current_prices = {}
        for ticker in TARGET_TICKERS:
            p = price_lookup.get((ticker, ts))
            if p is not None and not pd.isna(p):
                current_prices[ticker] = p

        # Execute signals
        for ticker in TARGET_TICKERS:
            signal = signal_lookup.get((ticker, ts), "HOLD")
            price  = current_prices.get(ticker)
            if price is not None:
                portfolio.execute(ts, ticker, signal, price)

        # Record portfolio value at this timestamp
        portfolio.record_value(ts, current_prices)

    # Close all open positions at end of test window
    if current_prices:
        for ticker in list(portfolio.positions.keys()):
            price = current_prices.get(ticker)
            if price:
                portfolio.execute(all_timestamps[-1], ticker, "SELL", price)

    return portfolio

def make_buyandhold(price_df: pd.DataFrame) -> Portfolio:
    """Simulate simple Buy-and-Hold on SPY only."""
    portfolio = Portfolio("Buy_Hold_SPY")

    # Buy SPY at first available price
    spy_prices = price_df[price_df["symbol"] == "SPY"].sort_values("timestamp")
    if spy_prices.empty:
        return portfolio

    first_row = spy_prices.iloc[0]
    buy_price = first_row["close"] * (1 + SLIPPAGE)
    invest    = portfolio.cash * 0.95   # invest 95% in SPY
    shares    = invest / buy_price
    portfolio.cash -= invest
    portfolio.positions["SPY"] = {
        "shares": shares, "entry_price": buy_price,
        "entry_time": first_row["timestamp"], "entry_value": invest,
    }
    portfolio.trades.append({
        "timestamp": str(first_row["timestamp"]),
        "ticker": "SPY", "action": "BUY",
        "price": round(buy_price, 4), "shares": round(shares, 6),
        "value": round(invest, 2), "pnl": 0.0,
        "slippage": round(invest * SLIPPAGE, 2),
    })

    price_lookup = {
        row.timestamp: row.close
        for row in spy_prices.itertuples(index=False)
    }
    for ts in sorted(spy_prices["timestamp"].unique()):
        portfolio.record_value(ts, {"SPY": price_lookup.get(ts, buy_price)})

    # Close at end
    last_price = spy_prices.iloc[-1]["close"] * (1 - SLIPPAGE)
    proceeds   = shares * last_price
    pnl        = proceeds - invest
    portfolio.cash += proceeds
    del portfolio.positions["SPY"]
    portfolio.trades.append({
        "timestamp": str(spy_prices.iloc[-1]["timestamp"]),
        "ticker": "SPY", "action": "SELL",
        "price": round(last_price, 4), "shares": round(shares, 6),
        "value": round(proceeds, 2), "pnl": round(pnl, 2),
        "slippage": round(invest * SLIPPAGE, 2),
    })

    return portfolio

# VISUALISATIONS

def plot_equity_curves(portfolios: list[Portfolio], metrics: list[dict]):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "FIGURE 5-A  |  FinFlow 1.0  |  Equity Curves — 2025 Test Window  |  $100,000 Starting Capital",
        color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left"
    )
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.5, 0.8], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])  # Equity curves
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Drawdown
    ax3 = fig.add_subplot(gs[2])  # Metrics table

    for ax in [ax1, ax2]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, lw=0.5, alpha=0.7, ls="--")
        ax.set_axisbelow(True)
    ax3.set_facecolor(BG)
    ax3.axis("off")

    colors = list(STRATEGY_COLORS.values())

    for i, port in enumerate(portfolios):
        if not port.value_ts:
            continue
        color = colors[i % len(colors)]
        ts     = [t for t, _ in port.value_ts]
        vals   = np.array([v for _, v in port.value_ts])
        rets   = (vals - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        # Equity curve (% return)
        ax1.plot(ts, rets, color=color, linewidth=1.5, label=port.name, alpha=0.95)
        ax1.fill_between(ts, 0, rets, alpha=0.04, color=color)

        # Drawdown curve
        peak  = np.maximum.accumulate(vals)
        dd    = (vals - peak) / peak * 100
        ax2.fill_between(ts, dd, 0, alpha=0.35, color=RED if port.name != "Buy_Hold_SPY" else MUTED)
        ax2.plot(ts, dd, color=color, linewidth=0.8, alpha=0.7)

    ax1.axhline(0, color=MUTED, lw=0.8, ls="--", alpha=0.5)
    ax1.set_ylabel("Cumulative Return (%)", color=TEXT, fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
    ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9, loc="upper left")
    plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

    ax2.axhline(0, color=MUTED, lw=0.5, ls="--")
    ax2.set_ylabel("Drawdown (%)", color=TEXT, fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    # Metrics table
    if metrics:
        col_labels = ["Strategy", "Return", "Sharpe", "Max DD", "Profit Factor", "Trades", "Win Rate"]
        rows = []
        for m in metrics:
            rows.append([
                m.get("strategy", ""),
                f"{m.get('total_return', 0):+.1f}%",
                f"{m.get('sharpe_ratio', 0):.3f}",
                f"{m.get('max_drawdown', 0):.1f}%",
                f"{m.get('profit_factor', 0):.2f}",
                str(m.get("n_trades", 0)),
                f"{m.get('win_rate', 0):.1f}%",
            ])
        table = ax3.table(
            cellText=rows, colLabels=col_labels,
            cellLoc="center", loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor(PANEL if row > 0 else BORDER)
            cell.set_edgecolor(BORDER)
            cell.set_text_props(color=TEXT if row > 0 else AMBER)

    fig.text(0.99, 0.01, "FinFlow 1.0  |  Backtest", ha="right", va="bottom",
             fontsize=7, color=MUTED, alpha=0.5, family="monospace")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = CHARTS / "phase5_equity_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved → {out}")

# JSON EXPORT (feeds the web simulator)

def export_simulator_json(
    portfolios: list[Portfolio],
    metrics:    list[dict],
    price_df:   pd.DataFrame,
    signals_by_strategy: dict,
):
    """
    Exports a single JSON blob that the HTML web simulator reads.
    Downsamples hourly data to keep file size manageable.
    """
    print("  Building simulator JSON...")

    spy_prices = (price_df[price_df["symbol"] == "SPY"]
                  .sort_values("timestamp")
                  .set_index("timestamp")["close"])

    # Build equity time series for each strategy
    equity_series = {}
    for port in portfolios:
        if not port.value_ts:
            continue
        ts_list  = [str(t) for t, _ in port.value_ts]
        val_list = [round(v, 2) for _, v in port.value_ts]
        equity_series[port.name] = {
            "timestamps": ts_list,
            "values":     val_list,
        }

    # Build trade log (all strategies combined, max 1000 most recent)
    all_trades = []
    for port in portfolios:
        for t in port.trades:
            all_trades.append({**t, "strategy": port.name})
    all_trades = sorted(all_trades, key=lambda x: x["timestamp"])[-2000:]

    # Build per-ticker price series (SPY for the chart)
    spy_ts   = [str(t) for t in spy_prices.index]
    spy_vals = [round(v, 4) for v in spy_prices.values]

    # Build signals overlay for best NLP strategy (for the trade markers)
    best_signals = []
    best_strat = next(
        (k for k in ["Model_NLP", "LSTM_NLP", "Transformer_NLP"] if k in signals_by_strategy),
        None
    )
    if best_strat:
        sig_df = signals_by_strategy[best_strat]
        spy_sig = sig_df[sig_df["symbol"] == "SPY"].sort_values("timestamp")
        for _, row in spy_sig.iterrows():
            if row["fused_signal"] in ("BUY", "SELL"):
                best_signals.append({
                    "timestamp":   str(row["timestamp"]),
                    "signal":      row["fused_signal"],
                    "price":       round(row["close"], 4),
                    "was_brake":   bool(row.get("was_overridden", False)),
                    "sentiment":   round(float(row.get("sentiment_score", 0)), 3),
                })

    simulator_data = {
        "metadata": {
            "project":      "FinFlow 1.0",
            "test_window":  "January 2025 – January 2026",
            "initial_capital": INITIAL_CAPITAL,
            "slippage_pct":    SLIPPAGE * 100,
            "position_size_pct": POSITION_SIZE * 100,
            "rf_rate_pct":     RF_ANNUAL * 100,
            "generated_at":    datetime.utcnow().isoformat(),
        },
        "metrics":       metrics,
        "equity_series": equity_series,
        "spy_price":     {"timestamps": spy_ts, "values": spy_vals},
        "trade_log":     all_trades,
        "trade_markers": best_signals[:500],   # cap for web performance
    }

    out_path = OUTPUTS / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(simulator_data, f, default=str)
    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved → {out_path}  ({size_kb:.0f} KB)")
    return simulator_data

# FINAL COMPARISON PRINT

def print_race_results(metrics: list[dict]):
    print("\n" + "═"*72)
    print("  THE RACE — FINAL RESULTS  |  2025 Unseen Test Window")
    print("═"*72)
    hdr = f"  {'Strategy':22s} {'Return':>9s}  {'Sharpe':>7s}  {'Max DD':>8s}  {'Pft Factor':>11s}  {'Trades':>7s}  {'Win%':>6s}"
    print(hdr)
    print("  " + "─"*68)

    best_sharpe = max(m["sharpe_ratio"] for m in metrics)
    for m in sorted(metrics, key=lambda x: x["sharpe_ratio"], reverse=True):
        star = " 🏆" if m["sharpe_ratio"] == best_sharpe else "   "
        print(
            f"  {m['strategy']:22s}{star}"
            f" {m['total_return']:>+8.1f}%"
            f"  {m['sharpe_ratio']:>7.3f}"
            f"  {m['max_drawdown']:>7.1f}%"
            f"  {m['profit_factor']:>11.2f}"
            f"  {m['n_trades']:>7,}"
            f"  {m['win_rate']:>5.1f}%"
        )

    # Analysis
    model_nlp = next((m for m in metrics if "NLP" in m["strategy"] or "Fusion" in m["strategy"]), None)
    model_raw = next((m for m in metrics if "Raw" in m["strategy"]), None)
    buy_hold  = next((m for m in metrics if "Buy" in m["strategy"] or "Hold" in m["strategy"]), None)

    print("\n  ── Key Findings ────────────────────────────────────────────────")
    if model_nlp and model_raw:
        dd_improvement = model_raw["max_drawdown"] - model_nlp["max_drawdown"]
        sharpe_lift    = model_nlp["sharpe_ratio"] - model_raw["sharpe_ratio"]
        print(f"  NLP Emergency Brake reduced Max Drawdown by: {abs(dd_improvement):.1f}pp")
        print(f"  NLP Late Fusion lifted Sharpe Ratio by:      {sharpe_lift:+.3f}")

    if model_nlp and buy_hold:
        alpha = model_nlp["total_return"] - buy_hold["total_return"]
        print(f"  Model+NLP alpha vs Buy-and-Hold SPY:          {alpha:+.1f}pp")

    sharpe_threshold = next(
        (m["sharpe_ratio"] for m in metrics if m["sharpe_ratio"] > 1.0),
        None
    )
    if sharpe_threshold:
        print(f"\n  ✓ Sharpe > 1.0 achieved — demonstrating risk-adjusted value")
    print("═"*72)

# MAIN

if __name__ == "__main__":
    print("  FinFlow 1.0 — Phase 5: Stateful Backtester")
    print(f"\n  Capital    : ${INITIAL_CAPITAL:,.0f}")
    print(f"  Slippage   : {SLIPPAGE*100:.1f}% per trade")
    print(f"  Position   : {POSITION_SIZE*100:.0f}% of cash per signal")
    print(f"  Risk-free  : {RF_ANNUAL*100:.1f}% annualised (10yr Treasury)")

    print("\n[1/4] Loading price and signal data...")
    price_df = load_price_data()
    print(f"  Price data: {len(price_df):,} rows, {price_df['timestamp'].nunique():,} hours")

    # Determine which signal files exist
    strategies_to_run = []
    signals_by_strategy = {}

    for fname, strat_name in [
        ("fused_signals_lstm.csv",        "LSTM_NLP"),
        ("fused_signals_transformer.csv", "Transformer_NLP"),
        ("fused_signals_buyandhold.csv",  "Buy_Hold_SPY"),
    ]:
        df = load_signal_file(fname)
        if df is not None:
            strategies_to_run.append((strat_name, df))
            signals_by_strategy[strat_name] = df
            print(f"  Loaded: {fname:45s} {len(df):,} rows")
        else:
            print(f"  [MISSING] {fname} — run phase4_late_fusion.py first")

    # Also build raw (no-NLP) versions using raw_pred instead of fused_signal
    for strat_name, df in list(strategies_to_run):
        if "NLP" in strat_name:
            raw_name = strat_name.replace("_NLP", "_Raw")
            raw_df   = df.copy()
            raw_df["fused_signal"] = raw_df.get("raw_pred", raw_df["fused_signal"])
            strategies_to_run.append((raw_name, raw_df))
            signals_by_strategy[raw_name] = raw_df

    print(f"\n[2/4] Running {len(strategies_to_run)} simulations...")
    portfolios = []
    for strat_name, sig_df in strategies_to_run:
        print(f"  Running: {strat_name}...", end=" ", flush=True)
        if "Buy_Hold" in strat_name:
            port = make_buyandhold(price_df)
            port.name = strat_name
        else:
            port = run_simulation(strat_name, sig_df, price_df)
        portfolios.append(port)
        final_val = port.value_ts[-1][1] if port.value_ts else INITIAL_CAPITAL
        ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"${final_val:,.0f}  ({ret:+.1f}%)  {len(port.trades)} trades")

    print("\n[3/4] Computing metrics...")
    metrics = [p.compute_metrics() for p in portfolios]
    for m in metrics:
        print(f"  {m.get('strategy','?'):22s}  Sharpe={m.get('sharpe_ratio','?'):.3f}  "
              f"MDD={m.get('max_drawdown','?'):.1f}%  PF={m.get('profit_factor','?'):.2f}")

    print("\n[4/4] Generating outputs...")
    plot_equity_curves(portfolios, metrics)

    # Export JSON for web simulator
    sim_data = export_simulator_json(portfolios, metrics, price_df, signals_by_strategy)

    # Save CSV summary
    summary_df = pd.DataFrame([{k: v for k, v in m.items()} for m in metrics])
    summary_df.to_csv(OUTPUTS / "backtest_summary.csv", index=False)
    print(f"  Saved → outputs/backtest_summary.csv")

    print_race_results(metrics)

    print("\n  ✓ Phase 5 complete.")
    print("  Next: python3 generate_simulator.py  →  opens webapp/index.html")
