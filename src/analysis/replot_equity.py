"""Poster-quality equity + drawdown plot for FinFlow 3.0.

Only plots the 6 strategies from the Results table:
    G4 Majority Ensemble
    Transformer + LF + NLP (M1)
    G3 Majority Ensemble
    Buy & Hold ex-NVDA
    LSTM + V4 conditional gate
    CrossModal × Tx-agreement (R6)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import phase11_meta_ensemble as p11
import phase6_backtester_v2 as bt

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "final_results" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# Build data
print("Building member positions ...")

M1_w, prices = p11.position_M1_transformer_lf_nlp()
M2_w, _      = p11.position_M2_lstm_v4()
M3_w, _      = p11.position_M3_crossmodal_r6()
M4_w, _      = p11.position_M4_sentlstm_r6()

idx = M1_w.index

members = {
    "M1": M1_w,
    "M2": M2_w.reindex(idx).fillna(0),
    "M3": M3_w.reindex(idx).fillna(0),
    "M4": M4_w.reindex(idx).fillna(0),
}

prices = prices.reindex(idx).ffill()

# G4 majority
W_g4 = p11.agg_majority([members[m] for m in ("M1", "M2", "M3", "M4")])
gross = W_g4.abs().sum(axis=1).replace(0, np.nan)
W_g4 = W_g4.div(gross, axis=0).fillna(0)
_, eq_g4 = p11.backtest_weights(W_g4, prices, "G4 Majority")

# G3 majority
W_g3 = p11.agg_majority([members[m] for m in ("M1", "M2", "M3")])
gross = W_g3.abs().sum(axis=1).replace(0, np.nan)
W_g3 = W_g3.div(gross, axis=0).fillna(0)
_, eq_g3 = p11.backtest_weights(W_g3, prices, "G3 Majority")

# Individual members
_, eq_M1 = p11.backtest_weights(members["M1"], prices, "M1")
_, eq_M2 = p11.backtest_weights(members["M2"], prices, "M2")
_, eq_M3 = p11.backtest_weights(members["M3"], prices, "M3")

# Buy & Hold
_, eq_bh = bt.buyandhold(bt._load(ROOT / "outputs" / "fused_signals_buyandhold.csv"))

# Helpers
def drawdown(eq):
    return eq / eq.cummax() - 1

def end_label(ax, series, text, color, dy=0):
    x = series.index[-1]
    y = series.iloc[-1]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(12, dy),
        textcoords="offset points",
        color=color,
        fontsize=14,
        fontweight="bold",
        va="center",
        ha="left",
    )

colors = {
    "g4": "#FBBF24",      # gold
    "m1": "#F97316",      # orange
    "g3": "#3B82F6",      # blue
    "bh": "#64748B",      # grey/slate
    "m2": "#0E868B",      # teal
    "m3": "#7C3AED",      # purple
    "navy": "#1F4E79",
    "text": "#334155",
    "grid": "#CBD5E1",
}

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(15, 8.6),
    gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.08},
    sharex=True
)

fig.patch.set_facecolor("white")
ax1.set_facecolor("white")
ax2.set_facecolor("white")

# Equity curves
# Background / supporting models
ax1.plot(eq_M2.index, eq_M2.values, lw=1.2, color=colors["m2"], alpha=0.45,
         label="LSTM + V4 conditional gate")
ax1.plot(eq_M3.index, eq_M3.values, lw=1.2, color=colors["m3"], alpha=0.45,
         label="CrossModal × Tx-agreement")
ax1.plot(eq_g3.index, eq_g3.values, lw=1.6, color=colors["g3"], alpha=0.65,
         label="G3 Majority Ensemble")
ax1.plot(eq_bh.index, eq_bh.values, lw=1.5, color=colors["bh"], alpha=0.85,
         ls="--", label="Buy & Hold ex-NVDA")

# Main models
ax1.plot(eq_M1.index, eq_M1.values, lw=2.4, color=colors["m1"],
         label="Transformer + LF + NLP")
ax1.plot(eq_g4.index, eq_g4.values, lw=3.2, color=colors["g4"],
         label="G4 Majority Ensemble", solid_capstyle="round")

ax1.axhline(100000, color="#94A3B8", lw=1.1, ls=":")
ax1.text(eq_g4.index[5], 101500, "$100k start", color="#475569", fontsize=12)

end_label(ax1, eq_g4, "G4  +30.76% | Sharpe 1.71", colors["g4"], dy=8)
end_label(ax1, eq_M1, "M1  +69.22%", colors["m1"], dy=0)
end_label(ax1, eq_bh, "Buy & Hold  +24.57%", colors["bh"], dy=-10)

# Drawdown curves
ax2.plot(drawdown(eq_g4).index, drawdown(eq_g4).values * 100,
         lw=2.6, color=colors["g4"], label="G4")
ax2.plot(drawdown(eq_M1).index, drawdown(eq_M1).values * 100,
         lw=1.8, color=colors["m1"], alpha=0.85, label="M1")
ax2.plot(drawdown(eq_bh).index, drawdown(eq_bh).values * 100,
         lw=1.4, color=colors["bh"], alpha=0.85, ls="--", label="Buy & Hold")

ax2.axhline(0, color="#94A3B8", lw=1.0)
ax2.axhline(-7.37, color=colors["g4"], lw=1.2, ls=":", alpha=0.9)
ax2.text(eq_g4.index[5], -5.8, "G4 MaxDD −7.37%", color="#B45309",
         fontsize=13, fontweight="bold")

# Titles and labels
ax1.set_title(
    "G4 Majority Ensemble improves risk-adjusted performance on the 2025 hold-out",
    fontsize=20,
    fontweight="bold",
    color=colors["navy"],
    pad=14,
)

ax1.set_ylabel("Portfolio equity", fontsize=14, color=colors["text"], fontweight="bold")
ax2.set_ylabel("Drawdown",         fontsize=14, color=colors["text"], fontweight="bold")
ax2.set_xlabel("2025 test window", fontsize=14, color=colors["text"], fontweight="bold")

ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# X-axis formatting
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# Grid and spines
for ax in (ax1, ax2):
    ax.grid(True, color=colors["grid"], alpha=0.4, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(colors=colors["text"], labelsize=12)

# Legend for all six strategies — bigger, more readable
leg = ax1.legend(
    loc="upper left",
    fontsize=12,
    frameon=True,
    facecolor="white",
    edgecolor="#94A3B8",
    framealpha=0.96,
    ncol=2,
    handlelength=2.2,
    columnspacing=1.4,
    labelspacing=0.6,
)

for txt in leg.get_texts():
    txt.set_color("#1E293B")
    txt.set_fontweight("medium")

# Caption-style note — larger so it reads at poster distance
fig.text(
    0.5, 0.02,
    "Equity curves show return; drawdown panel shows capital preservation. "
    "G4 sacrifices raw return versus M1 but sharply reduces peak-to-trough risk.",
    ha="center",
    fontsize=13,
    color="#334155",
    style="italic",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out = CHARTS / "v3_ensemble_equity_poster_1.png"
plt.savefig(out, dpi=220, bbox_inches="tight")
plt.close()

print(f"saved {out} ({out.stat().st_size:,} bytes)")