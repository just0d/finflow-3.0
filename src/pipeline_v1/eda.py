"""
Exploratory Data Analysis: Generates the 9 primary charts and visualizations 
used in the final research report.

Figures:
  01_price_timeseries.png       — Close price over time per ticker
  02_news_volume_timeline.png   — Weekly news volume + train/test split line
  03_modality_imbalance.png     — Price rows vs news mentions per ticker
  04_returns_distribution.png   — Hourly returns distribution per ticker
  05_correlation_matrix.png     — Technical feature correlation matrix (SPY)
  06_finbert_token_dist.png     — FinBERT input word count distribution
  07_news_coverage_rate.png     — % of market hours covered by news per ticker
  08_volume_heatmap.png         — Volume by hour-of-day × day-of-week
  09_technical_dashboard.png    — NVDA: price + RSI + MACD + Volume (4-panel)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import ast

BASE    = Path(__file__).parent
ALIGNED = BASE / "data" / "aligned"
RAW     = BASE / "data" / "raw"
CHARTS  = BASE / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

BG        = "#0a0a0f" 
PANEL     = "#0f1117" 
BORDER    = "#1e2330" 
TEXT      = "#c8cdd6" 
MUTED     = "#5a6070" 
ORANGE    = "#f97316" 
BLUE      = "#3b82f6" 
GREEN     = "#22c55e" 
RED       = "#ef4444" 
AMBER     = "#f59e0b" 
PURPLE    = "#a78bfa" 
TEAL      = "#14b8a6" 

TICKER_COLORS = {
    "SPY":  ORANGE,
    "NVDA": BLUE,
    "TSLA": GREEN,
    "JPM":  AMBER,
}

def setup_bloomberg(fig, axes=None):
    """Apply Bloomberg Terminal dark theme to figure and axes."""
    fig.patch.set_facecolor(BG)
    if axes is None:
        return
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.7, linestyle="--")
        ax.set_axisbelow(True)

def add_watermark(fig, text="FinFlow 1.0  |  EDA"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=7, color=MUTED, alpha=0.5, family="monospace")

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading aligned data...")
train = pd.read_csv(ALIGNED / "train_aligned.csv", parse_dates=["timestamp"])
test  = pd.read_csv(ALIGNED / "test_aligned.csv",  parse_dates=["timestamp"])
full  = pd.concat([train, test], ignore_index=True)

print("Loading raw news data...")
news_train = pd.read_csv(RAW / "news_data_train.csv", parse_dates=["created_at"])
news_test  = pd.read_csv(RAW / "news_data_test.csv",  parse_dates=["created_at"])
news_all   = pd.concat([news_train, news_test], ignore_index=True)

TARGET_TICKERS = ["NVDA", "TSLA", "JPM", "SPY"]
SPLIT_DATE = pd.Timestamp("2025-01-01", tz="UTC")

# FIGURE 1 — Close Price Time Series
print("[1/9] Price time series...")

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)
setup_bloomberg(fig, axes)
fig.suptitle("FIGURE 1  |  Hourly Close Price — All Tickers  |  Jan 2023 – Jan 2026",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.98)

for ax, sym in zip(axes, TARGET_TICKERS):
    sym_data = full[full["symbol"] == sym].copy()
    color = TICKER_COLORS[sym]

    ax.plot(sym_data["timestamp"], sym_data["close"],
            color=color, linewidth=0.7, alpha=0.95)

    # Train/test split line
    ax.axvline(SPLIT_DATE, color=RED, linewidth=1.2, linestyle="--", alpha=0.8)

    # Shade test region
    ax.axvspan(SPLIT_DATE, sym_data["timestamp"].max(),
               alpha=0.05, color=RED, zorder=0)

    # Ticker label
    ax.text(0.01, 0.85, sym, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=color, family="monospace")

    # Current price annotation
    last_price = sym_data["close"].iloc[-1]
    ax.text(0.99, 0.85, f"${last_price:,.2f}", transform=ax.transAxes,
            fontsize=10, color=TEXT, ha="right", family="monospace")

    ax.set_ylabel("Close ($)", color=MUTED, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

# Split label on last axis
axes[-1].text(SPLIT_DATE, axes[-1].get_ylim()[0],
              "  TRAIN | TEST", color=RED, fontsize=7.5,
              va="bottom", family="monospace")

plt.tight_layout(rect=[0, 0, 1, 0.96])
add_watermark(fig)
fig.savefig(CHARTS / "01_price_timeseries.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 01_price_timeseries.png")

# FIGURE 2 — Weekly News Volume Timeline
print("[2/9] Weekly news volume...")

fig, ax = plt.subplots(figsize=(16, 5))
setup_bloomberg(fig, [ax])
fig.suptitle("FIGURE 2  |  Walk-Forward Analysis Timeline  |  Weekly News Volume",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

# Compute weekly counts
def parse_explode_news(df):
    d = df.copy()
    d["symbols_list"] = d["symbols"].apply(ast.literal_eval)
    d = d.explode("symbols_list")
    d = d[d["symbols_list"].isin(TARGET_TICKERS)]
    return d

news_train_exp = parse_explode_news(news_train)
news_test_exp  = parse_explode_news(news_test)

train_weekly = (news_train_exp.set_index("created_at")
                .resample("W")["id"].count())
test_weekly  = (news_test_exp.set_index("created_at")
                .resample("W")["id"].count())

ax.fill_between(train_weekly.index, train_weekly.values,
                alpha=0.3, color=ORANGE, zorder=2)
ax.plot(train_weekly.index, train_weekly.values,
        color=ORANGE, linewidth=1.2, label="Train (2023–2024)", zorder=3)

ax.fill_between(test_weekly.index, test_weekly.values,
                alpha=0.3, color=BLUE, zorder=2)
ax.plot(test_weekly.index, test_weekly.values,
        color=BLUE, linewidth=1.2, label="Test (2025)", zorder=3)

ax.axvline(SPLIT_DATE, color=RED, linewidth=2, linestyle="--",
           label="Train/Test Split", zorder=5)

ax.set_ylabel("Articles Published per Week", color=TEXT, fontsize=10)
ax.set_xlabel("Date", color=TEXT, fontsize=10)
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

plt.tight_layout()
add_watermark(fig)
fig.savefig(CHARTS / "02_news_volume_timeline.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 02_news_volume_timeline.png")

# FIGURE 3 — Modality Imbalance
print("[3/9] Modality imbalance...")

fig, ax = plt.subplots(figsize=(10, 6))
setup_bloomberg(fig, [ax])
fig.suptitle("FIGURE 3  |  Modality Imbalance Across Target Assets  |  Combined Dataset",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

price_counts = full.groupby("symbol").size()
news_counts  = pd.concat([news_train_exp, news_test_exp])["symbols_list"].value_counts()
news_counts  = news_counts.reindex(TARGET_TICKERS).fillna(0)

x      = np.arange(len(TARGET_TICKERS))
width  = 0.35

bars1 = ax.bar(x - width/2, [price_counts[s] for s in TARGET_TICKERS],
               width, label="Hourly Price Rows", color=BLUE, alpha=0.85, zorder=3)
bars2 = ax.bar(x + width/2, [news_counts[s] for s in TARGET_TICKERS],
               width, label="News Mentions", color=ORANGE, alpha=0.85, zorder=3)

ax.set_yscale("log")
ax.set_ylabel("Total Count (Log Scale)", color=TEXT, fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(TARGET_TICKERS, color=TEXT, fontsize=11, fontweight="bold")
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
            f"{int(bar.get_height()):,}", ha="center", va="bottom",
            color=BLUE, fontsize=8, fontweight="bold", family="monospace")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
            f"{int(bar.get_height()):,}", ha="center", va="bottom",
            color=ORANGE, fontsize=8, fontweight="bold", family="monospace")

plt.tight_layout()
add_watermark(fig)
fig.savefig(CHARTS / "03_modality_imbalance.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 03_modality_imbalance.png")

# FIGURE 4 — Returns Distribution
print("[4/9] Returns distribution...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
setup_bloomberg(fig, axes.flat)
fig.suptitle("FIGURE 4  |  Hourly Returns Distribution  |  Training Window 2023–2025",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

for ax, sym in zip(axes.flat, TARGET_TICKERS):
    color = TICKER_COLORS[sym]
    rets = train[train["symbol"] == sym]["returns"].dropna()
    rets_clipped = rets.clip(-0.05, 0.05)

    n, bins, patches = ax.hist(rets_clipped, bins=80, color=color,
                                alpha=0.7, edgecolor="none", density=True, zorder=3)

    # Overlay normal distribution
    from scipy.stats import norm, kurtosis, skew
    mu, sigma = rets.mean(), rets.std()
    x_range = np.linspace(rets_clipped.min(), rets_clipped.max(), 300)
    ax.plot(x_range, norm.pdf(x_range, mu, sigma),
            color="white", linewidth=1.5, alpha=0.6, linestyle="--", label="Normal fit")

    # Zero line
    ax.axvline(0, color=RED, linewidth=1.5, alpha=0.8)

    # Stats box
    stats_text = (f"μ = {mu*100:.4f}%\n"
                  f"σ = {sigma*100:.3f}%\n"
                  f"Skew = {skew(rets.dropna()):.2f}\n"
                  f"Kurt = {kurtosis(rets.dropna()):.2f}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            va="top", ha="right", fontsize=8, color=TEXT,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=BORDER, alpha=0.8))

    ax.set_title(sym, color=color, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Hourly Return", color=MUTED, fontsize=9)
    ax.set_ylabel("Density", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

plt.tight_layout()
add_watermark(fig)
fig.savefig(CHARTS / "04_returns_distribution.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 04_returns_distribution.png")

# FIGURE 5 — Technical Feature Correlation Matrix (SPY)
print("[5/9] Correlation matrix...")

fig, ax = plt.subplots(figsize=(13, 11))
setup_bloomberg(fig, [ax])
fig.suptitle("FIGURE 5  |  Technical Feature Correlation Matrix  |  SPY Training Window",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

feat_cols = [
    "open", "high", "low", "close", "vwap",
    "returns", "log_returns", "volatility_5h", "volatility_24h",
    "volume", "volume_change", "volume_ratio",
    "rsi_14", "macd", "macd_hist", "bb_pct", "bb_width",
    "momentum_4h", "momentum_24h", "vwap_spread"
]
spy_train = train[train["symbol"] == "SPY"][feat_cols].dropna()
corr = spy_train.corr()

# Custom diverging colormap anchored at 0
import matplotlib.colors as mcolors
cmap = plt.cm.RdBu_r

im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.ax.tick_params(colors=MUTED, labelsize=8)
cbar.outline.set_edgecolor(BORDER)

# Labels
ax.set_xticks(range(len(feat_cols)))
ax.set_yticks(range(len(feat_cols)))
ax.set_xticklabels(feat_cols, rotation=45, ha="right",
                   color=TEXT, fontsize=8, family="monospace")
ax.set_yticklabels(feat_cols, color=TEXT, fontsize=8, family="monospace")

# Annotate with values
for i in range(len(feat_cols)):
    for j in range(len(feat_cols)):
        val = corr.values[i, j]
        if abs(val) > 0.5 or i == j:
            color = "white" if abs(val) < 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color, fontweight="bold" if i == j else "normal")

plt.tight_layout()
add_watermark(fig)
fig.savefig(CHARTS / "05_correlation_matrix.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 05_correlation_matrix.png")

# FIGURE 6 — FinBERT Input Token Distribution
print("[6/9] FinBERT token distribution...")

fig, ax = plt.subplots(figsize=(12, 6))
setup_bloomberg(fig, [ax])
fig.suptitle("FIGURE 6  |  FinBERT NLP Input Word Count Distribution  |  Combined News",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

# Word counts from the aligned data (non-empty rows only)
word_counts = (train[train["finbert_input"].notna() & (train["finbert_input"] != "")]["finbert_input"]
               .astype(str)
               .apply(lambda x: len(x.split()))
               .clip(0, 120))

median_wc = word_counts.median()

ax.hist(word_counts, bins=60, color=PURPLE, alpha=0.8,
        edgecolor="none", zorder=3, density=False)
ax.axvline(median_wc, color=RED, linewidth=2, linestyle="--",
           label=f"Median: {median_wc:.0f} words")
ax.axvline(128, color=AMBER, linewidth=1.5, linestyle=":",
           label="FinBERT soft limit (~128 tokens)")

ax.set_xlabel("Total Word Count (Headline + Summary)", color=TEXT, fontsize=10)
ax.set_ylabel("Frequency", color=TEXT, fontsize=10)
ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)

# Coverage annotation
total = len(word_counts)
within = (word_counts <= 128).sum()
ax.text(0.97, 0.97, f"{within/total:.1%} of inputs\nwithin 128-token limit",
        transform=ax.transAxes, va="top", ha="right", fontsize=9,
        color=GREEN, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=BORDER))

plt.tight_layout()
add_watermark(fig)
fig.savefig(CHARTS / "06_finbert_token_dist.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 06_finbert_token_dist.png")

# FIGURE 7 — News Coverage Rate per Ticker
print("[7/9] News coverage rate...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
setup_bloomberg(fig, axes)
fig.suptitle("FIGURE 7  |  News Coverage Rate  |  % of Market Hours with ≥1 Article",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

for ax, df, split_label in zip(axes, [train, test], ["Train (2023–2025)", "Test (2025–2026)"]):
    coverage = {}
    avg_count = {}
    for sym in TARGET_TICKERS:
        subset = df[df["symbol"] == sym]
        cov = (subset["news_count"] > 0).mean()
        avg = subset.loc[subset["news_count"] > 0, "news_count"].mean()
        coverage[sym] = cov * 100
        avg_count[sym] = avg

    colors = [TICKER_COLORS[s] for s in TARGET_TICKERS]
    bars = ax.barh(TARGET_TICKERS, [coverage[s] for s in TARGET_TICKERS],
                   color=colors, alpha=0.85, zorder=3, height=0.5)

    # Reference line at 50%
    ax.axvline(50, color=MUTED, linewidth=1, linestyle="--", alpha=0.5)

    # Value labels
    for bar, sym in zip(bars, TARGET_TICKERS):
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2,
                f"{w:.1f}%  (avg {avg_count[sym]:.1f} art/hr)",
                va="center", color=TICKER_COLORS[sym], fontsize=9, family="monospace")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Coverage %", color=TEXT, fontsize=10)
    ax.set_title(split_label, color=AMBER, fontsize=11, pad=8)
    ax.yaxis.set_tick_params(labelcolor=TEXT, labelsize=11)

plt.tight_layout()
add_watermark(fig)
fig.savefig(CHARTS / "07_news_coverage_rate.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 07_news_coverage_rate.png")

# FIGURE 8 — Volume Heatmap by Hour of Day × Day of Week
print("[8/9] Volume heatmap...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
setup_bloomberg(fig, axes.flat)
fig.suptitle("FIGURE 8  |  Avg Hourly Volume  |  Hour of Day × Day of Week  |  Training Window",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

days   = ["Mon", "Tue", "Wed", "Thu", "Fri"]
hours  = list(range(9, 20))  # market hours approx

for ax, sym in zip(axes.flat, TARGET_TICKERS):
    color = TICKER_COLORS[sym]
    sym_data = train[train["symbol"] == sym].copy()
    sym_data["hour"] = pd.DatetimeIndex(sym_data["timestamp"]).hour
    sym_data["dow"]  = pd.DatetimeIndex(sym_data["timestamp"]).dayofweek  # 0=Mon

    pivot = (sym_data[sym_data["dow"] < 5]
             .groupby(["hour", "dow"])["volume"]
             .mean()
             .unstack(fill_value=0))

    # Reindex to ensure consistent shape
    pivot = pivot.reindex(hours, fill_value=0)
    pivot = pivot.reindex(columns=[0,1,2,3,4], fill_value=0)

    # Custom colormap: dark bg → ticker color
    from matplotlib.colors import LinearSegmentedColormap
    cmap_custom = LinearSegmentedColormap.from_list(
        "bb", [PANEL, color], N=256
    )

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap_custom,
                   interpolation="nearest")

    ax.set_xticks(range(5))
    ax.set_xticklabels(days, color=TEXT, fontsize=9)
    ax.set_yticks(range(len(hours)))
    ax.set_yticklabels([f"{h:02d}:00" for h in hours],
                        color=TEXT, fontsize=8, family="monospace")
    ax.set_title(sym, color=color, fontsize=12, fontweight="bold")

    # Colorbar per subplot
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.set_label("Avg Volume", color=MUTED, fontsize=7)

plt.tight_layout()
add_watermark(fig)
fig.savefig(CHARTS / "08_volume_heatmap.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 08_volume_heatmap.png")

# FIGURE 9 — NVDA 4-Panel Technical Dashboard
print("[9/9] Technical dashboard (NVDA)...")

# Use 2024 data for readability
nvda = train[train["symbol"] == "NVDA"].copy()
nvda = nvda[nvda["timestamp"] >= "2024-01-01"].copy()

fig = plt.figure(figsize=(18, 12))
setup_bloomberg(fig)
fig.suptitle("FIGURE 9  |  NVDA  |  Technical Analysis Dashboard  |  2024  (Training Window)",
             color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

gs = GridSpec(4, 1, figure=fig, hspace=0.06,
              height_ratios=[3, 1, 1, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3], sharex=ax1)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.7, linestyle="--")
    ax.set_axisbelow(True)

ts = nvda["timestamp"]

# Panel 1: Price + Bollinger Bands
ax1.plot(ts, nvda["close"], color=ORANGE, linewidth=1, label="Close", zorder=4)
ax1.fill_between(ts, nvda["bb_lower"], nvda["bb_upper"],
                 alpha=0.08, color=BLUE, zorder=2)
ax1.plot(ts, nvda["bb_upper"], color=BLUE, linewidth=0.6, alpha=0.5,
         linestyle="--", label="BB Upper")
ax1.plot(ts, nvda["bb_lower"], color=BLUE, linewidth=0.6, alpha=0.5,
         linestyle="--", label="BB Lower")
ax1.set_ylabel("Price ($)", color=TEXT, fontsize=9)
ax1.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
           fontsize=8, loc="upper left")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

# Panel 2: RSI
ax2.plot(ts, nvda["rsi_14"], color=PURPLE, linewidth=0.9, label="RSI(14)")
ax2.axhline(70, color=RED,   linewidth=0.8, linestyle="--", alpha=0.7)
ax2.axhline(30, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.7)
ax2.axhline(50, color=MUTED, linewidth=0.5, linestyle=":", alpha=0.4)
ax2.fill_between(ts, nvda["rsi_14"], 70,
                 where=(nvda["rsi_14"] >= 70), alpha=0.15, color=RED, zorder=2)
ax2.fill_between(ts, nvda["rsi_14"], 30,
                 where=(nvda["rsi_14"] <= 30), alpha=0.15, color=GREEN, zorder=2)
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI", color=TEXT, fontsize=9)
ax2.text(ts.iloc[-1], 72, "  OB", color=RED,   fontsize=7.5, va="bottom")
ax2.text(ts.iloc[-1], 28, "  OS", color=GREEN, fontsize=7.5, va="top")
plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)

# Panel 3: MACD
ax3.plot(ts, nvda["macd"],        color=ORANGE, linewidth=0.9, label="MACD")
ax3.plot(ts, nvda["macd_signal"], color=BLUE,   linewidth=0.9, label="Signal")
ax3.bar(ts, nvda["macd_hist"],
        color=np.where(nvda["macd_hist"] >= 0, GREEN, RED),
        alpha=0.5, width=0.04, label="Histogram")
ax3.axhline(0, color=MUTED, linewidth=0.5, linestyle="--")
ax3.set_ylabel("MACD", color=TEXT, fontsize=9)
ax3.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=7, ncol=3)
plt.setp(ax3.xaxis.get_majorticklabels(), visible=False)

# Panel 4: Volume
vol_colors = np.where(nvda["close"].diff() >= 0, GREEN, RED)
ax4.bar(ts, nvda["volume"], color=vol_colors, alpha=0.7, width=0.04)
ax4.plot(ts, nvda["volume_ma_24h"], color=AMBER, linewidth=1.2,
         alpha=0.8, label="24h MA")
ax4.set_ylabel("Volume", color=TEXT, fontsize=9)
ax4.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

# Shared x-axis: remove gap
plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)

add_watermark(fig)
fig.savefig(CHARTS / "09_technical_dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✓ Saved 09_technical_dashboard.png")
