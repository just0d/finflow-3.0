"""
FinFlow 1.0 — Phase 1: Data Alignment Pipeline
Maps irregular, second-level news timestamps onto strict 1-hour price intervals.
For each (symbol, hour) bucket:
  - Combines all headlines + summaries into a single FinBERT input string
  - Counts articles published in that window
  - Left-joins onto the master price grid so every hour has a record
  - Hours with zero news get empty text and a news_count of 0
  - Adds lagged technical features needed for the LSTM

"""

import pandas as pd
import numpy as np
import ast
import os
from pathlib import Path

# Paths
BASE    = Path(__file__).parent
RAW     = BASE / "data" / "raw"
ALIGNED = BASE / "data" / "aligned"
ALIGNED.mkdir(parents=True, exist_ok=True)

# Move raw CSVs to data/raw if they're still in root
for fname in ["price_data_train.csv", "price_data_test.csv",
              "news_data_train.csv",  "news_data_test.csv"]:
    src = BASE / fname
    dst = RAW / fname
    if src.exists() and not dst.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        print(f"  Moved {fname} → data/raw/")

TARGET_TICKERS = ["NVDA", "TSLA", "JPM", "SPY"]

# 1. Load Data
print("\n[1/6] Loading raw data...")

price_train = pd.read_csv(RAW / "price_data_train.csv", parse_dates=["timestamp"])
price_test  = pd.read_csv(RAW / "price_data_test.csv",  parse_dates=["timestamp"])
news_train  = pd.read_csv(RAW / "news_data_train.csv",  parse_dates=["created_at"])
news_test   = pd.read_csv(RAW / "news_data_test.csv",   parse_dates=["created_at"])

print(f"  Price train : {price_train.shape[0]:>6,} rows | {price_train['symbol'].nunique()} tickers")
print(f"  Price test  : {price_test.shape[0]:>6,} rows")
print(f"  News  train : {news_train.shape[0]:>6,} rows")
print(f"  News  test  : {news_test.shape[0]:>6,} rows")

# 2. Parse symbols & Explode to per-ticker rows
print("\n[2/6] Parsing symbols column & exploding to per-ticker rows...")

def parse_and_explode(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stringified list in 'symbols' → actual list, then explode."""
    df = df.copy()
    df["symbols_list"] = df["symbols"].apply(ast.literal_eval)
    df = df.explode("symbols_list").rename(columns={"symbols_list": "symbol"})
    # Keep only our 4 target tickers
    df = df[df["symbol"].isin(TARGET_TICKERS)].reset_index(drop=True)
    return df

news_train_exploded = parse_and_explode(news_train)
news_test_exploded  = parse_and_explode(news_test)

print(f"  News train (filtered) : {news_train_exploded.shape[0]:>6,} rows")
print(f"  News test  (filtered) : {news_test_exploded.shape[0]:>6,} rows")
print(f"  Articles per ticker (train):\n{news_train_exploded['symbol'].value_counts().to_string()}")

# 3. Build FinBERT Input Text
print("\n[3/6] Building FinBERT input text (headline + summary fallback)...")

def build_finbert_input(row) -> str:
    """
    FinBERT works best on short, focused text.
    Strategy: headline alone if no summary; headline + '. ' + summary if available.
    Cap at ~512 characters to stay within BERT token limits.
    """
    headline = str(row["headline"]).strip() if pd.notna(row["headline"]) else ""
    summary  = str(row["summary"]).strip()  if pd.notna(row["summary"])  else ""

    if summary:
        text = f"{headline}. {summary}"
    else:
        text = headline

    return text[:512]  # hard cap for FinBERT token safety

for df in [news_train_exploded, news_test_exploded]:
    df["finbert_input"] = df.apply(build_finbert_input, axis=1)

# Coverage stats
train_has_summary = news_train_exploded["summary"].notna().mean()
print(f"  Summary coverage (train) : {train_has_summary:.1%}")
print(f"  Using headline-only for  : {1-train_has_summary:.1%} of articles")

# 4. Floor Timestamps to Hourly Buckets
print("\n[4/6] Flooring news timestamps → hourly buckets...")

def floor_to_hour(df: pd.DataFrame, ts_col: str = "created_at") -> pd.DataFrame:
    df = df.copy()
    # Ensure UTC-aware
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize("UTC")
    else:
        df[ts_col] = df[ts_col].dt.tz_convert("UTC")
    df["hour_bucket"] = df[ts_col].dt.floor("h")
    return df

news_train_exploded = floor_to_hour(news_train_exploded)
news_test_exploded  = floor_to_hour(news_test_exploded)

# 5. Aggregate News per (symbol, hour)
print("\n[5/6] Aggregating news per (symbol, hour_bucket)...")

def aggregate_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (symbol, hour) bucket:
      - news_count        : number of articles
      - finbert_input     : all article texts joined by ' [SEP] '
      - headline_texts    : all headlines joined (for EDA / display)
    """
    agg = df.groupby(["symbol", "hour_bucket"]).agg(
        news_count      = ("finbert_input", "count"),
        finbert_input   = ("finbert_input", lambda x: " [SEP] ".join(x)),
        headline_texts  = ("headline",      lambda x: " | ".join(x.dropna().astype(str)))
    ).reset_index()
    return agg

news_agg_train = aggregate_news(news_train_exploded)
news_agg_test  = aggregate_news(news_test_exploded)

print(f"  Aggregated train buckets : {news_agg_train.shape[0]:>6,}")
print(f"  Aggregated test  buckets : {news_agg_test.shape[0]:>6,}")
print(f"  Max articles in one hour (train): {news_agg_train['news_count'].max()}")
print(f"  Median articles per hour (train): {news_agg_train['news_count'].median():.1f}")

# 6. Merge with Price Grid & Add Technical Features
print("\n[6/6] Left-joining onto price grid & engineering technical features...")

def normalize_price_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df

price_train = normalize_price_timestamps(price_train)
price_test  = normalize_price_timestamps(price_test)

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators per ticker in chronological order.
    All features are LAGGED (shifted by 1) to prevent look-ahead bias.
    """
    df = df.sort_values(["symbol", "timestamp"]).copy()
    results = []

    for sym, grp in df.groupby("symbol"):
        g = grp.copy().reset_index(drop=True)

        # Returns
        g["returns"]         = g["close"].pct_change()
        g["log_returns"]     = np.log(g["close"] / g["close"].shift(1))

        # Volatility (rolling std of returns)
        g["volatility_5h"]   = g["returns"].rolling(5).std()
        g["volatility_24h"]  = g["returns"].rolling(24).std()

        #  Volume features
        g["volume_change"]   = g["volume"].pct_change()
        g["volume_ma_24h"]   = g["volume"].rolling(24).mean()
        g["volume_ratio"]    = g["volume"] / g["volume_ma_24h"].replace(0, np.nan)

        #  Price momentum 
        g["momentum_4h"]     = g["close"] / g["close"].shift(4) - 1
        g["momentum_12h"]    = g["close"] / g["close"].shift(12) - 1
        g["momentum_24h"]    = g["close"] / g["close"].shift(24) - 1

        #  RSI (14-period) 
        delta   = g["close"].diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rs      = gain / loss.replace(0, np.nan)
        g["rsi_14"] = 100 - (100 / (1 + rs))

        #  MACD (12/26/9) 
        ema12        = g["close"].ewm(span=12, adjust=False).mean()
        ema26        = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"]    = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"]   = g["macd"] - g["macd_signal"]

        #  Bollinger Bands (20-period) 
        sma20         = g["close"].rolling(20).mean()
        std20         = g["close"].rolling(20).std()
        g["bb_upper"] = sma20 + 2 * std20
        g["bb_lower"] = sma20 - 2 * std20
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / sma20.replace(0, np.nan)
        g["bb_pct"]   = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)

        #  Price-VWAP spread 
        g["vwap_spread"] = (g["close"] - g["vwap"]) / g["vwap"].replace(0, np.nan)

        #  Candle body & wick ratios 
        g["body_size"]   = abs(g["close"] - g["open"]) / g["open"].replace(0, np.nan)
        g["upper_wick"]  = (g["high"] - g[["close", "open"]].max(axis=1)) / g["open"].replace(0, np.nan)
        g["lower_wick"]  = (g[["close", "open"]].min(axis=1) - g["low"]) / g["open"].replace(0, np.nan)

        #  Target label: direction of NEXT hour's close 
        # BUY=1, HOLD=0, SELL=-1 (multiclass for LSTM)
        next_ret       = g["close"].shift(-1) / g["close"] - 1
        g["target_direction"] = np.where(next_ret > 0.001, 1,
                                 np.where(next_ret < -0.001, -1, 0))

        # SHIFT all features by 1 to prevent look-ahead (except target which looks forward)
        lag_cols = [
            "returns", "log_returns", "volatility_5h", "volatility_24h",
            "volume_change", "volume_ma_24h", "volume_ratio",
            "momentum_4h", "momentum_12h", "momentum_24h",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_width", "bb_pct",
            "vwap_spread", "body_size", "upper_wick", "lower_wick"
        ]
        for col in lag_cols:
            if col in g.columns:
                g[col] = g[col].shift(1)

        results.append(g)

    return pd.concat(results, ignore_index=True).sort_values(["symbol", "timestamp"])

print("  Engineering features for price_train...")
price_train_feat = add_technical_features(price_train)
print("  Engineering features for price_test...")
price_test_feat  = add_technical_features(price_test)

def merge_price_news(price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """Left join price (master) ← news aggregates. Fill silence with neutral values."""
    merged = pd.merge(
        price_df,
        news_df.rename(columns={"hour_bucket": "timestamp"}),
        on=["symbol", "timestamp"],
        how="left"
    )
    # Fill silence
    merged["news_count"]     = merged["news_count"].fillna(0).astype(int)
    merged["finbert_input"]  = merged["finbert_input"].fillna("")
    merged["headline_texts"] = merged["headline_texts"].fillna("")

    # Placeholder sentiment columns (will be populated in Phase 2 by FinBERT)
    merged["sentiment_score"]    = np.nan   # [-1.0, +1.0]
    merged["sentiment_positive"] = np.nan   # raw FinBERT prob
    merged["sentiment_negative"] = np.nan
    merged["sentiment_neutral"]  = np.nan

    return merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

print("  Merging train...")
train_aligned = merge_price_news(price_train_feat, news_agg_train)
print("  Merging test...")
test_aligned  = merge_price_news(price_test_feat,  news_agg_test)

train_out = ALIGNED / "train_aligned.csv"
test_out  = ALIGNED / "test_aligned.csv"
train_aligned.to_csv(train_out, index=False)
test_aligned.to_csv(test_out,   index=False)

print("\n" + "="*60)
print("  PHASE 1 COMPLETE — ALIGNMENT SUMMARY")
print(f"\n  Train aligned : {train_aligned.shape[0]:>6,} rows × {train_aligned.shape[1]} cols")
print(f"  Test  aligned : {test_aligned.shape[0]:>6,} rows × {test_aligned.shape[1]} cols")
print(f"\n  Columns: {list(train_aligned.columns)}")

print("\n  News Coverage Rate (% of hours with ≥1 article):")
for sym in TARGET_TICKERS:
    subset = train_aligned[train_aligned["symbol"] == sym]
    coverage = (subset["news_count"] > 0).mean()
    avg_articles = subset.loc[subset["news_count"] > 0, "news_count"].mean()
    print(f"    {sym:5s}  {coverage:5.1%} covered  |  avg {avg_articles:.1f} articles/hour when covered")

print("\n  Null counts in feature columns (train):")
feat_cols = ["returns", "volatility_24h", "rsi_14", "macd", "bb_pct"]
print(train_aligned[feat_cols].isnull().sum().to_string())

print(f"\n  Saved → {train_out}")
print(f"  Saved → {test_out}")
print("\n  Phase 2 next: run FinBERT on the 'finbert_input' column to fill sentiment scores.")
