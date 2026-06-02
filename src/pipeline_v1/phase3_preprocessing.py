"""
FinFlow 1.0 — Phase 3 Step 1: Tensor Preprocessing
Converts aligned CSVs into 3D tensors ready for neural networks.

Key design decisions:
  - Uses ONLY mathematical features (no sentiment) for Phase 3 baselines
  - Per-ticker StandardScaler (fit on train, transform test) — handles
    tickers like NVDA ($800) vs JPM ($200) having different MACD scales
  - Chronological train/val split: train=Jan2023-Aug2024, val=Sep-Dec2024
  - 24-hour lookback sliding window (standard for hourly financial models)
  - Labels remapped: SELL(-1)→0, HOLD(0)→1, BUY(1)→2
  - Saved outputs that Prophet, LSTM, and Transformer all read from
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Config
BASE    = Path(__file__).parent
ALIGNED = BASE / "data" / "aligned"
PROC    = BASE / "data" / "processed"
PREP    = BASE / "data" / "preprocessed"
PREP.mkdir(parents=True, exist_ok=True)

LOOKBACK         = 24        # 24-hour lookback window
VAL_SPLIT_DATE   = pd.Timestamp("2024-09-01", tz="UTC")   # last ~4 months of train = val
TARGET_TICKERS   = ["NVDA", "TSLA", "JPM", "SPY"]

# Mathematical features only (relative / normalized-safe)
FEATURE_COLS = [
    # Returns & log-returns
    "returns", "log_returns",
    # Volatility regime
    "volatility_5h", "volatility_24h",
    # Volume signals
    "volume_change", "volume_ratio",
    # Momentum signals
    "momentum_4h", "momentum_12h", "momentum_24h",
    # Oscillators
    "rsi_14",
    # Trend signals
    "macd", "macd_signal", "macd_hist",
    # Bollinger band geometry
    "bb_width", "bb_pct",
    # Microstructure
    "vwap_spread", "body_size", "upper_wick", "lower_wick",
]
N_FEATURES = len(FEATURE_COLS)

# 1. Load Data
print("[1/5] Loading data...")

def load_split(split: str) -> pd.DataFrame:
    """Load from data/processed/ if Phase 2 ran, else data/aligned/ — either works."""
    for candidate in [
        PROC    / f"{split}_sentiment.csv",
        ALIGNED / f"{split}_aligned.csv",
    ]:
        if candidate.exists():
            print(f"  {split:5s} ← {candidate.name}")
            return pd.read_csv(candidate, parse_dates=["timestamp"])
    raise FileNotFoundError(f"No data file found for split='{split}'")

train_raw = load_split("train")
test_raw  = load_split("test")

# Ensure UTC-aware timestamps
for df in [train_raw, test_raw]:
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

print(f"  Train: {train_raw.shape[0]:,} rows | Test: {test_raw.shape[0]:,} rows")

# 2. Remap Labels 
print("[2/5] Remapping labels: SELL(-1)→0  HOLD(0)→1  BUY(1)→2 ...")

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}

train_raw["label"] = train_raw["target_direction"].map(LABEL_MAP)
test_raw["label"]  = test_raw["target_direction"].map(LABEL_MAP)

# 3. Fit per-ticker StandardScalers on TRAIN data
print("[3/5] Fitting per-ticker StandardScalers on training data...")

scalers: dict[str, StandardScaler] = {}
for sym in TARGET_TICKERS:
    sym_train = train_raw[train_raw["symbol"] == sym][FEATURE_COLS].dropna()
    sc = StandardScaler()
    sc.fit(sym_train)
    scalers[sym] = sc
    print(f"  {sym:5s}: fitted on {len(sym_train):,} clean rows")

# Save scalers
with open(PREP / "scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)
with open(PREP / "feature_cols.txt", "w") as f:
    f.write("\n".join(FEATURE_COLS))

# 4. Create Sliding Window Sequences
print(f"[4/5] Creating {LOOKBACK}-hour sliding windows...")

def build_sequences(df: pd.DataFrame, scalers: dict, split_label: str):
    """
    For each ticker, build (X, y) sequences using a sliding 24-hour window.
    Returns:
        X : ndarray (N, LOOKBACK, N_FEATURES)
        y : ndarray (N,)
        meta : DataFrame with symbol, timestamp for each sequence
    """
    all_X, all_y, all_meta = [], [], []

    for sym in TARGET_TICKERS:
        sym_df = (df[df["symbol"] == sym]
                  .sort_values("timestamp")
                  .copy()
                  .reset_index(drop=True))

        # Drop rows with any NaN in features OR label
        valid_mask = sym_df[FEATURE_COLS + ["label"]].notna().all(axis=1)
        sym_df = sym_df[valid_mask].reset_index(drop=True)

        if len(sym_df) < LOOKBACK + 1:
            print(f"  WARNING: {sym} has {len(sym_df)} clean rows — too few for windows")
            continue

        # Scale features using this ticker's fitted scaler
        features_raw    = sym_df[FEATURE_COLS].values.astype(np.float32)
        features_scaled = scalers[sym].transform(features_raw).astype(np.float32)

        labels     = sym_df["label"].values.astype(np.int64)
        timestamps = sym_df["timestamp"].values

        n_windows = 0
        for i in range(LOOKBACK - 1, len(sym_df)):
            window = features_scaled[i - LOOKBACK + 1 : i + 1]   # shape (24, N_FEATURES)
            target = int(labels[i])

            # Skip if any NaN survived scaling
            if np.isnan(window).any():
                continue

            all_X.append(window)
            all_y.append(target)
            all_meta.append({"symbol": sym, "timestamp": timestamps[i]})
            n_windows += 1

        print(f"  {sym:5s} [{split_label}]: {n_windows:,} sequences")

    X    = np.array(all_X,   dtype=np.float32)
    y    = np.array(all_y,   dtype=np.int64)
    meta = pd.DataFrame(all_meta)
    return X, y, meta

# Build full train set, then chronologically split into train/val
X_all, y_all, meta_all = build_sequences(train_raw, scalers, "full-train")

# Chronological split — val = sequences whose final timestamp >= VAL_SPLIT_DATE
timestamps_utc = pd.to_datetime(meta_all["timestamp"])
# Normalise to UTC-aware for comparison
if timestamps_utc.dt.tz is None:
    timestamps_utc = timestamps_utc.dt.tz_localize("UTC")
else:
    timestamps_utc = timestamps_utc.dt.tz_convert("UTC")
val_mask = timestamps_utc >= VAL_SPLIT_DATE

train_mask = ~val_mask

X_train, y_train, meta_train = X_all[train_mask],  y_all[train_mask],  meta_all[train_mask]
X_val,   y_val,   meta_val   = X_all[val_mask],    y_all[val_mask],    meta_all[val_mask]

# Build test set
X_test, y_test, meta_test = build_sequences(test_raw, scalers, "test")

# 5. Save
print("[5/5] Saving preprocessed tensors...")

for name, X, y, meta in [
    ("train", X_train, y_train, meta_train),
    ("val",   X_val,   y_val,   meta_val),
    ("test",  X_test,  y_test,  meta_test),
]:
    np.save(PREP / f"X_{name}.npy", X)
    np.save(PREP / f"y_{name}.npy", y)
    meta.to_csv(PREP / f"meta_{name}.csv", index=False)

# Summary
print("\n" + "="*60)
print("  PREPROCESSING SUMMARY")
print(f"\n  Tensor shapes:")
print(f"    X_train : {X_train.shape}  ← Jan 2023 – Aug 2024")
print(f"    X_val   : {X_val.shape}  ← Sep 2024 – Jan 2025")
print(f"    X_test  : {X_test.shape}  ← Jan 2025 – Jan 2026")
print(f"    Features: {N_FEATURES} — {FEATURE_COLS}")

print(f"\n  Label distributions (SELL=0, HOLD=1, BUY=2):")
for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    counts = np.bincount(y, minlength=3)
    total  = len(y)
    print(f"    {name:5s}: "
          f"SELL={counts[0]:,}({counts[0]/total:.1%}) "
          f"HOLD={counts[1]:,}({counts[1]/total:.1%}) "
          f"BUY={counts[2]:,}({counts[2]/total:.1%})")

print(f"\n  Files saved to: {PREP}")
print("\n  ✓ Ready for phase3_train_evaluate.py")
