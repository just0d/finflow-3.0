"""
FinFlow 2.0 — Shared Data Loader for the new dual-modal models
Reuses *exactly* the same tensors that FinFlow 1.0's Phase 3 produced.

Splits used (identical to Phase 3):
    train: Jan 2023 – Aug 2024
    val  : Sep 2024 – Jan 2025
    test : Jan 2025 – Jan 2026
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT  = Path(__file__).resolve().parents[2]      # FinFlow 1.0 root
PREP  = ROOT / "data" / "preprocessed"
PROC  = ROOT / "data" / "processed"

LOOKBACK = 24
TEXT_FEATURE_NAMES = [
    "sentiment_score",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "log_news_count",
    "score_x_news",
]
N_TEXT_FEAT = len(TEXT_FEATURE_NAMES)

# Build sentiment-aligned tensors for every (X_price, label) sequence

def _load_sentiment_long(split: str) -> pd.DataFrame:
    """
    Returns a DataFrame keyed by (symbol, timestamp) with the
    FinBERT-summary columns we need.  For the *val* split we slice
    the train CSV (Phase 3 split val out of train chronologically).
    """
    if split in ("train", "val"):
        path = PROC / "train_sentiment.csv"
    else:
        path = PROC / "test_sentiment.csv"

    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Fill any missing sentiment fields with neutral defaults
    df["sentiment_score"]    = df["sentiment_score"].fillna(0.0)
    df["sentiment_positive"] = df["sentiment_positive"].fillna(0.0)
    df["sentiment_negative"] = df["sentiment_negative"].fillna(0.0)
    df["sentiment_neutral"]  = df["sentiment_neutral"].fillna(1.0)
    df["news_count"]         = df["news_count"].fillna(0)

    df["log_news_count"] = np.log1p(df["news_count"].astype(float))
    df["score_x_news"]   = df["sentiment_score"] * df["log_news_count"]

    keep_cols = ["symbol", "timestamp"] + TEXT_FEATURE_NAMES
    return df[keep_cols].copy()

def _build_aligned_tensors(split: str) -> dict:
    """
    Loads the preprocessed price tensors AND a 24-hour sentiment tensor
    matched to each window.
    """
    X_price = np.load(PREP / f"X_{split}.npy")          # (N, 24, 19)
    y       = np.load(PREP / f"y_{split}.npy")          # (N,)
    meta    = pd.read_csv(PREP / f"meta_{split}.csv", parse_dates=["timestamp"])
    if meta["timestamp"].dt.tz is None:
        meta["timestamp"] = meta["timestamp"].dt.tz_localize("UTC")

    # Long table of per-hour sentiment for the symbols we care about
    sent = _load_sentiment_long(split)
    # Multi-index for fast lookup
    sent = sent.set_index(["symbol", "timestamp"]).sort_index()

    # Build a dense per-symbol → DataFrame of timestamps for offset lookups
    by_sym = {sym: g.reset_index(level="symbol", drop=True)
              for sym, g in sent.groupby(level="symbol")}

    N = len(meta)
    X_text = np.zeros((N, LOOKBACK, N_TEXT_FEAT), dtype=np.float32)
    X_sent = np.zeros((N, LOOKBACK),              dtype=np.float32)
    X_news = np.zeros((N, LOOKBACK),              dtype=np.float32)

    # Per-symbol caching: for each symbol we have the FULL hourly grid
    # already in `by_sym[sym]`, indexed by timestamp.  We use
    # asof-style nearest-on-or-before lookup with a 1-hour tolerance:
    # since the upstream pipeline is hourly-aligned, this is just an
    # exact match that tolerates a daylight-saving 30-min shift.
    for sym, sub in by_sym.items():
        # Prepare numpy arrays for fast indexing
        sub_index = sub.index.values.astype("datetime64[ns]")
        sub_score = sub["sentiment_score"].to_numpy(dtype=np.float32)
        sub_pos   = sub["sentiment_positive"].to_numpy(dtype=np.float32)
        sub_neg   = sub["sentiment_negative"].to_numpy(dtype=np.float32)
        sub_neu   = sub["sentiment_neutral"].to_numpy(dtype=np.float32)
        sub_lcnt  = sub["log_news_count"].to_numpy(dtype=np.float32)
        sub_xnws  = sub["score_x_news"].to_numpy(dtype=np.float32)

        # Build a hash from timestamp (ns int) → row index
        ts_to_idx = {int(t): i for i, t in enumerate(sub_index)}

        # Walk through every meta row for this symbol
        rows_for_sym = meta.index[meta["symbol"] == sym].to_numpy()
        for row_i in rows_for_sym:
            anchor = pd.Timestamp(meta.at[row_i, "timestamp"]).tz_convert("UTC")
            # Build window of 24 hourly slots ending at anchor
            for k in range(LOOKBACK):
                slot = anchor - pd.Timedelta(hours=LOOKBACK - 1 - k)
                t_ns = int(np.datetime64(slot.tz_localize(None), "ns"))
                idx  = ts_to_idx.get(t_ns)
                if idx is None:
                    continue   # no row → leave as zeros (neutral)
                X_text[row_i, k, 0] = sub_score[idx]
                X_text[row_i, k, 1] = sub_pos  [idx]
                X_text[row_i, k, 2] = sub_neg  [idx]
                X_text[row_i, k, 3] = sub_neu  [idx]
                X_text[row_i, k, 4] = sub_lcnt [idx]
                X_text[row_i, k, 5] = sub_xnws [idx]
                X_sent[row_i, k]    = sub_score[idx]
                # log_news_count > 0 ⇔ news_count > 0
                X_news[row_i, k]    = 1.0 if sub_lcnt[idx] > 0 else 0.0

    return {
        "X_price": X_price.astype(np.float32),
        "X_text":  X_text,
        "X_sent":  X_sent,
        "X_news":  X_news,
        "y":       y.astype(np.int64),
        "meta":    meta.reset_index(drop=True),
    }

# DualModalDataset

class DualModalDataset(Dataset):
    """Dataset returning (X_price, X_text, X_sent, X_news, y) per item."""

    def __init__(self, data_dict: dict):
        self.X_price = torch.from_numpy(data_dict["X_price"])
        self.X_text  = torch.from_numpy(data_dict["X_text"])
        self.X_sent  = torch.from_numpy(data_dict["X_sent"])
        self.X_news  = torch.from_numpy(data_dict["X_news"])
        self.y       = torch.from_numpy(data_dict["y"])

    def __len__(self):  return self.y.shape[0]

    def __getitem__(self, i):
        return (self.X_price[i], self.X_text[i],
                self.X_sent[i], self.X_news[i], self.y[i])

def load_cached(split: str) -> dict:
    """Fast loader if precompute_tensors.py has been run."""
    cache = ROOT / "finflow2" / "cache"
    if not (cache / f"{split}_y.npy").exists():
        return _build_aligned_tensors(split)
    return {
        "X_price": np.load(cache / f"{split}_X_price.npy"),
        "X_text":  np.load(cache / f"{split}_X_text.npy"),
        "X_sent":  np.load(cache / f"{split}_X_sent.npy"),
        "X_news":  np.load(cache / f"{split}_X_news.npy"),
        "y":       np.load(cache / f"{split}_y.npy"),
        "meta":    pd.read_csv(cache / f"{split}_meta.csv",
                                parse_dates=["timestamp"]),
    }

def make_loaders(batch_size: int = 256,
                 num_workers: int = 0) -> tuple[dict, DataLoader, DataLoader, DataLoader]:
    """
    Builds train/val/test DualModalDataset + DataLoaders.

    Returns: (cache_dict, train_loader, val_loader, test_loader)
    The cache_dict maps split → raw numpy data so callers can use the
    underlying tensors directly (e.g. for inference on test).
    """
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = _build_aligned_tensors(split)

    train_loader = DataLoader(DualModalDataset(splits["train"]),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(DualModalDataset(splits["val"]),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(DualModalDataset(splits["test"]),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return splits, train_loader, val_loader, test_loader

# CLI quick-check
if __name__ == "__main__":
    splits, tl, vl, sl = make_loaders(batch_size=4)
    print("Split sizes:")
    for k in ("train", "val", "test"):
        d = splits[k]
        print(f"  {k:5s} N={len(d['y']):,}  "
              f"X_price={d['X_price'].shape}  X_text={d['X_text'].shape}  "
              f"news_hours={d['X_news'].sum():.0f}")

    # Example batch
    p, t, s, n, y = next(iter(tl))
    print(f"\nBatch shapes: price={p.shape}  text={t.shape}  "
          f"sent={s.shape}  news={n.shape}  y={y.shape}")
