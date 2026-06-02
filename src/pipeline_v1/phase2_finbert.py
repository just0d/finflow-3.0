"""
FinFlow 1.0 — Phase 2: FinBERT Sentiment Inference
Runs ProsusAI/finbert over the aligned datasets to fill the four
reserved sentiment columns:
  - sentiment_score     : Positive_prob - Negative_prob  → [-1.0, +1.0]
  - sentiment_positive  : raw FinBERT positive probability
  - sentiment_negative  : raw FinBERT negative probability
  - sentiment_neutral   : raw FinBERT neutral probability

"""

import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[WARN] tqdm not found — install with: pip install tqdm")

#  Config
BASE         = Path(__file__).parent
ALIGNED      = BASE / "data" / "aligned"
PROCESSED    = BASE / "data" / "processed"
CHECKPOINTS  = BASE / "data" / "processed" / "checkpoints"
PROCESSED.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "ProsusAI/finbert"
BATCH_SIZE   = 16        # 16 is safe for MPS/CPU; bump to 32 if on CUDA
MAX_LENGTH   = 128       # FinBERT's comfortable token limit
CHECKPOINT_N = 500       # save progress every N batches

#  1. Device Selection 
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"  🔥 CUDA GPU detected: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  🍎 Apple Silicon MPS detected — using your M-chip GPU")
    else:
        device = torch.device("cpu")
        print(f"  ⚠️  No GPU found — falling back to CPU (will be slow)")
    return device

#  2. Load Model 
def load_finbert(device: torch.device):
    print(f"\n[LOADING] Downloading ProsusAI/finbert...")
    print("  (First run: ~440 MB download. Subsequent runs: cached locally.)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    # FinBERT label order: 0=positive, 1=negative, 2=neutral
    print(f"  Model labels: {model.config.id2label}")
    return tokenizer, model

#  3. Batch Inference
@torch.no_grad()
def run_batch(texts: list[str], tokenizer, model, device: torch.device) -> np.ndarray:
    """
    Returns np.ndarray of shape (N, 3): [positive_prob, negative_prob, neutral_prob]
    FinBERT id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    # Move tensors to device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    try:
        logits = model(**encoded).logits
    except RuntimeError as e:
        # MPS fallback: some ops not yet supported — run on CPU
        if "mps" in str(device):
            print("\n  [MPS FALLBACK] Switching this batch to CPU...")
            cpu = torch.device("cpu")
            model.to(cpu)
            encoded = {k: v.to(cpu) for k, v in encoded.items()}
            logits = model(**encoded).logits
            model.to(device)   # move back for next batch
        else:
            raise e

    probs = softmax(logits, dim=-1).cpu().numpy()  # shape (N, 3)
    return probs  # columns: [positive, negative, neutral]

#  4. Checkpoint Helpers
def save_checkpoint(results: dict, split: str, batch_idx: int):
    path = CHECKPOINTS / f"{split}_checkpoint.json"
    with open(path, "w") as f:
        json.dump({"batch_idx": batch_idx, "results": results}, f)

def load_checkpoint(split: str) -> tuple[dict, int]:
    path = CHECKPOINTS / f"{split}_checkpoint.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"  ♻️  Resuming {split} from batch {data['batch_idx']} "
              f"({len(data['results'])} rows already done)")
        return data["results"], data["batch_idx"]
    return {}, 0

def clear_checkpoint(split: str):
    path = CHECKPOINTS / f"{split}_checkpoint.json"
    if path.exists():
        path.unlink()

#  5. Main Inference Loop 
def infer_sentiment(df: pd.DataFrame, tokenizer, model, device: torch.device,
                    split: str) -> pd.DataFrame:
    """
    Fills sentiment columns in df. Bypasses FinBERT for empty rows.
    Returns the df with columns filled.
    """
    df = df.copy()

    # Separate active (has text) vs silent (empty) rows
    has_news = df["news_count"] > 0
    active_df = df[has_news].copy()
    silent_df  = df[~has_news].copy()

    n_active  = len(active_df)
    n_silent  = len(silent_df)
    n_total   = len(df)
    n_batches = (n_active + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n  Rows to score   : {n_active:,}  (news_count > 0)")
    print(f"  Neutral bypass  : {n_silent:,}  (news_count == 0) → hardcoded 0.0")
    print(f"  Total batches   : {n_batches:,}  (batch_size={BATCH_SIZE})")

    #  Apply neutral bypass immediately 
    for col, val in [("sentiment_score", 0.0), ("sentiment_positive", 0.0),
                     ("sentiment_negative", 0.0), ("sentiment_neutral", 1.0)]:
        df.loc[~has_news, col] = val

    #  Load checkpoint if resuming
    results_cache, start_batch = load_checkpoint(split)

    # Build index list for active rows
    active_indices = active_df.index.tolist()
    texts_all      = active_df["finbert_input"].fillna("").astype(str).tolist()

    #  Inference loop 
    start_time = time.time()
    iterator = range(start_batch, n_batches)

    if HAS_TQDM:
        iterator = tqdm(iterator, desc=f"  FinBERT [{split}]",
                        unit="batch", initial=start_batch, total=n_batches,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}, {rate_fmt}]")

    for batch_idx in iterator:
        i_start = batch_idx * BATCH_SIZE
        i_end   = min(i_start + BATCH_SIZE, n_active)

        batch_texts   = texts_all[i_start:i_end]
        batch_indices = active_indices[i_start:i_end]

        probs = run_batch(batch_texts, tokenizer, model, device)
        # probs columns: [positive, negative, neutral]

        for idx, prob_row in zip(batch_indices, probs):
            pos, neg, neu = float(prob_row[0]), float(prob_row[1]), float(prob_row[2])
            results_cache[str(idx)] = {
                "sentiment_score":    pos - neg,
                "sentiment_positive": pos,
                "sentiment_negative": neg,
                "sentiment_neutral":  neu
            }

        # Checkpoint every N batches
        if (batch_idx + 1) % CHECKPOINT_N == 0:
            save_checkpoint(results_cache, split, batch_idx + 1)
            if not HAS_TQDM:
                elapsed = time.time() - start_time
                pct = (batch_idx + 1) / n_batches * 100
                eta  = elapsed / (batch_idx - start_batch + 1) * (n_batches - batch_idx - 1)
                print(f"  [{pct:5.1f}%] Checkpoint saved | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    #  Write results back to df 
    for idx_str, vals in results_cache.items():
        idx = int(idx_str)
        for col, val in vals.items():
            df.at[idx, col] = val

    # Clear checkpoint on success
    clear_checkpoint(split)

    elapsed = time.time() - start_time
    print(f"\n  ✓ Inference complete in {elapsed/60:.1f} minutes")

    return df

#  6. Validation Summary ─
def print_sentiment_summary(df: pd.DataFrame, split: str):
    scored = df[df["news_count"] > 0]
    print(f"\n   {split.upper()} Sentiment Summary ")
    print(f"  Rows with news     : {len(scored):,}")
    print(f"  Score range        : [{scored['sentiment_score'].min():.3f}, {scored['sentiment_score'].max():.3f}]")
    print(f"  Mean score         : {scored['sentiment_score'].mean():.4f}")
    print(f"  Std dev            : {scored['sentiment_score'].std():.4f}")

    # Distribution buckets
    pos = (scored["sentiment_score"] >  0.1).sum()
    neg = (scored["sentiment_score"] < -0.1).sum()
    neu = len(scored) - pos - neg
    print(f"  Bullish  (>+0.1)  : {pos:,}  ({pos/len(scored):.1%})")
    print(f"  Neutral  (±0.1)   : {neu:,}  ({neu/len(scored):.1%})")
    print(f"  Bearish  (<-0.1)  : {neg:,}  ({neg/len(scored):.1%})")

    # Per-ticker breakdown
    print(f"\n  Per-ticker mean sentiment score:")
    for sym in ["NVDA", "TSLA", "JPM", "SPY"]:
        sym_scored = scored[scored["symbol"] == sym]
        if len(sym_scored) > 0:
            mean = sym_scored["sentiment_score"].mean()
            print(f"    {sym:5s}  {mean:+.4f}  ({len(sym_scored):,} scored hours)")

#  MAIN 
if __name__ == "__main__":
    print("  FinFlow 1.0 — Phase 2: FinBERT Sentiment Inference")

    device = get_device()

    # Adjust batch size for device
    if device.type == "cuda":
        BATCH_SIZE = 32
        print(f"  Using batch_size={BATCH_SIZE} (CUDA)")
    elif device.type == "mps":
        BATCH_SIZE = 16
        print(f"  Using batch_size={BATCH_SIZE} (MPS — conservative for stability)")
    else:
        BATCH_SIZE = 8
        print(f"  Using batch_size={BATCH_SIZE} (CPU — this will be slow)")

    tokenizer, model = load_finbert(device)

    for split, filename in [("train", "train_aligned.csv"),
                             ("test",  "test_aligned.csv")]:
        print(f"\n{'─'*60}")
        print(f"  Processing: {split.upper()} SET")
        print(f"{'─'*60}")

        df = pd.read_csv(ALIGNED / filename, parse_dates=["timestamp"])
        print(f"  Loaded {df.shape[0]:,} rows")

        df_out = infer_sentiment(df, tokenizer, model, device, split)
        print_sentiment_summary(df_out, split)

        out_path = PROCESSED / f"{split}_sentiment.csv"
        df_out.to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"\n  Saved → {out_path}  ({size_mb:.1f} MB)")

    print("\n" + "="*60)
    print("  PHASE 2 COMPLETE")
    print("""
  Output files:
    data/processed/train_sentiment.csv   ← 39 cols + 4 sentiment cols filled
    data/processed/test_sentiment.csv

  Columns added:
    sentiment_score     : Pos_prob - Neg_prob  →  range [-1.0, +1.0]
    sentiment_positive  : raw FinBERT P(positive)
    sentiment_negative  : raw FinBERT P(negative)
    sentiment_neutral   : raw FinBERT P(neutral)

  Empty hours (news_count == 0):
    sentiment_score = 0.0, sentiment_neutral = 1.0 (hardcoded bypass)

  Phase 3 next: Build LSTM, Prophet, Transformer baseline models.
""")
