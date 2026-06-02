"""
FinFlow 2.0 — Precompute & cache aligned tensors
Builds (X_price, X_text, X_sent, X_news, y) per split once and saves
to disk so training scripts start instantly.

"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finflow2.training.data_loader import _build_aligned_tensors

CACHE = ROOT / "finflow2" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

def main():
    for split in ("train", "val", "test"):
        t0 = time.time()
        d = _build_aligned_tensors(split)
        np.save(CACHE / f"{split}_X_price.npy", d["X_price"])
        np.save(CACHE / f"{split}_X_text.npy",  d["X_text"])
        np.save(CACHE / f"{split}_X_sent.npy",  d["X_sent"])
        np.save(CACHE / f"{split}_X_news.npy",  d["X_news"])
        np.save(CACHE / f"{split}_y.npy",       d["y"])
        d["meta"].to_csv(CACHE / f"{split}_meta.csv", index=False)
        print(f"  {split}: N={len(d['y']):,}  built in {time.time()-t0:.1f}s")
    print(f"\n  Cached → {CACHE}")

if __name__ == "__main__":
    main()
