"""
FinFlow 2.0 — Train CrossModalTransformer
Same dataset, same splits, same preprocessing as Phase 3.
Uses early stopping on validation weighted-F1.

"""
from __future__ import annotations
import sys, time, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finflow2.models.cross_modal_transformer import CrossModalTransformer
from finflow2.training.data_loader import (
    DualModalDataset, _build_aligned_tensors, load_cached, N_TEXT_FEAT
)
from sklearn.metrics import f1_score, accuracy_score, precision_score

CKPT = ROOT / "finflow2" / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)

EPOCHS         = int(os.environ.get("EPOCHS", "10"))
EPOCHS_PER_RUN = int(os.environ.get("EPOCHS_PER_RUN", str(EPOCHS)))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", "512"))
LR             = float(os.environ.get("LR", "5e-4"))
WEIGHT_DEC     = 1e-4
PATIENCE       = 5
RESUME         = os.environ.get("RESUME", "1") == "1"
TRAIN_SUBSET   = int(os.environ.get("TRAIN_SUBSET", "0"))   # 0 = use all

# Reproducibility
torch.manual_seed(42); np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if hasattr(torch.backends, "mps")
                                    and torch.backends.mps.is_available()
                                    else "cpu")
print(f"[cross_modal] device={device}  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
torch.set_num_threads(4)

# Data
print("[cross_modal] Loading cached tensors…")
splits = {s: load_cached(s) for s in ("train", "val", "test")}

if TRAIN_SUBSET > 0 and TRAIN_SUBSET < len(splits["train"]["y"]):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(splits["train"]["y"]), TRAIN_SUBSET, replace=False)
    for k in ("X_price", "X_text", "X_sent", "X_news", "y"):
        splits["train"][k] = splits["train"][k][idx]
    print(f"[cross_modal] subsampled train to {TRAIN_SUBSET:,}")

train_loader = DataLoader(DualModalDataset(splits["train"]),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(DualModalDataset(splits["val"]),
                          batch_size=BATCH_SIZE, shuffle=False)

n_price = splits["train"]["X_price"].shape[2]
n_text  = splits["train"]["X_text"].shape[2]
print(f"[cross_modal] N_train={len(splits['train']['y']):,}  "
      f"N_val={len(splits['val']['y']):,}  n_price={n_price}  n_text={n_text}")

# Class weights for imbalance
y_train = splits["train"]["y"]
counts  = np.bincount(y_train, minlength=3).astype(np.float32)
inv_freq = (counts.sum() / (3.0 * np.maximum(counts, 1)))
class_weights = torch.tensor(inv_freq, device=device)
print(f"[cross_modal] class_weights = {inv_freq}")

# Model
# Compact configuration for CPU training; the architecture scales, but
# the ~25k-sample dataset doesn't benefit from a deeper Transformer.
model = CrossModalTransformer(
    n_price_feat=n_price, n_text_feat=n_text,
    d_model=32, n_heads=4,
    n_self_layers=1, n_cross_layers=1,
    d_ff=96, dropout=0.15,
).to(device)
print(f"[cross_modal] params = {model.count_parameters():,}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Training loop (resumable)
state_path = CKPT / "cross_modal_runstate.pt"
hist_path  = CKPT / "cross_modal_history.csv"

start_epoch = 1
hist = []
best_f1 = -1.0; bad_epochs = 0

if RESUME and state_path.exists():
    blob = torch.load(state_path, map_location=device)
    model.load_state_dict(blob["model"])
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    start_epoch = blob["epoch"] + 1
    best_f1     = blob["best_f1"]
    bad_epochs  = blob["bad_epochs"]
    if hist_path.exists():
        hist = pd.read_csv(hist_path).to_dict("records")
    print(f"[cross_modal] RESUMING from epoch {start_epoch}  (best_f1={best_f1:.4f})")

end_epoch = min(EPOCHS, start_epoch + EPOCHS_PER_RUN - 1)
t_total = time.time()

for ep in range(start_epoch, end_epoch + 1):
    model.train(); ep_loss = 0; n_seen = 0; t0 = time.time()
    for p, t, _s, _n, y in train_loader:
        p = p.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(p, t)
        loss   = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item() * p.size(0); n_seen += p.size(0)

    # Validation
    model.eval(); val_preds = []; val_true = []
    with torch.no_grad():
        for p, t, _s, _n, y in val_loader:
            p = p.to(device); t = t.to(device)
            logits = model(p, t)
            val_preds.append(logits.argmax(-1).cpu().numpy())
            val_true.append(y.numpy())
    val_pred = np.concatenate(val_preds); val_true = np.concatenate(val_true)
    val_acc  = accuracy_score(val_true, val_pred)
    val_f1   = f1_score(val_true, val_pred, average="weighted", zero_division=0)
    val_prec_buy  = precision_score(val_true, val_pred, labels=[2], average="macro", zero_division=0)
    val_prec_sell = precision_score(val_true, val_pred, labels=[0], average="macro", zero_division=0)

    # Average gate value (sentiment usage)
    gate_avg = float(model.last_gates[-1].mean().item()) if model.last_gates else float("nan")

    hist.append({
        "epoch": ep, "train_loss": ep_loss/n_seen,
        "val_acc": val_acc, "val_f1": val_f1,
        "val_buy_prec": val_prec_buy, "val_sell_prec": val_prec_sell,
        "gate_avg": gate_avg, "lr": opt.param_groups[0]["lr"],
        "time_s": time.time()-t0,
    })
    sched.step()

    is_best = val_f1 > best_f1
    if is_best:
        best_f1 = val_f1
        torch.save({"state_dict": model.state_dict(),
                    "config": {"n_price_feat": n_price, "n_text_feat": n_text,
                                "d_model": 32, "n_heads": 4,
                                "n_self_layers": 1, "n_cross_layers": 1,
                                "d_ff": 96, "dropout": 0.15}},
                   CKPT / "cross_modal_best.pt")
        bad_epochs = 0
    else:
        bad_epochs += 1

    print(f"[cross_modal] ep {ep:2d}/{EPOCHS}  loss={ep_loss/n_seen:.4f}  "
          f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
          f"buyP={val_prec_buy:.3f} sellP={val_prec_sell:.3f}  "
          f"gate={gate_avg:.3f}  t={time.time()-t0:.1f}s  "
          f"{'★best' if is_best else ''}")

    pd.DataFrame(hist).to_csv(hist_path, index=False)

    # Save resume state every epoch
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "epoch": ep,
                "best_f1": best_f1, "bad_epochs": bad_epochs}, state_path)

    if bad_epochs >= PATIENCE:
        print(f"[cross_modal] Early stopping (no val_f1 improvement for {PATIENCE} epochs)")
        break

is_done = (ep >= EPOCHS) or (bad_epochs >= PATIENCE)
with open(CKPT / "cross_modal_status.json", "w") as f:
    json.dump({"status": "done" if is_done else "in_progress",
               "best_val_f1": best_f1,
               "last_epoch": ep,
               "total_epochs": EPOCHS,
               "total_time_s": time.time() - t_total}, f, indent=2)

print(f"[cross_modal] {'DONE' if is_done else f'PAUSED at ep {ep}'}   "
      f"best_val_f1 = {best_f1:.4f}   "
      f"chunk_time = {time.time() - t_total:.1f}s")
