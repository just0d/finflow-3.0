"""
FinFlow 1.0 — Phase 3 Step 3: Train All Baselines & Evaluate
Trains Prophet, LSTM, and Transformer on the 2023-2024 training window.
Evaluates all three on the UNSEEN 2025 test set.
Crowns a "Draft Pick" for Phase 4 (Late-Fusion NLP integration).

Run order:
    python3 phase3_preprocessing.py create tensors first
    python3 phase3_train_evaluate.py tis script

"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, f1_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from phase3_models import LSTMClassifier, TransformerClassifier

BASE    = Path(__file__).parent
PREP    = BASE / "data" / "preprocessed"
ALIGNED = BASE / "data" / "aligned"
MODELS  = BASE / "models"; MODELS.mkdir(exist_ok=True)
OUTPUTS = BASE / "outputs"; OUTPUTS.mkdir(exist_ok=True)
CHARTS  = BASE / "outputs" / "charts"; CHARTS.mkdir(exist_ok=True)

# Bloomberg palette
BG, PANEL, BORDER = "#0a0a0f", "#0f1117", "#1e2330"
TEXT, MUTED = "#c8cdd6", "#5a6070"
ORANGE, BLUE, GREEN, RED, AMBER, PURPLE = (
    "#f97316", "#3b82f6", "#22c55e", "#ef4444", "#f59e0b", "#a78bfa"
)
MODEL_COLORS = {"LSTM": BLUE, "Transformer": ORANGE, "Prophet": GREEN}

CFG = {
    "lstm": {
        "hidden_size":  128,
        "num_layers":   2,
        "dropout":      0.3,
        "lr":           1e-3,
        "weight_decay": 1e-4,
        "epochs":       60,
        "batch_size":   256,
        "patience":     12,
    },
    "transformer": {
        "d_model":      64,
        "nhead":        4,
        "num_layers":   2,
        "dim_ff":       256,
        "dropout":      0.1,
        "lr":           5e-4,
        "weight_decay": 1e-4,
        "epochs":       60,
        "batch_size":   256,
        "patience":     12,
    },
}

LABEL_NAMES  = {0: "SELL", 1: "HOLD", 2: "BUY"}
TARGET_TICKERS = ["NVDA", "TSLA", "JPM", "SPY"]

# UTILITIES

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
        print(f"  Device: MPS (Apple Silicon)")
    else:
        d = torch.device("cpu")
        print(f"  Device: CPU")
    return d

def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights to help with slight imbalance."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = len(y) / (n_classes * counts)
    return torch.FloatTensor(weights)

def load_tensors():
    print("[LOAD] Reading preprocessed tensors...")
    X_train = np.load(PREP / "X_train.npy")
    y_train = np.load(PREP / "y_train.npy")
    X_val   = np.load(PREP / "X_val.npy")
    y_val   = np.load(PREP / "y_val.npy")
    X_test  = np.load(PREP / "X_test.npy")
    y_test  = np.load(PREP / "y_test.npy")
    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# NEURAL NETWORK TRAINING LOOP (shared by LSTM and Transformer)

def train_nn(
    model_name: str,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    device:  torch.device,
    cfg:     dict,
) -> tuple[nn.Module, dict]:
    """
    Full training loop with:
      - CrossEntropyLoss with class weighting
      - Adam + ReduceLROnPlateau scheduler
      - Early stopping (saves best checkpoint by val loss)
    Returns the best model and a history dict.
    """
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"\n{'─'*60}")
    print(f"  Training {model_name}  ({n_params:,} parameters)")
    print(f"{'─'*60}")

    # DataLoaders
    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    Xt = torch.FloatTensor(X_train)
    yt = torch.LongTensor(y_train)
    Xv = torch.FloatTensor(X_val)
    yv = torch.LongTensor(y_val)

    train_loader = DataLoader(TensorDataset(Xt, yt),
                              batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(TensorDataset(Xv, yv),
                              batch_size=cfg["batch_size"] * 2, shuffle=False,
                              num_workers=0)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(yb)
            train_n += len(yb)
        train_loss = train_loss_sum / train_n

        # Validate
        model.eval()
        val_loss_sum, val_n, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss   = criterion(logits, yb)
                val_loss_sum += loss.item() * len(yb)
                val_n += len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()

        val_loss = val_loss_sum / val_n
        val_acc  = val_correct / val_n
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss    = val_loss
            patience_counter = 0
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}/{cfg['epochs']} | "
                  f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
                  f"ValAcc={val_acc:.2%}  LR={lr_now:.2e}  [{elapsed:.0f}s]")

        if patience_counter >= cfg["patience"]:
            print(f"  Early stop at epoch {epoch} (patience={cfg['patience']})")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODELS / f"{model_name.lower()}_best.pt")
    print(f"  Best val loss: {best_val_loss:.4f}  |  Saved → models/{model_name.lower()}_best.pt")
    return model, history

def predict_nn(model: nn.Module, X: np.ndarray, device: torch.device,
               batch_size: int = 512) -> np.ndarray:
    """Return class predictions (argmax) for the full array X."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X)),
        batch_size=batch_size, shuffle=False
    )
    preds = []
    with torch.no_grad():
        for (Xb,) in loader:
            logits = model(Xb.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)

# PROPHET MODEL — PER-TICKER

def run_prophet(X_train_raw: np.ndarray, y_test: np.ndarray,
                meta_test: pd.DataFrame) -> np.ndarray:
    """
    Trains one Prophet model per ticker on close prices.
    Makes 1-step-ahead predictions on the test window.

    How it maps to 3-class:
      - Forecast return > +0.1%  → BUY  (label=2)
      - Forecast return < -0.1%  → SELL (label=0)
      - Otherwise               → HOLD (label=1)
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("  [PROPHET] Install with: pip install prophet")
        return np.ones(len(y_test), dtype=np.int64)  # all HOLD as placeholder

    print(f"\n{'─'*60}")
    print(f"  Training Prophet (1 model per ticker)")
    print(f"{'─'*60}")

    # Load raw aligned training data for close prices
    aligned_path = None
    for p in [BASE / "data" / "processed" / "train_sentiment.csv",
               BASE / "data" / "aligned"  / "train_aligned.csv"]:
        if p.exists():
            aligned_path = p
            break

    if aligned_path is None:
        print("  [ERROR] Cannot find training data for Prophet")
        return np.ones(len(y_test), dtype=np.int64)

    train_df = pd.read_csv(aligned_path, parse_dates=["timestamp"])
    if train_df["timestamp"].dt.tz is not None:
        train_df["timestamp"] = train_df["timestamp"].dt.tz_localize(None)

    test_df_path = None
    for p in [BASE / "data" / "processed" / "test_sentiment.csv",
               BASE / "data" / "aligned"  / "test_aligned.csv"]:
        if p.exists():
            test_df_path = p
            break
    test_df = pd.read_csv(test_df_path, parse_dates=["timestamp"])
    if test_df["timestamp"].dt.tz is not None:
        test_df["timestamp"] = test_df["timestamp"].dt.tz_localize(None)

    all_prophet_preds = {}   # symbol → {timestamp: predicted_class}
    THRESHOLD = 0.001

    for sym in TARGET_TICKERS:
        print(f"  {sym}: fitting Prophet...", end=" ", flush=True)
        sym_train = (train_df[train_df["symbol"] == sym]
                     .sort_values("timestamp")[["timestamp", "close"]]
                     .rename(columns={"timestamp": "ds", "close": "y"})
                     .dropna())

        sym_test  = (test_df[test_df["symbol"] == sym]
                     .sort_values("timestamp")[["timestamp", "close"]]
                     .rename(columns={"timestamp": "ds", "close": "y"})
                     .dropna())

        if len(sym_train) < 100:
            print(f"  [{sym}] Too few rows — using HOLD")
            for ts in sym_test["ds"]:
                all_prophet_preds[(sym, ts)] = 1
            continue

        try:
            m = Prophet(
                changepoint_prior_scale = 0.05,
                seasonality_mode        = "multiplicative",
                daily_seasonality       = True,
                weekly_seasonality      = True,
                yearly_seasonality      = False,
                uncertainty_samples     = 0,           # skip uncertainty intervals → 10x faster
            )
            # Suppress cmdstanpy output
            import logging
            logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
            logging.getLogger("prophet").setLevel(logging.WARNING)

            m.fit(sym_train)

            # Forecast the full test window + 1 step
            future = m.make_future_dataframe(periods=len(sym_test) + 1, freq="h",
                                              include_history=False)
            forecast = m.predict(future)

            # Build a lookup: for each test timestamp t, predict_return = (yhat_t+1 - actual_t) / actual_t
            actual_close = sym_test.set_index("ds")["y"]
            forecast_close = forecast.set_index("ds")["yhat"]

            sym_preds = {}
            for i, row in sym_test.iterrows():
                ts = row["ds"]
                actual_price = row["y"]
                # Get Prophet's forecast for the NEXT timestamp after ts
                try:
                    future_prices = forecast_close[forecast_close.index > ts]
                    if len(future_prices) == 0:
                        sym_preds[ts] = 1  # HOLD
                        continue
                    next_yhat = future_prices.iloc[0]
                    pred_return = (next_yhat - actual_price) / actual_price
                    if pred_return > THRESHOLD:
                        sym_preds[ts] = 2  # BUY
                    elif pred_return < -THRESHOLD:
                        sym_preds[ts] = 0  # SELL
                    else:
                        sym_preds[ts] = 1  # HOLD
                except Exception:
                    sym_preds[ts] = 1  # HOLD

            all_prophet_preds[sym] = sym_preds
            n_buy  = sum(1 for v in sym_preds.values() if v == 2)
            n_sell = sum(1 for v in sym_preds.values() if v == 0)
            print(f"done | BUY={n_buy} SELL={n_sell} HOLD={len(sym_preds)-n_buy-n_sell}")

        except Exception as e:
            print(f"  [ERROR] Prophet failed for {sym}: {e}")
            all_prophet_preds[sym] = {}

    # Align with meta_test sequence index
    meta = pd.read_csv(PREP / "meta_test.csv", parse_dates=["timestamp"])
    if meta["timestamp"].dt.tz is not None:
        meta["timestamp"] = meta["timestamp"].dt.tz_localize(None)

    prophet_preds = np.ones(len(meta), dtype=np.int64)  # default HOLD
    for idx, row in meta.iterrows():
        sym = row["symbol"]
        ts  = pd.Timestamp(row["timestamp"])
        if sym in all_prophet_preds and isinstance(all_prophet_preds[sym], dict):
            pred = all_prophet_preds[sym].get(ts, 1)
            prophet_preds[idx] = pred

    return prophet_preds

# EVALUATION & REPORTING

def evaluate_model(name: str, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute classification metrics. Returns summary dict."""
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                    target_names=["SELL(0)", "HOLD(1)", "BUY(2)"],
                                    output_dict=True)
    cm     = confusion_matrix(y_true, y_pred)

    # Extract key metrics
    buy_precision  = report.get("BUY(2)", {}).get("precision", 0.0)
    sell_precision = report.get("SELL(0)", {}).get("precision", 0.0)
    buy_f1         = report.get("BUY(2)", {}).get("f1-score", 0.0)
    sell_f1        = report.get("SELL(0)", {}).get("f1-score", 0.0)
    weighted_f1    = report.get("weighted avg", {}).get("f1-score", 0.0)

    print(f"\n  ── {name} Test Set Results ─────────────────────")
    print(f"  Overall Accuracy : {acc:.2%}")
    print(f"  BUY  Precision   : {buy_precision:.3f}  F1: {buy_f1:.3f}")
    print(f"  SELL Precision   : {sell_precision:.3f}  F1: {sell_f1:.3f}")
    print(f"  Weighted F1      : {weighted_f1:.3f}")
    print()
    print(classification_report(y_true, y_pred,
                                  target_names=["SELL(0)", "HOLD(1)", "BUY(2)"]))
    return {
        "model":           name,
        "accuracy":        round(acc,           4),
        "buy_precision":   round(buy_precision, 4),
        "sell_precision":  round(sell_precision,4),
        "buy_f1":          round(buy_f1,        4),
        "sell_f1":         round(sell_f1,       4),
        "weighted_f1":     round(weighted_f1,   4),
        "confusion_matrix": cm,
    }

def plot_training_curves(histories: dict):
    """Bloomberg-style training curves for LSTM and Transformer."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("FIGURE P3-A  |  Phase 3  |  Training Curves",
                 color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

    for ax in axes.flat:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, lw=0.5, alpha=0.7, ls="--")
        ax.set_axisbelow(True)

    subplot_data = [
        (axes[0,0], "LSTM",        "train_loss", "Train Loss",       BLUE),
        (axes[0,1], "Transformer", "train_loss", "Train Loss",       ORANGE),
        (axes[1,0], "LSTM",        "val_acc",    "Val Accuracy (%)", BLUE),
        (axes[1,1], "Transformer", "val_acc",    "Val Accuracy (%)", ORANGE),
    ]

    for ax, model_name, key, ylabel, color in subplot_data:
        if model_name not in histories:
            ax.text(0.5, 0.5, f"{model_name}\nnot trained", ha="center",
                    va="center", transform=ax.transAxes, color=MUTED)
            continue
        data = histories[model_name][key]
        epochs = range(1, len(data) + 1)

        if "loss" in key:
            val_data = histories[model_name]["val_loss"]
            ax.plot(epochs, data,     color=color,  lw=1.5, label="Train Loss", alpha=0.9)
            ax.plot(epochs, val_data, color=RED,    lw=1.5, label="Val Loss",   linestyle="--")
            # Mark best val loss
            best_ep = int(np.argmin(val_data)) + 1
            ax.axvline(best_ep, color=GREEN, lw=1, ls=":", alpha=0.7,
                       label=f"Best ep={best_ep}")
        else:
            ax.plot(epochs, [v * 100 for v in data], color=color, lw=1.5)
            # Reference lines
            ax.axhline(52, color=MUTED, lw=0.8, ls=":", alpha=0.5, label="52% (profitable)")
            ax.axhline(60, color=RED,   lw=0.8, ls=":", alpha=0.5, label="60% (overfit alert)")

        ax.set_title(f"{model_name} — {ylabel}", color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

    plt.tight_layout()
    out_path = CHARTS / "phase3_training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved → {out_path}")

def plot_confusion_matrices(results: list[dict]):
    """Bloomberg-style confusion matrices for all three models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("FIGURE P3-B  |  Phase 3  |  Confusion Matrices — 2025 Test Set",
                 color=TEXT, fontsize=13, fontweight="bold", x=0.02, ha="left")

    if n == 1:
        axes = [axes]

    label_names = ["SELL", "HOLD", "BUY"]

    for ax, res in zip(axes, results):
        color   = MODEL_COLORS.get(res["model"], BLUE)
        cm      = res["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        ax.set_facecolor(PANEL)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("bb", [PANEL, color], N=256)
        im   = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(
            colors=MUTED, labelsize=7
        )

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(label_names, color=TEXT, fontsize=10, fontweight="bold")
        ax.set_yticklabels(label_names, color=TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", color=TEXT, fontsize=10)
        ax.set_ylabel("Actual",    color=TEXT, fontsize=10)
        ax.set_title(
            f"{res['model']}\nAcc={res['accuracy']:.2%}  "
            f"BUY-P={res['buy_precision']:.3f}",
            color=color, fontsize=10, fontweight="bold", pad=8
        )

        for i in range(3):
            for j in range(3):
                ax.text(j, i,
                        f"{cm[i,j]:,}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[i,j] < 0.6 else "black",
                        fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    out_path = CHARTS / "phase3_confusion_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved → {out_path}")

def print_final_report(results: list[dict]):
    """Print the comparison table and crown the Draft Pick."""
    print("\n" + "═"*70)
    print("  PHASE 3 FINAL COMPARISON TABLE — 2025 UNSEEN TEST SET")
    print("═"*70)

    header = f"  {'Model':15s} {'Accuracy':>10s} {'BUY-Prec':>10s} {'SELL-Prec':>10s} {'Buy-F1':>8s} {'W-F1':>8s}"
    print(header)
    print("  " + "─"*65)

    df_rows = []
    for r in results:
        line = (f"  {r['model']:15s} "
                f"{r['accuracy']:>9.2%} "
                f"{r['buy_precision']:>9.3f} "
                f"{r['sell_precision']:>10.3f} "
                f"{r['buy_f1']:>7.3f} "
                f"{r['weighted_f1']:>7.3f}")
        print(line)
        df_rows.append(r)

    # Score = BUY precision + SELL precision + weighted_f1 (higher = better)
    def score(r):
        return (r["buy_precision"] + r["sell_precision"] + r["weighted_f1"]) / 3

    best = max(results, key=score)

    print("\n" + "═"*70)
    print(f"  🏆  DRAFT PICK: {best['model']}")
    print(f"      Accuracy={best['accuracy']:.2%}  BUY-P={best['buy_precision']:.3f}  "
          f"SELL-P={best['sell_precision']:.3f}  W-F1={best['weighted_f1']:.3f}")
    print("═"*70)
    print(f"""
  What this means for Phase 4:
  {best['model']} will carry into Phase 4 Late-Fusion.

  The Late-Fusion Meta-Rule:
    Step 1  {best['model']} predicts a trade signal (BUY / SELL).
    Step 2  FinBERT checks the sentiment_score for that exact hour.
    Step 3  If sentiment_score ≤ -0.50 → veto to HOLD (emergency brake).
            If sentiment_score ≥ +0.50 → amplify position (conviction boost).

  Academic note on accuracy: ~33-36% accuracy on a balanced 3-class problem
  is near random chance. Anything above ~38% with consistent BUY/SELL
  precision is genuinely useful. For a neural net on raw price data alone,
  52-54% is excellent. >60% almost always indicates overfitting.
    """)

    return best["model"]

# MAIN

if __name__ == "__main__":
    print("  FinFlow 1.0 — Phase 3: Training All Baselines")

    device = get_device()
    X_train, y_train, X_val, y_val, X_test, y_test = load_tensors()
    n_features = X_train.shape[2]
    print(f"  Features per timestep: {n_features}")

    histories = {}
    results   = []

    # MODEL 1: LSTM
    lstm = LSTMClassifier(
        input_size  = n_features,
        hidden_size = CFG["lstm"]["hidden_size"],
        num_layers  = CFG["lstm"]["num_layers"],
        dropout     = CFG["lstm"]["dropout"],
    )
    lstm, lstm_history = train_nn(
        "LSTM", lstm, X_train, y_train, X_val, y_val, device, CFG["lstm"]
    )
    histories["LSTM"] = lstm_history
    lstm_preds = predict_nn(lstm, X_test, device)
    results.append(evaluate_model("LSTM", lstm_preds, y_test))

    # MODEL 2: TRANSFORMER
    transformer = TransformerClassifier(
        input_size      = n_features,
        d_model         = CFG["transformer"]["d_model"],
        nhead           = CFG["transformer"]["nhead"],
        num_layers      = CFG["transformer"]["num_layers"],
        dim_feedforward = CFG["transformer"]["dim_ff"],
        dropout         = CFG["transformer"]["dropout"],
    )
    transformer, tf_history = train_nn(
        "Transformer", transformer,
        X_train, y_train, X_val, y_val, device, CFG["transformer"]
    )
    histories["Transformer"] = tf_history
    tf_preds = predict_nn(transformer, X_test, device)
    results.append(evaluate_model("Transformer", tf_preds, y_test))

    # MODEL 3: PROPHET
    prophet_preds = run_prophet(X_train, y_test, None)
    results.append(evaluate_model("Prophet", prophet_preds, y_test))

    # VISUALIZATIONS
    print("\n[CHARTS] Generating visualizations...")
    plot_training_curves(histories)
    plot_confusion_matrices(results)

    # FINAL REPORT
    draft_pick = print_final_report(results)

    # Save results CSV
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "confusion_matrix"}
        for r in results
    ])
    results_df["draft_pick"] = results_df["model"] == draft_pick
    results_df.to_csv(OUTPUTS / "phase3_results.csv", index=False)
    print(f"\n  Results saved → outputs/phase3_results.csv")
    print(f"\n  ✓ Phase 3 complete. Draft Pick: {draft_pick}")
    print(f"  Next: phase4_late_fusion.py — NLP override layer")
