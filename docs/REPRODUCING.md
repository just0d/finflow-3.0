# Reproducing FinFlow

This document gives the **exact** sequence of commands required to reproduce
every artefact in `outputs/` and `finflow2/checkpoints/` from a fresh checkout.

All seven models are evaluated on the **same** 2025 held-out window
(Jan 2025 – Jan 2026, 15,395 hourly windows), with the **same** preprocessing
and the **same** event-driven backtester (`phase5_backtester.py`, unchanged
between FinFlow 1.0 and 2.0).

---

## 0 — Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10. Apple-silicon (`mps`), CUDA, and CPU all work; the trainers
auto-detect the device.

## 1 — Data alignment & feature engineering

```bash
python phase1_alignment.py
```
Outputs `data/processed/aligned_master.csv` with 30,806 hourly bars × 39 columns.

## 2 — FinBERT scoring (≈15 min on M-chip / batch_size=16)

```bash
python phase2_finbert.py
```
Outputs `data/processed/{train,test}_sentiment.csv` covering all 46,395
news records.

## 3 — Build 24-hour tensors

```bash
python phase3_preprocessing.py
```
Outputs `data/preprocessed/X_{train,val,test}.npy`, `y_*.npy`, `meta_*.csv`.

## 4 — Train math baselines (LSTM, Transformer, Prophet)

```bash
python phase3_train_evaluate.py
```
Outputs `models/lstm_best.pt`, `models/transformer_best.pt`,
`phase3_outputs/phase3_results.csv`.

## 5 — Late-fusion fused signals

```bash
python phase4_late_fusion.py
```
Outputs `outputs/fused_signals_{lstm,transformer,buyandhold}.csv`.

## 6 — Backtest the FinFlow 1.0 strategies

```bash
python phase5_backtester.py
```
Outputs `outputs/backtest_results.json`, `outputs/backtest_summary.csv`.

---

## 7 — Train the FinFlow research extension

The two new architectures share a single aligned-tensor cache. Build it once:

```bash
python finflow2/training/precompute_tensors.py
```

Then train each model. The trainers are **resumable**: pass `EPOCHS_PER_RUN`
small enough to fit your environment's wall-clock limit, and re-launch with
`RESUME=1` to continue.

```bash
EPOCHS=6 EPOCHS_PER_RUN=2 BATCH_SIZE=256 RESUME=1 \
    python finflow2/training/train_cross_modal.py

EPOCHS=6 EPOCHS_PER_RUN=2 BATCH_SIZE=256 RESUME=1 \
    python finflow2/training/train_sentiment_lstm.py
```

Outputs `finflow2/checkpoints/{cross_modal,sentiment_lstm}_best.pt` plus
training-history CSVs.

## 8 — Generate FinFlow 2.0 signals

```bash
python finflow2/evaluation/generate_signals.py
```
Outputs `outputs/fused_signals_{crossmodal,sentlstm}.csv`.

## 9 — Backtest the new models through the unchanged engine

```bash
python finflow2/backtesting/run_backtests.py
```
Outputs `outputs/finflow2_backtest_summary.csv`.

## 10 — Unified comparison report

```bash
python finflow2/evaluation/compare_all_models.py
```
Outputs:

* `outputs/finflow2_comparison_table.csv`
* `outputs/finflow2_comparison_ranked.csv`
* `outputs/finflow2_comparison_report.md`

These three files contain the headline result table you see in
`docs/FinFlow_Final_Report.docx`.

---

## Running everything in Google Colab

Every step above has a paired notebook in `notebooks/`. The recommended order:

1. `01_phase1_alignment.ipynb`
2. `02_phase2_finbert.ipynb`        *(GPU runtime recommended)*
3. `03_phase3_preprocessing.ipynb`
4. `04_phase3_train_evaluate.ipynb` *(GPU runtime recommended)*
5. `05_phase4_late_fusion.ipynb`
6. `06_phase5_backtester.ipynb`
7. `07_finflow2_precompute_tensors.ipynb`
8. `08_finflow2_train_cross_modal.ipynb`     *(GPU)*
9. `09_finflow2_train_sentiment_lstm.ipynb`  *(GPU)*
10. `10_finflow2_generate_signals.ipynb`
11. `11_finflow2_run_backtests.ipynb`
12. `12_finflow2_compare_all_models.ipynb`

Each notebook self-installs its dependencies and mounts Drive to access the
repo folder.
