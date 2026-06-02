# Running FinFlow End-to-End on Google Colab

A practical, ordered checklist for reproducing every result in the project from a
fresh Colab account. Total wall-clock time on a free T4 GPU: **roughly 75–90
minutes** if you start from raw data, or **under 5 minutes** if you only want to
re-build the final comparison from already-trained checkpoints.

---

## Step 0 — One-time setup (do this once, before any notebook)

1. **Upload the repo to Google Drive.**
   Drag the entire `FinFlow/` folder into `My Drive`. The exact target path
   every notebook expects is:

   ```
   /content/drive/MyDrive/FinFlow/
   ```

   The folder must contain the original `data/`, `models/`, `phase*.py`,
   `finflow2/`, and `notebooks/` subtree.

2. **Open any notebook from `notebooks/` in Colab.**
   Right-click the `.ipynb` file in Drive → *Open with → Google Colaboratory*.

3. **Pick the right runtime.** *Runtime → Change runtime type*:
   - **GPU (T4)** — required for `02`, `04`, `08`, `09`. Recommended for the
     whole pipeline; everything still runs on CPU but training is 10–30× slower.
   - **High-RAM** — only needed if you scale up the FinFlow-2 trainers
     (e.g., remove the compact configs).

4. **Mount Drive once.** Each notebook has a Drive-mount cell at the top; the
   first time you run it Colab will pop up a permission dialog.

---

## The 12-step pipeline — order, runtime, time, outputs

Run the notebooks in numerical order. Every step writes its outputs to disk, so
you can stop and restart any time without losing progress.

| #  | Notebook                                  | Runtime | ≈ Time | Reads                                                        | Writes (key outputs)                                                                      |
|---:|-------------------------------------------|---------|-------:|--------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 01 | `01_phase1_alignment.ipynb`               | CPU     |  5 min | `data/raw/*.csv` (OHLCV + news)                              | `data/processed/aligned_master.csv` (30,806 hourly bars × 39 cols)                        |
| 02 | `02_phase2_finbert.ipynb`                 | **GPU** | 15 min | aligned master, raw news                                     | `data/processed/train_sentiment.csv`, `data/processed/test_sentiment.csv` (46,395 records)|
| 03 | `03_phase3_preprocessing.ipynb`           | CPU     |  2 min | aligned master                                               | `data/preprocessed/X_{train,val,test}.npy`, `y_*.npy`, `meta_*.csv`                       |
| 04 | `04_phase3_train_evaluate.ipynb`          | **GPU** | 25 min | preprocessed tensors                                         | `models/lstm_best.pt`, `models/transformer_best.pt`, `phase3_outputs/phase3_results.csv`  |
| 05 | `05_phase4_late_fusion.ipynb`             | CPU     |  2 min | trained models + sentiment CSVs                              | `outputs/fused_signals_lstm.csv`, `outputs/fused_signals_transformer.csv`, `outputs/fused_signals_buyandhold.csv` |
| 06 | `06_phase5_backtester.ipynb`              | CPU     |  1 min | fused signals                                                | `outputs/backtest_results.json`, `outputs/backtest_summary.csv`                           |
| 07 | `07_finflow2_precompute_tensors.ipynb`    | CPU     |  3 min | preprocessed tensors + sentiment CSVs                        | `finflow2/cache/{train,val,test}_X_{price,text,sent,news}.npy`                            |
| 08 | `08_finflow2_train_cross_modal.ipynb`     | **GPU** |  8 min | finflow2 cache                                               | `finflow2/checkpoints/cross_modal_best.pt` + history CSV                                  |
| 09 | `09_finflow2_train_sentiment_lstm.ipynb`  | **GPU** |  8 min | finflow2 cache                                               | `finflow2/checkpoints/sentiment_lstm_best.pt` + history CSV                               |
| 10 | `10_finflow2_generate_signals.ipynb`      | CPU     |  2 min | both new checkpoints + cache                                 | `outputs/fused_signals_crossmodal.csv`, `outputs/fused_signals_sentlstm.csv`              |
| 11 | `11_finflow2_run_backtests.ipynb`         | CPU     |  1 min | new fused signals                                            | `outputs/finflow2_backtest_summary.csv`                                                   |
| 12 | `12_finflow2_compare_all_models.ipynb`    | CPU     | < 1 min| every backtest summary + every classification result         | `outputs/finflow2_comparison_table.csv`, `outputs/finflow2_comparison_ranked.csv`, `outputs/finflow2_comparison_report.md` |

---

## Three ways to run the project

### Path A — Cold start (reproduce everything from scratch)
Run every notebook **01 → 12** in order. Total: ~75 min on T4.

### Path B — Skip training, reproduce only the final analysis
The `.pt` checkpoints in `models/` and `finflow2/checkpoints/` are already in
the repo. If you only want to regenerate signals, backtests, and the unified
comparison, run **05, 06, 07, 10, 11, 12** in that order. Total: ~10 min, CPU
only.

### Path C — One-section reproduction
Each notebook is independent given its inputs exist on Drive. You can re-run
any single phase (e.g., re-tune the late-fusion thresholds in `phase4`) and
then run the downstream notebooks (`06` and onward) without touching the
upstream ones.

---

## Verification checkpoints

After each milestone you can spot-check that the pipeline is on the right track:

- **After `01`** — `data/processed/aligned_master.csv` should have ~30,806 rows.
- **After `02`** — `train_sentiment.csv` rows ≈ 30,806; sentiment_score range
  should span roughly [-0.97, +0.94].
- **After `03`** — `X_train.npy.shape == (25441, 24, 19)`,
  `X_val.npy.shape == (5172, 24, 19)`, `X_test.npy.shape == (15395, 24, 19)`.
- **After `04`** — `phase3_outputs/phase3_results.csv` should show LSTM
  weighted-F1 ≈ 0.458 and Transformer ≈ 0.448.
- **After `06`** — `outputs/backtest_summary.csv` should list four strategies
  (LSTM raw, LSTM+NLP, Transformer raw, Transformer+NLP) plus Buy-and-Hold.
- **After `12`** — `outputs/finflow2_comparison_report.md` should contain the
  full 7-model ranked table; the top-ranked composite should be
  *Transformer + Late Fusion* (≈ 0.404), with *SentimentLSTM* in second.

If any of these break, re-run the previous notebook before continuing — Colab
sometimes truncates uploads silently.

---

## Tips and pitfalls

- **Resumable trainers.** Notebooks `08` and `09` set `RESUME=1` by default, so
  if the Colab session times out mid-training, just re-run the same notebook
  cell — it will pick up from the last epoch checkpoint in
  `finflow2/checkpoints/*_runstate.pt`.

- **Free Colab disconnects after ~12 hours / 90 min idle.** Nothing in the
  pipeline takes that long, but if you do hit a disconnect, the `.npy` caches
  and `.pt` checkpoints survive on Drive — just re-run the affected notebook.

- **GPU not strictly required for steps 04, 08, 09.** All three trainers
  auto-detect MPS / CUDA / CPU. On CPU they finish in 1–4× the times above; on
  T4 GPU they’re comfortably under the times in the table.

- **Don’t edit `phase5_backtester.py`.** Both FinFlow 1.0 and FinFlow 2.0 score
  through the same simulator — modifying it invalidates every model
  comparison.

- **The `finflow2/` extension is non-destructive.** It only reads from
  `data/preprocessed/` and `data/processed/`; it never overwrites them.

- **If you change preprocessing**, re-run `03 → 04 → 05 → 06 → 07 → 08 → 09 →
  10 → 11 → 12`. You don’t need to re-run `01` or `02` unless the raw CSVs
  themselves change.