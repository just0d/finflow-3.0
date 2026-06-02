# FinFlow 3.0

**A sentiment-augmented algorithmic trading system with cross-modal deep fusion, an NLP risk gate, and a heterogeneous-member meta-ensemble.**

Author: Od Sukh-Ochir · Department of Data Science, Hanyang University

Contact: odstar259@gmail.com

---

## TL;DR

FinFlow studies whether financial-news sentiment (read by FinBERT) can act as
an automated **risk manager** for short-horizon trading, and whether a
**heterogeneous-member meta-ensemble** can extract complementary signal from
architectures that individually underperform. Built across three iterations
(1.0, 2.0, 3.0), the project culminates in a Sharpe-2.35 majority-vote
ensemble whose performance *improves* as the underlying assets get more
idiosyncratic — direct evidence of the diversity-of-decisions principle at
the trading-decision layer.

## Headline results (one-year held-out 2025 test set, no look-ahead)

| Strategy                                      | Return     | Sharpe   | Max DD     | Trades |
|-----------------------------------------------|-----------:|---------:|-----------:|-------:|
| G4 Majority Ensemble (3-ticker)               | +30.76 %   | **1.71** | **−7.37 %**| 156    |
| G4 Majority Ensemble (4-ticker robustness)    | +52.83 %   | **2.35** | −7.36 %    | 224    |
| Transformer + Late Fusion + NLP gate          | +69.22 %   | 1.66     | −22.21 %   | 953    |
| Buy & Hold (ex-NVDA benchmark)                | +24.57 %   | 0.80     | −31.13 %   | 1      |

NLP-as-risk-manager ablation on the Transformer baseline: **+46.6 pp return, +0.99 Sharpe**.

## Repository layout

```
finflow/
  ├── configs/          # YAML hyperparams + paths
  ├── data/             # excluded — request via email
  ├── models/           # excluded — request via email
  ├── notebooks/        # exploratory + Colab-ready
  ├── src/
  │   ├── pipeline_v1/    # FinFlow 1.0 baselines + late fusion
  │   ├── finflow2/       # FinFlow 2.0 deep-fusion architectures
  │   ├── backtesting/    # Protocol-B backtester + metrics
  │   ├── analysis/       # FinFlow 3.0 ensemble + diagnostics
  │   └── utils/
  ├── results/          # small CSVs + figures cited in the paper
  ├── webapp/           # interactive demo
  ├── docs/             # paper, poster, brief
  ├── scripts/          # operational helpers
  └── tests/
```

## Quickstart

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the data + model weights (email odstar259@gmail.com)
#    Drop them into data/ and models/ following the structure in
#    data/README.md and models/README.md.

# 3. End-to-end pipeline (run in order)
python src/pipeline_v1/phase1_alignment.py
python src/pipeline_v1/phase2_finbert.py
python src/pipeline_v1/phase3_preprocessing.py
python src/pipeline_v1/phase3_train_evaluate.py
python src/pipeline_v1/phase4_late_fusion.py
python src/pipeline_v1/phase5_backtester.py

# 4. FinFlow 2.0 deep-fusion training
python -m src.finflow2.training.precompute_tensors
python -m src.finflow2.training.train_cross_modal
python -m src.finflow2.training.train_sentiment_lstm

# 5. FinFlow 3.0 analyses
python src/backtesting/phase6_backtester_v2.py
python src/analysis/phase8_calibration_analysis.py
python src/analysis/phase9_lstm_gate_tuning.py
python src/analysis/phase10_deep_fusion_diagnose.py
python src/analysis/phase11_meta_ensemble.py
python src/analysis/robustness_with_nvda.py
```

## Data and weights

Data and trained weights are excluded from the repository for size and
licensing reasons. Request access from **odstar259@gmail.com** and follow
`data/README.md` and `models/README.md` for the folder structure inference
scripts expect.

## Paper, poster, demo

- `docs/PAPER.md` (and `PAPER.docx`) — full research paper
- `docs/POSTER_CONTENT.docx` — poster text and layout
- `webapp/index.html` — interactive results browser

## Citation

See `CITATION.cff` (to be added).

## License

MIT (see `LICENSE`, to be added). Research use only.
