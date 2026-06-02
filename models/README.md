# Models — checkpoints excluded from this repository

Trained weights are not included so the repo stays lightweight and binary
artefacts don't drift out of sync with code.

To request the trained weights, contact **Od Sukh-Ochir — odstar259@gmail.com**.

| Folder | Contents | Used by |
|--------|----------|---------|
| `v1/`  | `lstm_best.pt`, `transformer_best.pt`                                                                       | `src/pipeline_v1/phase3_train_evaluate.py`, `phase4_late_fusion.py`, `phase5_backtester.py` |
| `v2/`  | `cross_modal_best.pt`, `cross_modal_runstate.pt`, `cross_modal_history.csv`, `cross_modal_status.json` <br> `sentiment_lstm_best.pt`, `sentiment_lstm_runstate.pt`, `sentiment_lstm_history.csv`, `sentiment_lstm_status.json` | `src/finflow2/evaluation/predict.py`, `generate_signals.py`, `compare_all_models.py` |
