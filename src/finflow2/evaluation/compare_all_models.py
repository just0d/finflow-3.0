"""
FinFlow 2.0 — Unified Model Comparison Framework
Aggregates BOTH classification metrics AND trading metrics for every
model in the FinFlow ecosystem (1.0 baselines + 2.0 deep-fusion).

Models scored:

    Existing (FinFlow 1.0)
        LSTM                — phase3_train_evaluate.py result
        Transformer         — phase3_train_evaluate.py result
        Prophet             — phase3_train_evaluate.py result
        LSTM_NLP            — fused_signals_lstm.csv  (late fusion)
        Transformer_NLP     — fused_signals_transformer.csv (late fusion)

    NEW (FinFlow 2.0)
        CrossModalTransformer  — deep fusion via cross-attention
        SentimentLSTM          — deep fusion via custom sentiment-decay cell
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                classification_report, confusion_matrix)
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT  = ROOT / "outputs"
P3   = ROOT / "phase3_outputs"
CKPT = ROOT / "finflow2" / "checkpoints"

# 1. Classification metrics on the test set

def _classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy"      : float(accuracy_score(y_true, y_pred)),
        "buy_precision" : float(precision_score(y_true, y_pred, labels=[2],
                                                  average="macro", zero_division=0)),
        "sell_precision": float(precision_score(y_true, y_pred, labels=[0],
                                                  average="macro", zero_division=0)),
        "buy_f1"        : float(f1_score(y_true, y_pred, labels=[2],
                                          average="macro", zero_division=0)),
        "weighted_f1"   : float(f1_score(y_true, y_pred, average="weighted",
                                          zero_division=0)),
    }

def get_classification_for_new_models() -> dict[str, dict]:
    """Run inference on TEST split with the trained FinFlow 2.0 weights."""
    from finflow2.evaluation.predict import (
        predict_cross_modal, predict_sentiment_lstm, predict_baseline
    )
    from finflow2.training.data_loader import _build_aligned_tensors

    # True labels for test
    test_data = _build_aligned_tensors("test")
    y_true = test_data["y"]

    out = {}

    # Re-score the FinFlow 1.0 LSTM / Transformer with the same code path
    for name in ("lstm_baseline", "transformer_baseline"):
        try:
            df = predict_baseline(name.replace("_baseline", ""), "test")
            out[name] = _classification_metrics(y_true, df["raw_pred_num"].values)
        except Exception as exc:
            print(f"   [warn] {name}: {exc}")

    for name, fn in [("cross_modal", predict_cross_modal),
                       ("sentiment_lstm", predict_sentiment_lstm)]:
        try:
            df = fn("test")
            out[name] = _classification_metrics(y_true, df["raw_pred_num"].values)
        except Exception as exc:
            print(f"   [warn] {name}: {exc}")

    return out

# 2. Phase-3 baseline metrics from saved CSV

def load_phase3_classification() -> pd.DataFrame:
    p = P3 / "phase3_results.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

# 3. Trading metrics from backtest summaries

def load_backtests() -> pd.DataFrame:
    frames = []
    for path in [OUT / "backtest_summary.csv",
                  OUT / "finflow2_backtest_summary.csv"]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["strategy"],
                                                                keep="last")
    return df

# 4. Unified table builder
# How model names map across the three sources
MODEL_REGISTRY = [
    # display name           classif key         backtest strategy        family
    ("LSTM_baseline",         "LSTM",              "LSTM_Raw",              "FinFlow 1.0"),
    ("Transformer_baseline",  "Transformer",       "Transformer_Raw",       "FinFlow 1.0"),
    ("Prophet",               "Prophet",           None,                    "FinFlow 1.0"),
    ("LSTM_LateFusion",       "LSTM",              "LSTM_NLP",              "FinFlow 1.0"),
    ("Transformer_LateFusion","Transformer",       "Transformer_NLP",       "FinFlow 1.0"),
    ("CrossModalTransformer", "cross_modal",       "CrossModalTransformer", "FinFlow 2.0"),
    ("SentimentLSTM",         "sentiment_lstm",    "SentimentLSTM",         "FinFlow 2.0"),
    ("BuyAndHold_SPY",        None,                "Buy_Hold_SPY",          "Benchmark"),
]

def build_unified_table() -> pd.DataFrame:
    print("[1/3] Loading FinFlow 1.0 classification metrics…")
    p3 = load_phase3_classification()
    p3_by_name = {r["model"]: r for _, r in p3.iterrows()} if len(p3) else {}

    print("[2/3] Re-scoring FinFlow 2.0 models on TEST split…")
    new_class = get_classification_for_new_models()

    print("[3/3] Loading trading metrics…")
    bt_df = load_backtests()
    bt_by_strat = ({r["strategy"]: r for _, r in bt_df.iterrows()}
                    if len(bt_df) else {})

    rows = []
    for display, ckey, skey, family in MODEL_REGISTRY:
        # classification
        if ckey in ("LSTM", "Transformer", "Prophet") and ckey in p3_by_name:
            r = p3_by_name[ckey]
            cls_acc, cls_buyP, cls_sellP, cls_buyF1, cls_wF1 = (
                r.get("accuracy"), r.get("buy_precision"), r.get("sell_precision"),
                r.get("buy_f1"), r.get("weighted_f1"))
        elif ckey in new_class:
            r = new_class[ckey]
            cls_acc   = r["accuracy"]
            cls_buyP  = r["buy_precision"]
            cls_sellP = r["sell_precision"]
            cls_buyF1 = r["buy_f1"]
            cls_wF1   = r["weighted_f1"]
        else:
            cls_acc=cls_buyP=cls_sellP=cls_buyF1=cls_wF1=None

        # trading
        b = bt_by_strat.get(skey, {}) if skey else {}
        rows.append({
            "Model":         display,
            "Family":        family,
            "Accuracy":      cls_acc,
            "Buy_Precision": cls_buyP,
            "Sell_Precision":cls_sellP,
            "Buy_F1":        cls_buyF1,
            "Weighted_F1":   cls_wF1,
            "Total_Return_%":      b.get("total_return"),
            "Sharpe":              b.get("sharpe_ratio"),
            "Max_Drawdown_%":      b.get("max_drawdown"),
            "Profit_Factor":       b.get("profit_factor"),
            "N_Trades":            b.get("n_trades"),
            "Win_Rate_%":          b.get("win_rate"),
        })

    df = pd.DataFrame(rows)
    return df

# 5. Ranking

def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by composite score: 0.4·norm(Sharpe) + 0.3·norm(Return) +
    0.2·norm(WeightedF1) + 0.1·(-norm(MaxDrawdown))."""
    sub = df.copy()
    def _norm(col, *, lower_is_better=False):
        vals = pd.to_numeric(sub[col], errors="coerce")
        if vals.notna().sum() == 0: return pd.Series([np.nan]*len(sub))
        v_min, v_max = vals.min(), vals.max()
        if v_max == v_min: return pd.Series([0.5]*len(sub))
        n = (vals - v_min) / (v_max - v_min)
        return 1 - n if lower_is_better else n

    n_sharpe = _norm("Sharpe")
    n_ret    = _norm("Total_Return_%")
    n_f1     = _norm("Weighted_F1")
    n_dd     = _norm("Max_Drawdown_%", lower_is_better=True)   # lower DD better

    score = 0.40 * n_sharpe + 0.30 * n_ret + 0.20 * n_f1 + 0.10 * n_dd
    sub["Composite_Score"] = score.round(4)
    sub = sub.sort_values("Composite_Score", ascending=False, na_position="last")
    sub.insert(0, "Rank", range(1, len(sub) + 1))
    return sub

# 6. Pretty report

def write_markdown_report(table: pd.DataFrame, ranked: pd.DataFrame, out_md: Path):
    lines = []
    lines.append("# FinFlow 2.0 — Unified Model Comparison\n")
    lines.append("## Evaluation contract\n")
    lines.append(
        "All models scored on the **same** held-out 2025 test window with "
        "the **same** 24-hour windows, the **same** preprocessing, and the "
        "**same** event-driven backtester (`phase5_backtester.py`, "
        "unchanged).\n")

    lines.append("\n## Full metric table\n")
    pretty = table.copy()
    for col in ["Accuracy","Buy_Precision","Sell_Precision","Buy_F1","Weighted_F1"]:
        pretty[col] = pretty[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    for col in ["Total_Return_%","Max_Drawdown_%","Win_Rate_%"]:
        pretty[col] = pretty[col].apply(
            lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    for col in ["Sharpe","Profit_Factor"]:
        pretty[col] = pretty[col].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    pretty["N_Trades"] = pretty["N_Trades"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    lines.append(pretty.to_markdown(index=False))

    lines.append("\n\n## Ranked by composite score\n")
    lines.append(
        "Composite = 0.40·Sharpe + 0.30·Return + 0.20·WeightedF1 + "
        "0.10·(-MaxDrawdown), all min-max normalised across models.\n\n")
    pretty_r = ranked[["Rank","Model","Family","Composite_Score",
                         "Sharpe","Total_Return_%","Max_Drawdown_%",
                         "Weighted_F1"]].copy()
    pretty_r["Sharpe"]         = pretty_r["Sharpe"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    pretty_r["Total_Return_%"] = pretty_r["Total_Return_%"].apply(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    pretty_r["Max_Drawdown_%"] = pretty_r["Max_Drawdown_%"].apply(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    pretty_r["Weighted_F1"]    = pretty_r["Weighted_F1"].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    lines.append(pretty_r.to_markdown(index=False))

    # Best model highlight
    best = ranked.iloc[0]
    lines.append(f"\n\n## Best model: **{best['Model']}** ({best['Family']})\n")
    lines.append(
        f"- Composite score: **{best['Composite_Score']}**\n"
        f"- Sharpe ratio: **{best.get('Sharpe', '—')}**\n"
        f"- Total return: **{best.get('Total_Return_%', '—')}%**\n"
        f"- Max drawdown: **{best.get('Max_Drawdown_%', '—')}%**\n"
        f"- Weighted F1: **{best.get('Weighted_F1', '—')}**\n")

    out_md.write_text("\n".join(lines))

# Main

def main():
    print("  FinFlow 2.0 — Unified Comparison Framework")

    table = build_unified_table()
    ranked = rank_models(table)

    OUT.mkdir(exist_ok=True, parents=True)
    table_csv  = OUT / "finflow2_comparison_table.csv"
    ranked_csv = OUT / "finflow2_comparison_ranked.csv"
    report_md  = OUT / "finflow2_comparison_report.md"

    table.to_csv(table_csv, index=False)
    ranked.to_csv(ranked_csv, index=False)
    write_markdown_report(table, ranked, report_md)

    print("\nFull table:")
    print(table.to_string(index=False))
    print("\nRanked:")
    print(ranked[["Rank","Model","Family","Composite_Score","Sharpe",
                   "Total_Return_%","Weighted_F1"]].to_string(index=False))

    print(f"\n  Saved → {table_csv}")
    print(f"  Saved → {ranked_csv}")
    print(f"  Saved → {report_md}")
    print(f"\n  🏆 Best model: {ranked.iloc[0]['Model']}  "
          f"(score = {ranked.iloc[0]['Composite_Score']})")

if __name__ == "__main__":
    main()
