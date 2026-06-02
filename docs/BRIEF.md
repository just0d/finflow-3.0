# FinFlow v3 — Final Results Brief

Universe: JPM, SPY, TSLA (NVDA excluded). Strategy: long+short, confidence-weighted, NLP risk gate, with a four-member meta-ensemble on top.

## Headline ranking (Protocol B, 2025 test window)

| # | Strategy | Return | Sharpe | Max DD | Trades |
|---:|---|---:|---:|---:|---:|
| **1** | **G4 Majority Ensemble (M1+M2+M3+M4)** | **+30.76 %** | **1.71** | **−7.37 %** | 156 |
| 2 | Transformer + Late Fusion + NLP gate (M1) | +69.22 % | 1.66 | −22.21 % | 953 |
| 3 | G3 Majority Ensemble (M1+M2+M3) | +43.83 % | 1.51 | −14.65 % | 462 |
| 4 | G2 Majority Ensemble (M1+M2) | +35.39 % | 1.42 | −13.88 % | 374 |
| 5 | G3 Sharpe-weighted Ensemble | +67.75 % | 1.38 | −25.63 % | 1,473 |
| 6 | CrossModalTransformer × Transformer agreement (R6) | +20.58 % | 0.81 | — | 453 |
| — | Buy & Hold (ex-NVDA, benchmark) | +24.57 % | 0.80 | −31.13 % | 1 |

The G4 Majority Ensemble — a per-bar majority vote across the four post-fix members — is the project's best risk-adjusted result.

## Members (after the Protocol-B repairs)

- **M1** — Transformer + Late Fusion + NLP gate (strict thresholds).
- **M2** — LSTM + V4 conf-conditional gate (gate fires only when confidence > 0.55).
- **M3** — CrossModalTransformer × R6 Transformer-agreement.
- **M4** — SentimentLSTM × R6 Transformer-agreement.

## Three findings

**NLP-as-risk-manager.** The NLP gate adds +46.59 pp of return and +0.99 Sharpe to the Transformer baseline, while removing 370 of the worst trades. Mechanism: blocks shorting against good news and longing into bad news.

**Per-model gate calibration is required.** A reliability audit shows the LSTM and Transformer have similar ECE (≈ 0.10) but very different vetoed-trade P&L distributions: the gate's risk-protection benefit on LSTM is ≈ 1/3 of that on Transformer because the LSTM's contrarian-sentiment signals are near coin-flip rather than systematically wrong. The conf-conditional gate (V4) recovers +12.4 pp of return for the LSTM by gating only at high confidence.

**Deep fusion as ensemble member, not standalone.** CrossModal and SentimentLSTM both fail individually (CrossModal accuracy 0.366, SentimentLSTM 0.353) — neither calibration tweaks nor stricter floors help. The only no-retrain repair that works is Transformer-agreement: act on the deep model's signal only when it agrees with the Transformer. Both architectures move from −15 % to +13–20 % return with this single change.

## How the ensemble achieves Sharpe 1.71

Naive equal- and Sharpe-weighted averaging across the four members produces Sharpe ~0.94–1.08 — diluting the strong M1. The **majority veto** (act only when ≥3 of 4 members agree on direction) is the active ingredient. It produces a fundamentally different trade pattern: 156 trades vs the members' 453–953, because most bars fail the agreement threshold and stay flat. Removing M4 (G3 instead of G4) costs 0.20 Sharpe even though M4's standalone Sharpe is only 0.64 — the diversity matters more than the individual quality.