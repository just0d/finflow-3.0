# FinFlow 2.0 — Unified Model Comparison

## Evaluation contract

All models scored on the **same** held-out 2025 test window with the **same** 24-hour windows, the **same** preprocessing, and the **same** event-driven backtester (`phase5_backtester.py`, unchanged).


## Full metric table

| Model                  | Family      | Accuracy   | Buy_Precision   | Sell_Precision   | Buy_F1   | Weighted_F1   | Total_Return_%   | Sharpe   | Max_Drawdown_%   | Profit_Factor   | N_Trades   | Win_Rate_%   |
|:-----------------------|:------------|:-----------|:----------------|:-----------------|:---------|:--------------|:-----------------|:---------|:-----------------|:----------------|:-----------|:-------------|
| LSTM_baseline          | FinFlow 1.0 | 0.4679     | 0.4106          | 0.4120           | 0.4425   | 0.4583        | -4.93            | -0.812   | -14.43           | 0.924           | 937        | +45.89       |
| Transformer_baseline   | FinFlow 1.0 | 0.4616     | 0.4095          | 0.3962           | 0.4287   | 0.4482        | -8.35            | -0.937   | -12.28           | 0.867           | 783        | +43.04       |
| Prophet                | FinFlow 1.0 | 0.3834     | 0.3718          | 0.3035           | 0.4043   | 0.3234        | —                | —        | —                | —               | —          | —            |
| LSTM_LateFusion        | FinFlow 1.0 | 0.4679     | 0.4106          | 0.4120           | 0.4425   | 0.4583        | -5.27            | -0.830   | -13.97           | 0.908           | 758        | +46.70       |
| Transformer_LateFusion | FinFlow 1.0 | 0.4616     | 0.4095          | 0.3962           | 0.4287   | 0.4482        | -2.32            | -0.607   | -10.84           | 0.955           | 523        | +44.36       |
| CrossModalTransformer  | FinFlow 2.0 | 0.3827     | 0.3549          | 0.3490           | 0.2899   | 0.3768        | +0.25            | -0.611   | -7.40            | 1.009           | 178        | +43.82       |
| SentimentLSTM          | FinFlow 2.0 | 0.4192     | 0.3975          | 0.3738           | 0.4249   | 0.4001        | -2.07            | -0.606   | -12.83           | 0.964           | 606        | +41.25       |
| BuyAndHold_SPY         | Benchmark   | —          | —               | —                | —        | —             | +15.55           | 0.215    | -19.47           | 99.900          | 1          | +100.00      |


## Ranked by composite score

Composite = 0.40·Sharpe + 0.30·Return + 0.20·WeightedF1 + 0.10·(-MaxDrawdown), all min-max normalised across models.


|   Rank | Model                  | Family      |   Composite_Score | Sharpe   | Total_Return_%   | Max_Drawdown_%   | Weighted_F1   |
|-------:|:-----------------------|:------------|------------------:|:---------|:-----------------|:-----------------|:--------------|
|      1 | Transformer_LateFusion | FinFlow 1.0 |            0.4038 | -0.607   | -2.32            | -10.84           | 0.4482        |
|      2 | SentimentLSTM          | FinFlow 2.0 |            0.3525 | -0.606   | -2.07            | -12.83           | 0.4001        |
|      3 | LSTM_baseline          | FinFlow 1.0 |            0.3444 | -0.812   | -4.93            | -14.43           | 0.4583        |
|      4 | LSTM_LateFusion        | FinFlow 1.0 |            0.3302 | -0.830   | -5.27            | -13.97           | 0.4583        |
|      5 | CrossModalTransformer  | FinFlow 2.0 |            0.3002 | -0.611   | +0.25            | -7.40            | 0.3768        |
|      6 | Transformer_baseline   | FinFlow 1.0 |            0.2255 | -0.937   | -8.35            | -12.28           | 0.4482        |
|      7 | Prophet                | FinFlow 1.0 |          nan      | —        | —                | —                | 0.3234        |
|      8 | BuyAndHold_SPY         | Benchmark   |          nan      | 0.215    | +15.55           | -19.47           | —             |


## Best model: **Transformer_LateFusion** (FinFlow 1.0)

- Composite score: **0.4038**
- Sharpe ratio: **-0.6069**
- Total return: **-2.32%**
- Max drawdown: **-10.84%**
- Weighted F1: **0.4482**
