# Data — excluded from this repository

All raw and derived data files are excluded for size and licensing reasons
(the news dataset cannot be redistributed in raw form).

To request access, contact **Od Sukh-Ochir — odstar259@gmail.com**.

| Folder            | Contents                                      | Pipeline stage |
|-------------------|-----------------------------------------------|----------------|
| `raw/`            | Hourly OHLCV (NVDA, TSLA, JPM, SPY) + 46,395 news articles | Source |
| `aligned/`        | Price ↔ news aligned at hourly granularity    | Phase 1 |
| `processed/`      | Per-hour FinBERT sentiment scores             | Phase 2 |
| `preprocessed/`   | v1 tensors (X, y, meta + scalers)             | Phase 3 |
| `cache/`          | v2 tensors (price / news / sentiment / text)  | precompute_tensors |
