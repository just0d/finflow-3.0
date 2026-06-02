# FinFlow 2.0 — Research Architectures

This document explains the **two new model architectures** added in
FinFlow 2.0 and the research questions each one is designed to test.

The original FinFlow 1.0 system (Prophet, LSTM, Transformer baselines,
FinBERT sentiment, late-fusion meta-rules, event-driven backtester)
is preserved unchanged. Both new models are scored against the
existing baselines using the **same** dataset, splits, preprocessing,
and backtester (`phase5_backtester.py`).

---

## 1. CrossModalTransformer — Deep Fusion via Cross-Attention

### Hypothesis
> *Hypothesis H1.* The hand-tuned late-fusion rule used in FinFlow 1.0
> ("if FinBERT sentiment ≤ −0.5 and the model says BUY → veto to HOLD")
> is brittle: it applies the same threshold to every ticker, every
> regime, and every level of news intensity. A learned cross-attention
> layer that takes price as the **Query** and FinBERT-derived text
> features as the **Key/Value** can re-derive a context-dependent
> version of the same logic — and discover better gating policies the
> rule cannot express.

### Mathematical core

For each timestep *t* of the 24-hour window we compute

| Symbol | Meaning |
|---|---|
| **h_price**[t] ∈ ℝ^{d_model} | encoded math features at hour t |
| **e_text**[t] ∈ ℝ^{d_model}  | encoded FinBERT-summary embedding at hour t |

The cross-modal layer is a multi-head attention block where Query
comes from the **price** stream and Key/Value come from the **text**
stream:

```
     Q = W_Q · h_price[t]
     K = W_K · e_text                  (over all 24 hours)
     V = W_V · e_text
     attn_t = softmax( Q · Kᵀ / √d_k ) · V
```

…followed by a **learned per-step fusion gate** γ_t:

```
     γ_t = σ( W_g · [ h_price[t] ‖ attn_t ] )
     h'  = h_price[t] + γ_t · attn_t          ← gated residual fusion
     h'' = h' + FFN(LN(h'))                   ← position-wise FFN
```

γ_t ∈ [0, 1] is the FinFlow 2.0 analogue of the FinFlow 1.0
emergency-brake rule. When γ_t → 0 the network is saying
"ignore news at this hour"; when γ_t → 1 it is saying
"trust news fully here". Crucially γ_t is **per-timestep,
per-sample, per-ticker** and is learned end-to-end from the
classification loss.

### Architecture diagram

```
                ┌──────────────────────────┐
   Price        │  Linear(19→64) + PE      │
   features ───►│  Self-Attention encoder  │── h_price (B,24,64)
   (B,24,19)    │  (1 layer, 4 heads)      │
                └──────────────────────────┘
                                                ┌──── CrossModalBlock × 2 ────┐
                                                │  Q=h_price, K=V=e_text      │
                                                │  attn → gated residual γ_t  │── h
                                                │  + FFN                       │
                                                └──────────────────────────────┘
                ┌──────────────────────────┐                ▲
   Text         │  Linear(6→64) + PE       │                │
   features ───►│  Self-Attention encoder  │── e_text ──────┘
   (B,24,6)     │  (1 layer, 4 heads)      │
                └──────────────────────────┘

                          h ──► LayerNorm ──► mean over 24 ──► Linear(64→32)
                                                                ──► GELU
                                                                ──► Linear(32→3)
                                                                  = logits
                                                                  (SELL / HOLD / BUY)
```

### What is different from a standard model

A standard Transformer **mixes price and text by concatenation** —
either at the input (early fusion) or at the logit level (late
fusion). Both throw away the structural difference between the two
modalities: prices are dense and high-frequency, news is sparse and
bursty.

CrossModalTransformer keeps the two streams architecturally separate
all the way to the cross-attention block. That block is the *only*
place where the modalities meet, and even there the interaction is
asymmetric (price queries text). The fusion gate γ_t makes the
interaction *adaptive*: under normal conditions γ_t can stay near
zero and the model behaves like the pure price Transformer; in
high-news regimes γ_t opens up and sentiment flows in.

### What each part contributes

| Component | Role |
|---|---|
| Separate price encoder | Captures intra-day momentum / volatility regime |
| Separate text encoder  | Aggregates the 24-hour news context (a piece of news at hour 5 may matter at hour 23) |
| Cross-attention        | "Where in the past 24 h did relevant news happen?" |
| Fusion gate γ_t        | "How much should I trust news at this exact hour?" |
| Mean pooling           | Robust aggregation for short sequences |

### Why this is research-novel

Architecturally this is a *data-asymmetric* cross-modal Transformer:
the modality used as Query is the modality we want to *predict*
about, while the modality used as Key/Value provides *contextual
explanation*. This pattern has been used in vision-language alignment
(e.g. BLIP-2, Q-Former) but applying it inside a quantitative trading
loop where the gate is interpretable as "how much sentiment to use"
is, to the best of my knowledge, novel for this dataset.

### Connection to the FinBERT embeddings

The original cross-modal contract calls for K/V drawn from FinBERT
hidden states. We use the FinBERT *output simplex* plus auxiliary
news statistics:

```
e_text[t] = [ sentiment_score,
              p_pos, p_neg, p_neu,
              log1p(news_count),
              sentiment_score · log1p(news_count) ]
```

This is a **6-dim** compact summary embedding per timestep. The
projection `Linear(6→d_model)` lifts it to the model dimension so the
cross-attention contract (Q from price, K/V from text) is preserved.
Swapping in true FinBERT [CLS] hidden states later is a one-line
change to `n_text_feat`.

---

## 2. SentimentLSTM — Custom Recurrent Cell with Sentiment Decay

### Hypothesis
> *Hypothesis H2.* News sentiment is a **short-half-life** signal: a
> bearish headline at hour *t* should depress the BUY signal sharply
> at *t*, less so at *t+1*, and barely at all by *t+12*. A vanilla
> LSTM has no inductive bias for this kind of decay; it must waste
> capacity learning to "forget" sentiment. By baking exponential
> decay directly into the cell-state recursion we encode this prior,
> freeing capacity for price dynamics.

### Mathematical core

We implement an LSTM cell **from scratch** — `nn.LSTM` /
`nn.LSTMCell` are not used anywhere in `SentimentLSTMModel`.

Standard LSTM:

```
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                         (vanilla)
   h_t = o_t ⊙ tanh(c_t)
```

FinFlow 2.0 cell (★ = the modification):

```
   i_t, f_t, g_t, o_t = standard sigmoidal/tanh gates
   s_t  = σ( W_s · [x_t ‖ h_{t-1}] + b_s )                ← Sentiment Gate (NEW)
   ŝ_t  = sentiment_t · exp( -λ · age_t ) · v_s            ← decayed sentiment (NEW)
   c_t  = f_t ⊙ c_{t-1} + i_t ⊙ g_t  +  s_t ⊙ ŝ_t        ★
   h_t  = o_t ⊙ tanh(c_t)
```

where:

| Symbol | Meaning |
|---|---|
| `s_t`         | per-step sentiment gate, learned from x_t and h_{t-1} |
| `sentiment_t` | scalar FinBERT score at hour t (zero when no news) |
| `age_t`       | hours since the most recent news event for this stock |
| `λ`           | learnable decay rate, λ = softplus(λ_raw) > 0 |
| `v_s`         | learnable d_hidden direction in cell-state space |

The gate `s_t` decides **how much** to write the decayed sentiment
into memory; the term `exp(-λ·age_t)` decides **how stale** that
sentiment is. The two are multiplied so:

* At a fresh news hour with strong sentiment → both factors are large
  → big update to cell state.
* Hours after the news → `exp(-λ·age_t)` shrinks even if `s_t` stays
  high → the sentiment fades from memory.
* Quiet stock with no news ever → contribution is exactly 0.

### Decay reset rule

```
   age_t = t − argmax_{τ ≤ t} { news_mask_τ = 1 }
```

Whenever `news_mask_t = 1` (i.e. `news_count_t > 0`), `age_t` resets
to 0 and `last_sentiment` is overwritten with the freshest score. If
no news ever arrived for this window, the decay term is multiplied by
the indicator `ever_news` so it stays exactly 0 — a desirable
"no-prior-information" boundary condition.

### Architecture diagram

```
   t=0      t=1      ...      t=23
   x_0       x_1               x_23           ← price features (B, T, 19)
    │         │                  │
    ▼         ▼                  ▼
  ┌─────┐  ┌─────┐            ┌─────┐
  │cell │─►│cell │─► ... ─────►│cell │─► h, c (Layer 1, hidden=96)
  └─────┘  └─────┘            └─────┘
    ▲         ▲                  ▲
    │         │                  │
   ŝ_0       ŝ_1               ŝ_23           ← decayed-sentiment input
                                                 (computed per-step, depends on
                                                  sentiment series + news_mask)

   Stack of 2 layers, dropout on inter-layer hidden state.
   Final h ──► LayerNorm ──► Linear(96→64) ──► ReLU ──► Linear(64→3) = logits
```

### What is different from a standard LSTM

| Aspect | Vanilla `nn.LSTM` | SentimentLSTM (FinFlow 2.0) |
|---|---|---|
| Cell-state recursion | `c_t = f·c_{t−1} + i·g` | `c_t = f·c_{t−1} + i·g + s·ŝ` |
| Number of gates | 4 | **5** (added Sentiment Gate) |
| Time-decay of news | ❌ must be learned implicitly | ✅ baked-in `exp(-λ·age_t)` |
| Decay rate | n/a | **learnable** scalar λ |
| Multimodal input | single tensor | (price, sentiment, news_mask) |
| Implementation | `torch.nn.LSTM` (CUDA-fused) | hand-rolled cell, pure Python loop |

### What each part contributes

| Component | Role |
|---|---|
| Standard 4 gates       | Same job as vanilla LSTM — track price dynamics |
| Sentiment Gate `s_t`   | "Should I bother writing this news to memory?" |
| `exp(-λ·age_t)` factor | "How stale is the news?" |
| Learnable λ            | The model decides if sentiment lives 2 h or 20 h |
| `v_s` direction        | Maps a scalar sentiment to a hidden-state direction (so different parts of the cell state can encode bullish vs bearish) |

### What this *tests*

The most interesting research observable is the **converged value of
λ** (and thus the *learned half-life* `ln 2 / λ`):

* If λ converges large (half-life ≈ 1–2 h) → news is a very
  short-lived signal, validating the inductive bias.
* If λ converges small (half-life ≈ 24 h) → news has lasting impact;
  the model would have done better with a vanilla LSTM.
* If λ converges to its initialisation → the dataset doesn't contain
  enough signal to identify the decay constant.

The training script logs `half_life_h` every epoch precisely for this
analysis.

### Why this is research-novel

Most multimodal recurrent fusion in finance attaches sentiment **as
an extra input feature**. This forces a vanilla LSTM to (a) learn
that the sentiment dimension is special, and (b) learn its own
forgetting rate for it. SentimentLSTM hard-codes both, leaving the
network only the parameters that actually need to be learned. It is
the recurrent analogue of an *informed prior*.

---

## 3. Common evaluation contract

Both new models are evaluated under exactly the same harness as the
FinFlow 1.0 baselines:

* Same 24-hour lookback windows
* Same per-ticker StandardScaler fitted on train, applied to test
* Same chronological splits (train: Jan 2023 – Aug 2024, val: Sep 2024
  – Jan 2025, test: Jan 2025 – Jan 2026)
* Same SELL/HOLD/BUY classification target (`target_direction`)
* Same `phase5_backtester.py` engine (untouched)
* Same Sharpe / Max Drawdown / Profit Factor / Win Rate metrics

This is essential for the comparison to be fair — see
`finflow2/evaluation/compare_all_models.py` for the unified
report.

## 4. Code map

```
finflow2/
 ├── models/
 │    ├── lstm_baseline.py            # re-exports phase3_models.LSTMClassifier
 │    ├── transformer_baseline.py     # re-exports phase3_models.TransformerClassifier
 │    ├── cross_modal_transformer.py  # NEW
 │    └── sentiment_lstm.py           # NEW
 ├── training/
 │    ├── data_loader.py              # builds aligned (price, text, sentiment) tensors
 │    ├── train_cross_modal.py
 │    └── train_sentiment_lstm.py
 ├── evaluation/
 │    ├── predict.py                  # batch inference helpers
 │    ├── generate_signals.py         # writes fused_signals_{crossmodal,sentlstm}.csv
 │    └── compare_all_models.py       # builds the unified comparison table
 ├── backtesting/
 │    └── run_backtests.py            # ADAPTER — calls existing phase5_backtester
 ├── docs/
 │    └── RESEARCH.md                 # this document
 └── checkpoints/
      ├── cross_modal_best.pt
      └── sentiment_lstm_best.pt
```
