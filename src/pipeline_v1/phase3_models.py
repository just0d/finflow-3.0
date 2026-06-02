"""
FinFlow 1.0 — Phase 3 Step 2: Model Architecture Definitions
Three models, one input format:  (batch, 24, n_features)

Model A — LSTM:
  Processes the 24-hour window SEQUENTIALLY.
  Each hour is read one at a time; the hidden state carries
  forward a compressed memory of what came before.
  Good at: short-term momentum, mean-reversion patterns.

Model B — Time-Series Transformer:
  Looks at ALL 24 hours SIMULTANEOUSLY via Self-Attention.
  Can discover that Hour 3 correlates with Hour 24, even if
  the hours between them are noisy — something LSTM forgets.
  Good at: long-range dependencies, regime change detection.
"""

import torch
import torch.nn as nn
import math

# MODEL A — LSTM CLASSIFIER

class LSTMClassifier(nn.Module):
    """
    2-layer bidirectional LSTM with Layer Normalization and dropout.

    Architecture:
        Input  (batch, 24, features)
        LSTM   (hidden=128, layers=2, bidirectional=False, dropout=0.3)
        LayerNorm(128)
        Dropout(0.3)
        Linear(128 → 64) → ReLU → Dropout(0.2)
        Linear(64 → 3)   ← output logits [SELL, HOLD, BUY]

    Design choices:
        - NOT bidirectional: in live trading you can't look at future hours.
          Bidirectional would cause look-ahead bias in the model itself.
        - LayerNorm after LSTM: stabilizes training on volatile financial data.
        - 2-layer LSTM: first layer catches short-term patterns (momentum
          reversals), second layer catches higher-order sequences.
    """

    def __init__(
        self,
        input_size:  int = 19,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        num_classes: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len=24, features)
        out, _ = self.lstm(x)          # (batch, 24, hidden)
        last   = out[:, -1, :]         # take ONLY the last timestep's output
        last   = self.norm(last)
        last   = self.drop(last)
        return self.head(last)         # (batch, 3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# MODEL B — TIME-SERIES TRANSFORMER CLASSIFIER

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need'.
    Injects a sense of WHEN each timestep is in the 24-hour window.
    Without this, the Transformer treats all timesteps as an unordered set.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                      # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    """
    Transformer Encoder for time-series classification.

    Architecture:
        Input  (batch, 24, features)
        Linear(features → d_model=64)      ← project to model dimension
        SinusoidalPositionalEncoding(d_model)
        TransformerEncoder(
            layers=2, heads=4, dim_ff=256, dropout=0.1
        )
        LayerNorm(d_model)
        Global Average Pool over the 24 timesteps
        Linear(64 → 32) → GELU → Dropout(0.1)
        Linear(32 → 3)                     ← output logits

    Design choices:
        - Global Average Pooling (not just CLS token): aggregates all
          24 hours equally; more robust for short sequences.
        - GELU activation: smoother gradient flow than ReLU.
        - d_model=64, nhead=4: each attention head captures a 16-dim
          sub-space; lightweight enough to train on ~20k sequences.
    """

    def __init__(
        self,
        input_size:   int = 19,
        d_model:      int = 64,
        nhead:        int = 4,
        num_layers:   int = 2,
        dim_feedforward: int = 256,
        dropout:      float = 0.1,
        num_classes:  int = 3,
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,    # Pre-LN variant: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 24, features)
        x = self.input_proj(x)          # (batch, 24, d_model)
        x = self.pos_enc(x)             # inject positional information
        x = self.transformer(x)         # self-attention across 24 hours
        x = self.norm(x)
        x = x.mean(dim=1)               # global average pool → (batch, d_model)
        return self.head(x)             # (batch, 3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    batch = torch.randn(32, 24, 19)   # 32 samples, 24h window, 19 features

    lstm_model = LSTMClassifier(input_size=19)
    tf_model   = TransformerClassifier(input_size=19)

    with torch.no_grad():
        out_lstm = lstm_model(batch)
        out_tf   = tf_model(batch)

    print(f"LSTM output shape    : {out_lstm.shape}  ✓")
    print(f"Transformer output   : {out_tf.shape}  ✓")
    print(f"LSTM parameters      : {lstm_model.count_parameters():,}")
    print(f"Transformer parameters: {tf_model.count_parameters():,}")
