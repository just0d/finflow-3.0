"""
CrossModalTransformer: Combines price and FinBERT sentiment streams. 
Uses a cross-attention layer to learn a dynamic weight between the two signals, 
updating representations jointly through a single differentiable graph.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal Positional Encoding (local copy; same maths as FinFlow 1.0)

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))    # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])

# Cross-Modal Attention Block

class CrossModalAttentionBlock(nn.Module):
    """
    One block of Pre-LN cross-modal attention with a learned
    *fusion gate* γ_t controlling how much sentiment is mixed in.

    Forward signature:
        forward(h_price, e_text)  → h_price' (same shape)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff:    int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Pre-LN normalisations
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(d_model)

        # Multi-head cross-attention (Q from price, K/V from text)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # Fusion gate γ_t = σ(W_g · [h_price ‖ attn_out])  per timestep
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Position-wise FFN (post-fusion)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Save the latest gate values for explainability/inspection
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        h_price: torch.Tensor,   # (B, T, d_model)
        e_text:  torch.Tensor,   # (B, T, d_model)
    ) -> torch.Tensor:
        # Cross-attention: Q from price, K/V from text
        q = self.ln_q(h_price)
        kv = self.ln_k(e_text)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv,
                                       need_weights=False)        # (B,T,d)

        # Learned fusion gate γ_t per timestep
        gate_input = torch.cat([h_price, attn_out], dim=-1)        # (B,T,2d)
        gate = self.fusion_gate(gate_input)                        # (B,T,1)  ∈ [0,1]
        self.last_gate = gate.detach()
        h = h_price + gate * attn_out                              # gated residual

        # Standard position-wise FFN with residual
        h = h + self.ffn(self.ln_ffn(h))
        return h

# CrossModalTransformer — full model

class CrossModalTransformer(nn.Module):
    """
    End-to-end deep-fusion classifier.

    Inputs:
        price_x : FloatTensor (B, T, n_price_feat)   — math features per hour
        text_x  : FloatTensor (B, T, n_text_feat)    — FinBERT-derived per-hour
                                                       sentiment summary

    Output:
        logits  : FloatTensor (B, 3)                 — SELL / HOLD / BUY

    The model exposes `last_gates` (list of per-block gate tensors) for
    post-hoc analysis of how much sentiment was actually used at each
    timestep — useful for the research write-up.
    """

    def __init__(
        self,
        n_price_feat: int = 19,
        n_text_feat:  int = 6,
        d_model:      int = 64,
        n_heads:      int = 4,
        n_self_layers:  int = 1,        # self-attention layers in EACH stream
        n_cross_layers: int = 2,        # cross-modal blocks
        d_ff:         int = 256,
        dropout:      float = 0.1,
        num_classes:  int = 3,
        seq_len:      int = 24,
    ):
        super().__init__()
        self.d_model     = d_model
        self.seq_len     = seq_len
        self.n_text_feat = n_text_feat

        # Price stream
        self.price_proj = nn.Linear(n_price_feat, d_model)
        self.price_pe   = _PositionalEncoding(d_model, max_len=seq_len + 16,
                                               dropout=dropout)
        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer,
                                                    num_layers=n_self_layers)

        # Text stream (FinBERT-summary encoder)
        self.text_proj = nn.Linear(n_text_feat, d_model)
        self.text_pe   = _PositionalEncoding(d_model, max_len=seq_len + 16,
                                              dropout=dropout)
        text_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer,
                                                   num_layers=n_self_layers)

        # Stack of cross-modal blocks
        self.cross_blocks = nn.ModuleList([
            CrossModalAttentionBlock(d_model=d_model, n_heads=n_heads,
                                     d_ff=d_ff, dropout=dropout)
            for _ in range(n_cross_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    # Inspection helper
    @property
    def last_gates(self) -> list[torch.Tensor]:
        """Per-block fusion-gate tensors from the most recent forward()."""
        return [b.last_gate for b in self.cross_blocks if b.last_gate is not None]

    # Forward
    def forward(
        self,
        price_x: torch.Tensor,    # (B, T, n_price_feat)
        text_x:  torch.Tensor,    # (B, T, n_text_feat)
    ) -> torch.Tensor:
        # Price stream
        h_p = self.price_proj(price_x)
        h_p = self.price_pe(h_p)
        h_p = self.price_encoder(h_p)               # (B, T, d_model)

        # Text stream
        e_t = self.text_proj(text_x)
        e_t = self.text_pe(e_t)
        e_t = self.text_encoder(e_t)                # (B, T, d_model)

        # Cross-modal fusion (price queries text)
        h = h_p
        for block in self.cross_blocks:
            h = block(h, e_t)

        h = self.norm_out(h).mean(dim=1)            # global avg pool
        return self.head(h)                         # (B, 3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Sanity check

if __name__ == "__main__":
    B, T, P, S = 8, 24, 19, 6
    price = torch.randn(B, T, P)
    text  = torch.randn(B, T, S)

    model = CrossModalTransformer(n_price_feat=P, n_text_feat=S)
    with torch.no_grad():
        logits = model(price, text)

    print(f"Output shape : {logits.shape}     (expected ({B}, 3))")
    print(f"Parameters   : {model.count_parameters():,}")
    print(f"Fusion gate  : block-0 mean = {model.last_gates[0].mean().item():.3f}")
