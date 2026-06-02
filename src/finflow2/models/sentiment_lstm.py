"""
SentimentLSTM: Custom LSTM cell built from scratch to incorporate a 5th 'Sentiment Gate'. 
Applies an exponential decay to the sentiment impulse to reflect the fading impact 
of news on price action over time. 
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sentiment-Decay LSTM Cell
class SentimentLSTMCell(nn.Module):
    """
    Custom LSTM cell with a learnable Sentiment Gate.

    Standard gates (input, forget, cell-candidate, output) are computed
    from the concatenated input and previous hidden state via a single
    fused linear layer (the conventional W_ih / W_hh decomposition,
    folded together for clarity and speed):

        [i_t, f_t, g_t, o_t] = split(W_x · x_t + W_h · h_{t-1} + b)

    The sentiment gate s_t is computed in parallel from the SAME
    [x_t ‖ h_{t-1}] context but with its own learned matrices.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Standard LSTM weights (fused i, f, g, o = 4 × hidden)
        self.W_x = nn.Parameter(torch.empty(input_size,  4 * hidden_size))
        self.W_h = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.b   = nn.Parameter(torch.zeros(4 * hidden_size))

        # Sentiment Gate weights (single hidden-dim sigmoid)
        self.W_xs = nn.Parameter(torch.empty(input_size,  hidden_size))
        self.W_hs = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_s  = nn.Parameter(torch.zeros(hidden_size))

        # Learned "sentiment direction" v_s in cell-state space
        # Initialised so that a neutral start gives near-zero contribution.
        self.v_s  = nn.Parameter(torch.randn(hidden_size) * 0.05)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier for stability; bias = 0 except forget-gate bias = +1
        # (the Jozefowicz-2015 trick: encourages remembering by default)
        std_x = 1.0 / math.sqrt(self.input_size)
        std_h = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W_x,  -std_x, std_x)
        nn.init.uniform_(self.W_h,  -std_h, std_h)
        nn.init.uniform_(self.W_xs, -std_x, std_x)
        nn.init.uniform_(self.W_hs, -std_h, std_h)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.b_s)
        # Forget-gate bias = +1
        H = self.hidden_size
        with torch.no_grad():
            self.b[H : 2 * H].fill_(1.0)

    def forward(
        self,
        x_t:        torch.Tensor,    # (B, input_size)
        h_prev:     torch.Tensor,    # (B, hidden_size)
        c_prev:     torch.Tensor,    # (B, hidden_size)
        s_decayed:  torch.Tensor,    # (B,) decayed sentiment scalar at this step
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (h_t, c_t, sentiment_gate_t).
        sentiment_gate_t is returned for inspection / regularisation.
        """
        # Standard four-gate computation
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b   # (B, 4H)
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)

        # Sentiment Gate
        s_t = torch.sigmoid(x_t @ self.W_xs + h_prev @ self.W_hs + self.b_s)  # (B, H)

        # Decayed sentiment vector ŝ_t = sentiment_t · v_s
        # s_decayed has shape (B,), broadcast it over hidden dim
        s_hat = s_decayed.unsqueeze(-1) * self.v_s                            # (B, H)

        # ★ Modified cell-state recursion
        c_t = f_t * c_prev + i_t * g_t + s_t * s_hat                          # (B, H)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t, s_t

# Sentiment-Decay LSTM Model (full classifier)

class SentimentLSTMModel(nn.Module):
    """
    Multi-layer Sentiment-Decay LSTM classifier.

    Forward signature:
        forward(price_x, sentiment, news_mask=None)
            price_x   : (B, T, n_price_feat)
            sentiment : (B, T)
            news_mask : (B, T) bool   — True when fresh news arrived
    """

    def __init__(
        self,
        n_price_feat: int = 19,
        hidden_size:  int = 128,
        num_layers:   int = 2,
        dropout:      float = 0.3,
        num_classes:  int = 3,
        # Decay configuration
        init_decay_hours: float = 6.0,    # initial half-life-ish constant
        learnable_decay:  bool  = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Build the stack: layer-0 takes raw price features, deeper
        # layers take the previous layer's hidden state.
        self.cells = nn.ModuleList()
        for ℓ in range(num_layers):
            in_size = n_price_feat if ℓ == 0 else hidden_size
            self.cells.append(SentimentLSTMCell(input_size=in_size,
                                                hidden_size=hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_size)

        # Learnable decay rate λ (parametrised via softplus to stay positive).
        # init: λ = log(2)/init_decay_hours so a half-life ≈ init_decay_hours
        init_lambda = math.log(2.0) / init_decay_hours
        # invert softplus: softplus(x) = log(1+e^x)  →  x = log(e^λ - 1)
        raw_init = math.log(math.expm1(init_lambda))
        self.lambda_raw = nn.Parameter(torch.tensor(raw_init),
                                        requires_grad=learnable_decay)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        # Buffers populated during forward() for inspection
        self.last_gates: list[torch.Tensor] | None = None
        self.last_decays: torch.Tensor | None = None

    # Helpers
    @property
    def decay_lambda(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)

    @property
    def half_life_hours(self) -> float:
        return float(math.log(2.0) / max(self.decay_lambda.item(), 1e-6))

    # Forward
    def forward(
        self,
        price_x:   torch.Tensor,   # (B, T, n_price_feat)
        sentiment: torch.Tensor,   # (B, T)
        news_mask: torch.Tensor | None = None,   # (B, T) bool/float
    ) -> torch.Tensor:
        B, T, _ = price_x.shape
        device  = price_x.device

        # Build the *decayed* sentiment series ŝ_t
        # If we have a news_mask, age_t resets to 0 whenever news_mask=1.
        # Otherwise, treat every nonzero sentiment as fresh news.
        if news_mask is None:
            news_mask = (sentiment.abs() > 1e-6).float()
        else:
            news_mask = news_mask.float()

        λ = self.decay_lambda                     # scalar tensor (>0)

        decayed = torch.zeros_like(sentiment)
        # Stateful scan to compute age & decayed sentiment
        last_news = torch.full((B,), -1.0, device=device)    # never seen news
        last_sent = torch.zeros(B, device=device)
        for t in range(T):
            fresh = news_mask[:, t]    # (B,)  ∈ {0,1}
            sent_t = sentiment[:, t]
            # When fresh news arrives, reset memory of the news event
            last_news = torch.where(fresh > 0.5, torch.tensor(float(t), device=device),
                                                  last_news)
            last_sent = torch.where(fresh > 0.5, sent_t, last_sent)
            age = torch.where(last_news >= 0,
                               torch.tensor(float(t), device=device) - last_news,
                               torch.tensor(0.0, device=device))
            # No news ever → decay term is 0
            ever_news = (last_news >= 0).float()
            decayed[:, t] = ever_news * last_sent * torch.exp(-λ * age)

        self.last_decays = decayed.detach()

        # Run the Sentiment-Decay LSTM stack
        # State per layer
        h = [torch.zeros(B, self.hidden_size, device=device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_size, device=device)
             for _ in range(self.num_layers)]

        gate_log: list[torch.Tensor] = [[] for _ in range(self.num_layers)]
        outputs = []

        for t in range(T):
            x_t = price_x[:, t, :]
            s_d = decayed[:, t]
            for ℓ, cell in enumerate(self.cells):
                h_new, c_new, s_gate = cell(x_t, h[ℓ], c[ℓ], s_d)
                # Apply dropout on inter-layer hidden state (training only)
                if ℓ < self.num_layers - 1:
                    h_new = self.dropout(h_new)
                h[ℓ], c[ℓ] = h_new, c_new
                x_t = h_new
                gate_log[ℓ].append(s_gate.detach())
            outputs.append(h[-1])

        self.last_gates = [torch.stack(g, dim=1) for g in gate_log]    # list of (B,T,H)

        last = self.norm(self.dropout(outputs[-1]))                    # (B, hidden)
        return self.head(last)                                          # (B, 3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Sanity check

if __name__ == "__main__":
    B, T, P = 4, 24, 19
    price = torch.randn(B, T, P)
    sent  = torch.zeros(B, T)
    sent[:, 5]  = -0.8     # bearish news at hour 5
    sent[:, 13] = +0.6     # bullish news at hour 13

    model = SentimentLSTMModel(n_price_feat=P, hidden_size=64, num_layers=2)
    with torch.no_grad():
        logits = model(price, sent)

    print(f"Output shape       : {logits.shape}")
    print(f"Parameters         : {model.count_parameters():,}")
    print(f"Initial decay λ    : {model.decay_lambda.item():.4f} per hour")
    print(f"Initial half-life  : {model.half_life_hours:.2f} hours")
    print(f"Decayed sentiment series at t=5..15 (sample 0):")
    print(model.last_decays[0, 5:16].tolist())
