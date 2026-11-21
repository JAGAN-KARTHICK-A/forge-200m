import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import GQAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.gamma


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        content = self.w_up(x)
        gated = gate * content
        return self.w_down(gated)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        kv_heads: int,
        q_heads: int,
        dropout: float,
        max_seq_len: int,
        window_size: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.attention = GQAttention(
            d_model, kv_heads, q_heads, max_seq_len, window_size
        )
        self.norm1 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, D)
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
