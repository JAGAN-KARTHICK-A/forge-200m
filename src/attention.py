import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
):
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(max_seq_len, dtype=torch.float32)

    freqs = torch.outer(t, freqs)

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_shape = x.shape
    x_per_tensor = x.float().view(*x_shape[:-1], -1, 2)

    x_complex = torch.view_as_complex(x_per_tensor)

    seq_len = x.shape[1]
    freqs_cis = freqs_cis[:seq_len]

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    x_rotated = x_complex * freqs_cis

    x_out = torch.view_as_real(x_rotated).flatten(3)

    return x_out.type_as(x)


class GQAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        kv_heads: int,
        q_heads: int,
        max_seq_len: int,
        window_size: int,
    ) -> None:
        super().__init__()
        assert d_model % q_heads == 0
        assert q_heads % kv_heads == 0
        assert window_size >= 0
        self.d_model = d_model
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.window_size = window_size
        self.d_q = self.d_model // self.q_heads
        self.q_per_kv_heads = self.q_heads // self.kv_heads
        self.kv_dim = self.d_q * self.kv_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.kv_dim)
        self.w_v = nn.Linear(d_model, self.kv_dim)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        freqs_cis = precompute_rope_frequencies(self.d_q, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x, mask=True):
        # x: (B, S, D)
        B, S, D = x.shape

        q = self.w_q(x).reshape(B, S, self.q_heads, self.d_q)
        k = self.w_k(x).reshape(B, S, self.kv_heads, self.d_q)
        v = self.w_v(x).reshape(B, S, self.kv_heads, self.d_q)

        q = apply_rotary_emb(q, self.freqs_cis)
        k = apply_rotary_emb(k, self.freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if mask:
            i = torch.arange(S, device=x.device).unsqueeze(1)
            j = torch.arange(S, device=x.device).unsqueeze(0)
            attn_mask = (j <= i) & (j >= i - self.window_size + 1)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )

        attn = attn.transpose(1, 2).reshape(B, S, D)
        attn = self.w_o(attn)
        return attn
