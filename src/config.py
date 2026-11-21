from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 1024
    vocab_size: int = 8192
    q_heads: int = 16
    kv_heads: int = 4
    n_layers: int = 17
    dropout: float = 0.01
    max_seq_len: int = 2048
    window_size: int = 2048
    d_ff: int = 2816
