import torch
import torch.nn as nn

from config import Config
from embedding import Embedding
from transformer import Decoder, RMSNorm


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    config.d_model,
                    config.kv_heads,
                    config.q_heads,
                    config.dropout,
                    config.max_seq_len,
                    config.window_size,
                    config.d_ff,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.emb = Embedding(config.vocab_size, config.d_model)
        self.vocab = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.emb.emb.weight = self.vocab.weight

    def forward(self, x):
        x = self.emb(x)

        for layer in self.decoders:
            x = layer(x)

        x = self.norm(x)
        return self.vocab(x)
