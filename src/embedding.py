import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        # x: B, S
        emb = self.emb(x)
        return emb
