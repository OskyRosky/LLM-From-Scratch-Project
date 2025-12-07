# src/model/layers.py (solo la clase FeedForward)

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Token embedding layer: maps token ids to vectors in R^{embed_dim}.

    Input:  (batch_size, seq_len)  of token ids
    Output: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input_ids:
            Tensor of shape (batch_size, seq_len) with token ids.

        Returns
        -------
        embeddings:
            Tensor of shape (batch_size, seq_len, embed_dim).
        """
        x = self.embedding(input_ids)
        # Common trick in GPT-style models: scale embeddings by sqrt(d_model)
        return x * (self.embed_dim ** 0.5)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    We assume 0-based positions [0, 1, ..., seq_len-1].

    Input:  (batch_size, seq_len)  of token ids (we only use shape)
    Output: (batch_size, seq_len, embed_dim)  positional vectors
    """

    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input_ids:
            Tensor of shape (batch_size, seq_len). We ignore the actual
            token values and only use seq_len to build positions.

        Returns
        -------
        pos_emb:
            Tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        device = input_ids.device
        # positions: (1, seq_len) -> broadcast to (batch_size, seq_len)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        pos_emb = self.pos_embedding(positions)
        return pos_emb


class FeedForward(nn.Module):
    """
    MLP posición-a-posición usado dentro del bloque Transformer.

    Estructura típica:
        x -> Linear(d_model -> hidden_dim)
           -> GELU
           -> Linear(hidden_dim -> d_model)
           -> Dropout
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * d_model  # valor típico en GPT

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    Thin wrapper around nn.LayerNorm to keep a consistent interface.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input and output shapes: (batch_size, seq_len, d_model)
        """
        return self.ln(x)