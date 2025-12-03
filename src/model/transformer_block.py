# src/model/transformer_block.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    """
    Bloque Transformer de tipo decoder-only (como GPT).

    Estructura (pre-norm):

        x -> x + MHA(LN(x))
        x -> x + FFN(LN(x))

    Donde:
        - MHA es multi-head self-attention con máscara causal (se pasa en el forward)
        - FFN es un MLP posición-a-posición
        - LN es LayerNorm sobre la última dimensión (embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * embed_dim  # típico en GPT

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim

        # Submódulos
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(
            d_model=embed_dim,
            hidden_dim=ff_hidden_dim,
            dropout=dropout,
            )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        x:
            Tensor de entrada de forma (batch_size, seq_len, embed_dim).
        mask:
            Máscara opcional broadcastable a
            (batch_size, num_heads, seq_len, seq_len).
        return_attn:
            Si True, también devuelve los pesos de atención.

        Returns
        -------
        y:
            Tensor de salida de forma (batch_size, seq_len, embed_dim).
        attn_weights (opcional):
            Tensor de forma (batch_size, num_heads, seq_len, seq_len)
            o None si return_attn=False.
        """
        # --- Bloque de atención (pre-norm) ---
        residual = x
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.attn(x_norm, mask=mask)
        x = residual + attn_out  # residual 1

        # --- Bloque FFN (pre-norm) ---
        residual = x
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = residual + ff_out  # residual 2

        if return_attn:
            return x, attn_weights
        return x, None