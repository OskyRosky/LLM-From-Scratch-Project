import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Create a lower-triangular (causal) mask of shape (1, 1, seq_len, seq_len),
    where positions [i, j] with j > i are masked (set to 0).

    This format is convenient for broadcasting with
    (batch_size, num_heads, seq_len, seq_len) attention scores.
    """
    # mask[i, j] = 1 if j <= i else 0
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    # reshape to (1, 1, seq_len, seq_len) for broadcasting over batch & heads
    return mask.unsqueeze(0).unsqueeze(0)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute scaled dot-product attention.

    Parameters
    ----------
    query, key, value:
        Tensors of shape (batch_size, num_heads, seq_len, head_dim).
    mask:
        Optional tensor that is broadcastable to (batch_size, num_heads, seq_len, seq_len).
        If provided, positions where mask == 0 (or False) will be masked out
        (assigned -inf before softmax).

    Returns
    -------
    output:
        Tensor of shape (batch_size, num_heads, seq_len, head_dim).
    attention_weights:
        Tensor of shape (batch_size, num_heads, seq_len, seq_len).
    """
    # query @ key^T -> (batch, heads, seq_len, seq_len)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # We assume mask is boolean or 0/1; positions with 0/False are masked.
        # Mask out by setting those positions to a very negative value.
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = F.softmax(scores, dim=-1)

    # Multiply by values -> (batch, heads, seq_len, head_dim)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.

    Input shape:
        x: (batch_size, seq_len, embed_dim)

    Output:
        output: (batch_size, seq_len, embed_dim)
        attention_weights: (batch_size, num_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Reshape a projected tensor from (batch_size, seq_len, embed_dim)
        to (batch_size, num_heads, seq_len, head_dim).
        """
        return (
            x.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)  # (batch, heads, seq, head_dim)
            .contiguous()
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply multi-head self-attention.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, seq_len, embed_dim).
        mask:
            Optional mask broadcastable to (batch_size, num_heads, seq_len, seq_len).

        Returns
        -------
        output:
            Tensor of shape (batch_size, seq_len, embed_dim).
        attention_weights:
            Tensor of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, _ = x.size()

        # Project inputs to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (batch, heads, seq_len, head_dim)
        q = self._shape(q, batch_size, seq_len)
        k = self._shape(k, batch_size, seq_len)
        v = self._shape(v, batch_size, seq_len)

        # Compute attention
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # Merge heads: (batch, seq_len, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)  # (batch, seq_len, heads, head_dim)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # Final linear projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        return output, attn_weights