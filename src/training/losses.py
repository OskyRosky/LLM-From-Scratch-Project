# src/training/losses.py

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def language_modeling_loss(
    logits: Tensor,          # (batch_size, seq_len, vocab_size)
    target_ids: Tensor,      # (batch_size, seq_len)
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """
    Cross-entropy para modelo de lenguaje autoregresivo.

    Asume que `target_ids[t]` es el token "siguiente" a predecir
    para la posición correspondiente de `logits[t]`.

    ignore_index:
        Útil si en el futuro queremos enmascarar posiciones (por ejemplo,
        padding) y que NO contribuyan a la loss.
    """
    # Aplanamos batch y secuencia para usar F.cross_entropy
    B, T, V = logits.shape

    logits_flat = logits.view(B * T, V)      # (B*T, V)
    targets_flat = target_ids.view(B * T)    # (B*T,)

    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    return loss


def lm_token_accuracy(
    logits: Tensor,
    target_ids: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """
    Accuracy simple por token (solo para monitoreo en validación).
    """
    B, T, V = logits.shape
    preds = logits.argmax(dim=-1)  # (B, T)

    mask = target_ids != ignore_index
    correct = (preds == target_ids) & mask

    # Evitar división entre cero si todo está ignorado
    total = mask.sum().clamp(min=1)
    acc = correct.sum().float() / total.float()
    return acc