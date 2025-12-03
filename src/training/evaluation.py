# src/training/evaluation.py

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor


Number = Union[float, int]


def loss_to_perplexity(loss: Union[Number, Tensor]) -> float:
    """
    Convierte una loss de language modeling (cross-entropy en nats)
    a perplexity = exp(loss).

    Acepta:
      - float / int
      - Tensor escalar
    """
    if isinstance(loss, Tensor):
        loss_val = float(loss.item())
    else:
        loss_val = float(loss)

    # Por seguridad, acotamos un poco (evitar overflow por si acaso)
    if math.isinf(loss_val) or math.isnan(loss_val):
        return float("inf")

    try:
        ppl = math.exp(loss_val)
    except OverflowError:
        ppl = float("inf")

    return float(ppl)