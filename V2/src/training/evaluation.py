# src/training/evaluation.py

from __future__ import annotations

import math
from typing import Optional


def loss_to_perplexity(loss: Optional[float]) -> float:
    """
    Convierte loss (cross-entropy promedio) a perplexity: ppl = exp(loss).

    Blindaje:
    - loss None -> nan
    - loss NaN -> nan
    - loss inf -> inf
    - loss muy grande -> inf (evita overflow)
    """
    if loss is None:
        return float("nan")

    try:
        loss_f = float(loss)
    except (TypeError, ValueError):
        return float("nan")

    if math.isnan(loss_f):
        return float("nan")
    if math.isinf(loss_f):
        return float("inf")

    # Umbral anti-overflow / anti-métricas inútiles
    if loss_f > 20.0:
        return float("inf")

    # Higiene: CE no debería ser negativa
    if loss_f < 0.0:
        loss_f = 0.0

    return math.exp(loss_f)