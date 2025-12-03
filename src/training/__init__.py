# src/training/optimizer.py

from typing import Any, Optional

import torch.nn as nn
from torch.optim import AdamW


def _get_param_groups(model: nn.Module, weight_decay: float):
    """
    Separa los parámetros en dos grupos:
      - con weight_decay (pesos de matrices)
      - sin weight_decay (bias y parámetros 1D como LayerNorm, embeddings escalares)

    Este patrón es típico en modelos tipo transformer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Parámetros 1D (LayerNorm, bias, etc.) -> sin weight decay
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    cfg: Optional[Any] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    betas: Optional[tuple] = None,
    eps: Optional[float] = None,
) -> AdamW:
    """
    Crea un AdamW con buenos valores por defecto.

    Prioridad de hiperparámetros:
        argumentos explícitos > atributos de cfg > valores por defecto.

    Se espera (si se pasa cfg) que pueda tener:
        - cfg.learning_rate
        - cfg.weight_decay
        - cfg.adam_betas
        - cfg.adam_eps
    pero si no existen, no pasa nada: usamos defaults.
    """

    # Valores por defecto razonables para GPT pequeño
    default_lr = 3e-4
    default_weight_decay = 0.01
    default_betas = (0.9, 0.95)
    default_eps = 1e-8

    if cfg is not None:
        lr = lr if lr is not None else getattr(cfg, "learning_rate", default_lr)
        weight_decay = (
            weight_decay
            if weight_decay is not None
            else getattr(cfg, "weight_decay", default_weight_decay)
        )
        betas = betas if betas is not None else getattr(cfg, "adam_betas", default_betas)
        eps = eps if eps is not None else getattr(cfg, "adam_eps", default_eps)
    else:
        lr = lr if lr is not None else default_lr
        weight_decay = weight_decay if weight_decay is not None else default_weight_decay
        betas = betas if betas is not None else default_betas
        eps = eps if eps is not None else default_eps

    param_groups = _get_param_groups(model, weight_decay=weight_decay)

    optimizer = AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )
    return optimizer