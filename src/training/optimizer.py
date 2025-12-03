# src/training/optimizer.py

from typing import Any, Optional, Tuple

import torch.nn as nn
from torch.optim import AdamW


def _get_param_groups(model: nn.Module, weight_decay: float):
    """
    Separa parámetros en dos grupos:
    - con weight decay (pesos de capas lineales/conv)
    - sin weight decay (biases, LayerNorm, embeddings)
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Regla típica: no aplicar weight decay a biases ni a normalizaciones/embeddings
        if (
            name.endswith("bias")
            or "ln" in name.lower()
            or "norm" in name.lower()
            or "embedding" in name.lower()
        ):
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
    betas: Optional[Tuple[float, float]] = None,
    eps: Optional[float] = None,
) -> AdamW:
    """
    Crea un AdamW con buenos valores por defecto, pero permitiendo
    que vengan de un TrainingConfig si se pasa `cfg`.
    """

    # 1) Si hay cfg, usamos sus valores por defecto cuando los args son None
    if cfg is not None:
        if lr is None:
            lr = getattr(cfg, "learning_rate", getattr(cfg, "lr", 3e-4))
        if weight_decay is None:
            weight_decay = getattr(cfg, "weight_decay", 0.01)
        if betas is None:
            # Si tu TrainingConfig tiene algo como betas=(0.9, 0.95), lo usamos
            betas = getattr(cfg, "betas", (0.9, 0.95))
        if eps is None:
            eps = getattr(cfg, "adam_eps", 1e-8)

    # 2) Valores por defecto si aún quedan en None
    if lr is None:
        lr = 3e-4
    if weight_decay is None:
        weight_decay = 0.01
    if betas is None:
        betas = (0.9, 0.95)
    if eps is None:
        eps = 1e-8

    # 3) AQUÍ estaba el fallo: faltaba crear `param_groups`
    param_groups = _get_param_groups(model, weight_decay)

    # 4) Crear AdamW
    optimizer = AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )
    return optimizer