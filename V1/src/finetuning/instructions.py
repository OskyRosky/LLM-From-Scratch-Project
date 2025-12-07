# src/finetuning/instructions.py

"""
Utilidades para fine-tuning del GPT en tareas de instrucciones
("instruction tuning") usando un objetivo de language modeling.

La idea es simple:
    - Tenemos secuencias de texto del estilo:
        "<instr> {prompt}\n<resp> {response}"
    - El modelo GPT recibe toda la secuencia como entrada.
    - El objetivo es predecir el siguiente token en cada posición
      (lenguaje autoregresivo estándar).

Este módulo NO define bucles de entrenamiento completos
(eso vive en src/cli/finetune_instructions.py),
sino una función central para calcular la pérdida.
"""

from typing import Tuple

import torch
from torch import nn

from src.model.gpt import GPTModel


def compute_instruction_loss(
    model: GPTModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcula la loss de language modeling para instruction tuning.

    Parameters
    ----------
    model:
        Instancia de GPTModel (con pesos preentrenados o ya finetuneados).
    input_ids:
        Tensor de forma (B, T) con IDs de tokens/caracteres.
        Se usa como entrada al modelo.
    labels:
        Tensor de forma (B, T) con los IDs a predecir.
        Normalmente será la misma secuencia que input_ids
        (InstructionDataset ya devuelve ambos iguales), y aquí
        haremos el "shift" de una posición.
    pad_token_id:
        ID que queremos ignorar en la loss (por ejemplo, padding).
        En nuestro setup usamos 0 como padding por defecto.

    Returns
    -------
    loss:
        Escalar (tensor 0D) con la CrossEntropyLoss promedio.
    logits:
        Tensor de forma (B, T, vocab_size) con los logits del modelo
        ANTES de aplicar el shift (útil para logging o depuración).
    """
    # Forward del modelo GPT: obtenemos logits LM (B, T, vocab_size)
    logits = model(input_ids)  # GPTModel.forward devuelve solo logits por defecto

    # Para LM autoregresivo:
    #   - en la posición t queremos predecir labels[t] usando input_ids[:t]
    #   - por conveniencia, desplazamos ambos tensores:
    #       logits[:, :-1, :]  vs  labels[:, 1:]
    #
    # De esta forma, ignoramos la última posición (no tiene "siguiente token").
    logits_shifted = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    labels_shifted = labels[:, 1:].contiguous()       # (B, T-1)

    B, Tm1, V = logits_shifted.shape

    # Reorganizamos para CrossEntropy:
    #   - logits: (B * (T-1), V)
    #   - labels: (B * (T-1),)
    logits_flat = logits_shifted.view(B * Tm1, V)
    labels_flat = labels_shifted.view(B * Tm1)

    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fct(logits_flat, labels_flat)

    return loss, logits