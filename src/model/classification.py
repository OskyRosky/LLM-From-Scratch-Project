# src/model/classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import GPTModel, GPTConfig


class ClassificationHead(nn.Module):
    """
    Capa lineal simple para mapear una representación a logits de clases.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor de forma (batch_size, in_dim)
        """
        return self.fc(x)


class GPTForClassification(nn.Module):
    """
    Envuelve un GPT de caracteres preentrenado y añade un head de clasificación.

    Estrategia sencilla:
        - Ejecutamos el GPT como siempre: obtenemos logits del LM (B, T, V).
        - Tomamos los logits del ÚLTIMO token: (B, V).
        - Usamos esos V logits como representación de la secuencia.
        - El ClassificationHead los mapea a num_classes.

    Nota: no modificamos GPTModel; lo reutilizamos tal cual.
    """

    def __init__(self, config: GPTConfig, num_classes: int):
        super().__init__()
        self.gpt = GPTModel(config)
        self.num_classes = num_classes

        # Usamos el tamaño del vocabulario como dimensión de entrada del head
        self.classifier = ClassificationHead(config.vocab_size, num_classes)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """
        input_ids: (batch_size, seq_len) con ids de caracteres.
        targets:  (batch_size,) con etiquetas enteras 0..num_classes-1 (opcional).

        Devuelve:
            logits_cls: (batch_size, num_classes)
            loss: escalar o None si no se pasan targets
        """
        # Salida del modelo de lenguaje: (B, T, V)
        logits_lm = self.gpt(input_ids)

        # Representación de la secuencia: logits del último token -> (B, V)
        last_token_logits = logits_lm[:, -1, :]

        # Head de clasificación -> (B, num_classes)
        logits_cls = self.classifier(last_token_logits)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits_cls, targets)

        return logits_cls, loss