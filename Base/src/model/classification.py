# src/model/classification.py

import torch
from torch import nn

from src.model.gpt import GPTModel, GPTConfig


class ClassificationHead(nn.Module):
    """
    Capa final de clasificación: toma un vector de tamaño d_model
    y lo proyecta a num_classes.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_model)
        return: (B, num_classes)
        """
        return self.linear(x)


class GPTForClassification(nn.Module):
    """
    Modelo GPT + cabeza de clasificación.

    - Usa GPTModel como backbone.
    - Obtiene las *hidden states* internas del GPT.
    - Aplica un pooling sobre la dimensión temporal (mean / first / last).
    - Pasa el vector resultante por una cabeza lineal de clasificación.
    """

    def __init__(
        self,
        config: GPTConfig,
        num_classes: int,
        pooling: str = "mean",  # "mean", "first", "last"
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.pooling = pooling

        # Backbone GPT
        self.gpt = GPTModel(config)

        # Cabeza de clasificación
        self.classifier = ClassificationHead(
            d_model=config.d_model,
            num_classes=num_classes,
        )

    def _pool_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, d_model)
        Devuelve un vector (B, d_model) según la estrategia de pooling.
        """
        if self.pooling == "first":
            # Usar el primer token como "CLS"
            return hidden[:, 0, :]  # (B, d_model)
        elif self.pooling == "last":
            # Usar el último token de la secuencia
            return hidden[:, -1, :]  # (B, d_model)
        else:
            # Por defecto: promedio sobre la dimensión temporal T
            return hidden.mean(dim=1)  # (B, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T) con IDs de tokens/caracteres.

        Devuelve:
            logits_cls: (B, num_classes)
        """
        # Pedimos al GPT tanto logits de LM como hidden states
        logits_lm, hidden = self.gpt(input_ids, return_hidden=True)
        # hidden: (B, T, d_model)

        # Pooling sobre hidden states
        features = self._pool_hidden(hidden)  # (B, d_model)

        # Cabeza de clasificación
        logits_cls = self.classifier(features)  # (B, num_classes)
        return logits_cls