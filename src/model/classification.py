import torch
from torch import nn

from src.model.gpt import GPTModel, GPTConfig


class ClassificationHead(nn.Module):
    """
    Capa final de clasificación: toma un vector de tamaño `in_dim`
    (en esta versión: vocab_size) y lo proyecta a `num_classes`.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        return self.linear(x)  # (B, num_classes)


class GPTForClassification(nn.Module):
    """
    Modelo GPT + cabeza de clasificación sencilla.

    En esta versión:

    - Usa GPTModel como backbone.
    - Toma los LOGITS de LM del ÚLTIMO token de la secuencia.
    - Los trata como un vector de features de tamaño vocab_size.
    - Los pasa por una capa lineal hacia num_classes.
    """

    def __init__(self, config: GPTConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Backbone GPT
        self.gpt = GPTModel(config)

        # En tu implementación actual, GPTModel devuelve logits de vocabulario
        # de tamaño vocab_size. Así que la última dimensión de logits_lm es
        # precisamente config.vocab_size.
        in_dim = config.vocab_size

        # Cabeza de clasificación
        self.classifier = ClassificationHead(in_dim=in_dim,
                                             num_classes=num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T) con IDs de caracteres.

        Devuelve:
            logits_cls: (B, num_classes)
        """
        # Llamamos al GPT. Dependiendo de cómo esté implementado GPTModel,
        # puede devolver:
        #   - solo logits: (B, T, vocab_size)
        #   - o una tupla: (logits, ...).
        out = self.gpt(input_ids=input_ids)

        if isinstance(out, tuple):
            # Nos quedamos con los logits de LM
            logits_lm = out[0]
        else:
            logits_lm = out

        # logits_lm: (B, T, vocab_size)
        # Tomamos la última posición temporal T-1 para cada ejemplo.
        last_token_logits = logits_lm[:, -1, :]  # (B, vocab_size)

        # Estos logits los usamos como features para la clasificación.
        logits_cls = self.classifier(last_token_logits)  # (B, num_classes)
        return logits_cls