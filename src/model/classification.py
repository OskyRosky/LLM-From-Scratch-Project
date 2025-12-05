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
        # x: (B, d_model)
        return self.linear(x)  # (B, num_classes)


class GPTForClassification(nn.Module):
    """
    Modelo GPT + cabeza de clasificación sencilla.

    - Usa GPTModel como backbone.
    - Toma la representación del ÚLTIMO token de la secuencia.
    - Pasa ese vector por un ClassificationHead (lineal).
    """

    def __init__(self, config: GPTConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Backbone GPT
        self.gpt = GPTModel(config)

        # Cabeza de clasificación
        self.classifier = ClassificationHead(d_model=config.d_model,
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
        # Esto equivale a usar el "último token" como resumen de la secuencia.
        last_hidden = logits_lm[:, -1, :]  # (B, vocab_size) si el GPT devuelve logits directos

        # OJO: en el libro de Raschka normalmente se usa el último HIDDEN STATE,
        # pero aquí nuestro GPTModel devuelve logits de vocabulario.
        # Para mantenerlo simple en esta versión, tratamos esos logits como features.
        # En una versión 2, podríamos modificar GPTModel para devolver hidden states
        # y aquí usar esos en lugar de logits_lm.

        logits_cls = self.classifier(last_hidden)  # (B, num_classes)
        return logits_cls