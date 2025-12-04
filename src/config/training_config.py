# src/config/training_config.py

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """
    Configuración para el entrenamiento del modelo.

    - batch_size: tamaño de batch por step.
    - learning_rate: tasa de aprendizaje base.
    - weight_decay: regularización L2 para parámetros con peso.
    - betas: coeficientes (beta1, beta2) de AdamW.
    - max_grad_norm: norma máxima para clip de gradiente (si > 0).
    - log_every: (opcional) cada cuántos steps loguear dentro de un epoch.
    - seed: semilla para reproducibilidad.
    - device: "auto" | "cpu" | "mps" | "cuda".
    """

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0

    log_every: int = 10
    seed: int = 42

    device: str = "auto"  # "auto" | "cpu" | "mps" | "cuda"

    def resolved_device(self) -> torch.device:
        """
        Devuelve un torch.device según:
          - valor explícito de self.device, o
          - autodetección de mps/cuda/cpu si device == "auto".
        """
        # Forzar CPU
        if self.device == "cpu":
            return torch.device("cpu")

        # Forzar MPS (Apple) si está disponible
        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")

        # Forzar CUDA si está disponible
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")

        # Modo auto
        if self.device == "auto":
            # Preferimos MPS, luego CUDA, luego CPU
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")

        # Fallback
        return torch.device("cpu")