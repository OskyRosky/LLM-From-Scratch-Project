# src/training/trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.training.losses import language_modeling_loss
from src.config.training_config import TrainingConfig


class Trainer:
    """
    Entrenador genérico para language modeling (GPT pequeño).

    Supone que:
      - el modelo devuelve (logits, attn) en el forward,
      - los dataloaders devuelven (input_ids, target_ids) de enteros.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        ckpt_dir: str = "models/checkpoints",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config

        # --------- Dispositivo ---------
        # Usamos la lógica centralizada en TrainingConfig
        self.device: torch.device = self.config.resolved_device()
        self.model.to(self.device)

        # Gradiente máximo (para clipping)
        self.grad_clip: Optional[float] = getattr(self.config, "max_grad_norm", None)

        # --------- Checkpoints ---------
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    #  Helpers
    # ---------------------------------------------------------
    def _move_batch_to_device(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Mueve un batch al dispositivo de entrenamiento."""
        return input_ids.to(self.device), target_ids.to(self.device)

    def _step(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        Un paso de forward + cálculo de loss (sin backward ni optimizer).

        Devuelve la loss (tensor escalar).
        """
        logits, _ = self.model(input_ids)  # (B, T, vocab)
        loss = language_modeling_loss(logits, target_ids)
        return loss

    # ---------------------------------------------------------
    #  Épocas
    # ---------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Entrena el modelo sobre un dataloader (una época completa).

        Devuelve la loss media.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader, start=1):
            input_ids, target_ids = batch
            input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self._step(input_ids, target_ids)
            loss.backward()

            # Clipping de gradiente (si está configurado)
            if self.grad_clip is not None and self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Logging opcional
            if self.config.log_every and step % self.config.log_every == 0:
                print(f"[train] step {step}  loss={loss.item():.4f}")

        mean_loss = total_loss / max(1, num_batches)
        return mean_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evalúa el modelo sobre un dataloader (sin gradientes).

        Devuelve la loss media.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader, start=1):
            input_ids, target_ids = batch
            input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

            loss = self._step(input_ids, target_ids)

            total_loss += loss.item()
            num_batches += 1

        mean_loss = total_loss / max(1, num_batches)
        return mean_loss

    # ---------------------------------------------------------
    #  Checkpoints
    # ---------------------------------------------------------
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        global_step: int,
        val_loss: Optional[float] = None,
    ) -> Path:
        """
        Guarda un checkpoint sencillo en self.ckpt_dir.

        Devuelve la ruta final del archivo.
        """
        ckpt_path = self.ckpt_dir / filename

        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "training_config": self.config.__dict__,
        }

        torch.save(payload, ckpt_path)
        return ckpt_path