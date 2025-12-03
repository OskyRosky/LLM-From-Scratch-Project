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

    Asume que:
      - el modelo devuelve (logits, attn) en el forward,
      - los dataloaders devuelven (input_ids, target_ids) de enteros.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_cfg: TrainingConfig,
        ckpt_dir: str = "models/checkpoints",
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_cfg = train_cfg

        # Dispositivo
        if device is None:
            device_str = train_cfg.resolved_device()
            self.device = torch.device(device_str)
        else:
            self.device = device

        # Grad clip: si no se pasa, intentamos leerlo del config
        if grad_clip is not None:
            self.grad_clip = grad_clip
        else:
            self.grad_clip = getattr(train_cfg, "grad_clip", None)

        # Directorio de checkpoints
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Enviar modelo al dispositivo
        self.model.to(self.device)

    # ---------------------------------------------------------
    #  Helpers
    # ---------------------------------------------------------
    def _move_batch_to_device(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        return input_ids.to(self.device), target_ids.to(self.device)

    def _step(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        Un paso de forward + loss (sin backward ni optimizer).
        Devuelve la loss (scalar tensor).
        """
        logits, _ = self.model(input_ids)  # (B, T, vocab)
        loss = language_modeling_loss(logits, target_ids)
        return loss

    # ---------------------------------------------------------
    #  EPCHAS
    # ---------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Entrena el modelo sobre un dataloader (una época completa).

        Devuelve la loss media.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids, target_ids = batch
            input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self._step(input_ids, target_ids)
            loss.backward()

            if self.grad_clip is not None and self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

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

        for batch in dataloader:
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
            "training_config": self.train_cfg.__dict__,
        }

        torch.save(payload, ckpt_path)
        return ckpt_path