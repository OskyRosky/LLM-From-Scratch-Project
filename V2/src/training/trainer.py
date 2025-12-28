# src/training/trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Any

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.training.losses import language_modeling_loss
from src.config.training_config import TrainingConfig


class Trainer:
    """
    Entrenador genérico para language modeling (GPT pequeño).

    Asume que:
      - los dataloaders devuelven (input_ids, target_ids) de enteros.
      - el modelo puede devolver:
          a) logits directamente (Tensor 3D: B,T,V)
          b) una tupla/lista que contiene logits en algún lugar
             (por ejemplo (logits, attn) o (loss, logits, extra)).
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
        self.config = train_cfg

        # Dispositivo
        if device is not None:
            self.device = device
        else:
            self.device = train_cfg.resolved_device()

        # Grad clip (usa max_grad_norm si existe en el config)
        if grad_clip is not None:
            self.grad_clip = grad_clip
        else:
            self.grad_clip = getattr(train_cfg, "max_grad_norm", None)

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
    ) -> Tuple[Tensor, Tensor]:
        return input_ids.to(self.device), target_ids.to(self.device)

    def _extract_logits(self, out: Any) -> Tensor:
        """
        Extrae logits (Tensor 3D: B, T, V) de la salida del modelo.

        Casos soportados:
          - out es Tensor 3D -> logits
          - out es tuple/list -> busca el primer Tensor 3D
        """
        # Caso 1: el modelo devuelve directamente logits
        if isinstance(out, torch.Tensor):
            if out.ndim != 3:
                raise ValueError(
                    "Salida del modelo es Tensor pero no es 3D (B,T,V). "
                    f"shape={tuple(out.shape)}"
                )
            return out

        # Caso 2: el modelo devuelve tuple/list (buscar el primer tensor 3D)
        if isinstance(out, (tuple, list)):
            for x in out:
                if isinstance(x, torch.Tensor) and x.ndim == 3:
                    return x

            shapes: List[Union[tuple, str]] = []
            for x in out:
                if isinstance(x, torch.Tensor):
                    shapes.append(tuple(x.shape))
                else:
                    shapes.append(type(x).__name__)
            raise ValueError(
                "No encontré logits 3D (B,T,V) en la salida del modelo. "
                f"Elementos={shapes}"
            )

        raise TypeError(f"Salida del modelo no soportada: {type(out)}")

    def _step(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        Un paso de forward + loss (sin backward ni optimizer).
        Devuelve la loss (scalar tensor).
        """
        out = self.model(input_ids)
        logits = self._extract_logits(out)  # (B, T, V)
        loss = language_modeling_loss(logits, target_ids)
        return loss

    # ---------------------------------------------------------
    #  Paso individual (lo que usará pretrain_gpt / pretrain_gpt_tokens)
    # ---------------------------------------------------------
    def train_step(self, input_ids: Tensor, target_ids: Tensor) -> float:
        """
        Un paso de entrenamiento (un batch):
          - mueve a device
          - forward
          - backward
          - clip grad (si aplica)
          - optimizer.step()

        Devuelve la loss como float.
        """
        self.model.train()
        input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

        self.optimizer.zero_grad(set_to_none=True)
        loss = self._step(input_ids, target_ids)
        loss.backward()

        if self.grad_clip is not None and self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return float(loss.detach().item())

    # ---------------------------------------------------------
    #  Época completa (compatibilidad)
    # ---------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Entrena el modelo sobre un dataloader (una época completa).
        Devuelve la loss media.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in dataloader:
            batch_loss = self.train_step(input_ids, target_ids)
            total_loss += batch_loss
            num_batches += 1

        mean_loss = total_loss / max(1, num_batches)
        return mean_loss

    # ---------------------------------------------------------
    #  Evaluación
    # ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        logger=None,
        log_every: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> float:
        """
        Evalúa el modelo sobre un dataloader (sin gradientes).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        if len(dataloader) == 0:
            return 0.0

        total_batches = len(dataloader)
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        for batch_idx, batch in enumerate(dataloader, start=1):
            if batch_idx > total_batches:
                break

            input_ids, target_ids = batch
            input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

            loss = self._step(input_ids, target_ids)
            loss_val = float(loss.detach().item())

            total_loss += loss_val
            num_batches += 1

            if logger is not None and log_every and batch_idx % log_every == 0:
                progress = 100.0 * batch_idx / total_batches
                logger.info(
                    f"[val {batch_idx}/{total_batches} ({progress:.2f}%)] loss={loss_val:.4f}"
                )

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