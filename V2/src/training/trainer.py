# src/training/trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Any, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.training.losses import language_modeling_loss
from src.config.training_config import TrainingConfig


class Trainer:
    """
    Trainer genérico para Language Modeling (GPT pequeño).

    Asume:
      - los dataloaders devuelven (input_ids, target_ids) con shape (B,T)
      - el modelo devuelve:
          a) logits directamente (Tensor 3D: B,T,V)
          b) dict con 'logits'
          c) tuple/list con logits en alguna posición
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_cfg: TrainingConfig,
        ckpt_dir: str = "models/checkpoints",
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = None,
        # --- Debug / Guardrails ---
        debug_first_batch: bool = False,
        expected_vocab_size: Optional[int] = None,
        expected_seq_len: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = train_cfg

        # Debug knobs
        self.debug_first_batch = bool(debug_first_batch)
        self.expected_vocab_size = expected_vocab_size
        self.expected_seq_len = expected_seq_len
        self._did_debug = False

        # Device
        self.device = device if device is not None else train_cfg.resolved_device()

        # Grad clip
        if grad_clip is not None:
            self.grad_clip = grad_clip
        else:
            self.grad_clip = getattr(train_cfg, "max_grad_norm", None)

        # Checkpoints dir
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _move_batch_to_device(self, input_ids: Tensor, target_ids: Tensor) -> Tuple[Tensor, Tensor]:
        return input_ids.to(self.device), target_ids.to(self.device)

    def _extract_logits(self, out: Any) -> Tensor:
        """
        Extrae logits 3D (B,T,V) de distintas salidas.

        Regla clave (para tuple/list):
          - escoger el tensor 3D cuyo último dim (V) sea el MÁS GRANDE
            (logits suele tener V=vocab_size, que normalmente > d_model).
        """
        # dict-like output
        if isinstance(out, dict):
            if "logits" not in out:
                raise ValueError(f"Model output dict missing 'logits'. keys={list(out.keys())}")
            logits = out["logits"]
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise ValueError(
                    f"out['logits'] must be 3D Tensor. Got: {type(logits)} shape={getattr(logits,'shape',None)}"
                )
            return logits

        # direct tensor
        if isinstance(out, torch.Tensor):
            if out.ndim != 3:
                raise ValueError(f"Model output Tensor must be 3D (B,T,V). Got shape={tuple(out.shape)}")
            return out

        # tuple/list
        if isinstance(out, (tuple, list)):
            candidates: List[Tensor] = [x for x in out if isinstance(x, torch.Tensor) and x.ndim == 3]
            if not candidates:
                shapes: List[Union[tuple, str]] = []
                for x in out:
                    if isinstance(x, torch.Tensor):
                        shapes.append(tuple(x.shape))
                    else:
                        shapes.append(type(x).__name__)
                raise ValueError(
                    "No 3D tensor (B,T,V) found in model output tuple/list. "
                    f"Elements={shapes}"
                )

            best = max(candidates, key=lambda t: int(t.shape[-1]))

            if self.expected_vocab_size is not None:
                V = int(best.shape[-1])
                if V != int(self.expected_vocab_size):
                    cand_shapes = [tuple(t.shape) for t in candidates]
                    raise ValueError(
                        f"Extracted logits candidate has V={V}, expected_vocab_size={self.expected_vocab_size}. "
                        f"3D candidates={cand_shapes}"
                    )

            return best

        raise TypeError(f"Unsupported model output type: {type(out)}")

    def _debug_first_batch_checks(self, input_ids: Tensor, target_ids: Tensor, logits: Tensor) -> None:
        """
        Valida y reporta (una sola vez) que TODO esté consistente:
          - shapes, dtypes, rangos, V == vocab_size, etc.
        """
        if not self.debug_first_batch or self._did_debug:
            return
        self._did_debug = True

        # shapes
        if input_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError(
                f"Expected input/target 2D (B,T). Got input={tuple(input_ids.shape)} target={tuple(target_ids.shape)}"
            )
        B, T = input_ids.shape
        Bt, Tt = target_ids.shape
        if (B, T) != (Bt, Tt):
            raise ValueError(f"Input/target shape mismatch: input={(B, T)} target={(Bt, Tt)}")

        if self.expected_seq_len is not None and T != int(self.expected_seq_len):
            raise ValueError(f"seq_len mismatch: got T={T} expected_seq_len={self.expected_seq_len}")

        # dtypes (ideal para CE)
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must be torch.long. Got {input_ids.dtype}")
        if target_ids.dtype != torch.long:
            raise TypeError(f"target_ids must be torch.long. Got {target_ids.dtype}")

        # token id ranges
        imin = int(input_ids.min().item())
        imax = int(input_ids.max().item())
        tmin = int(target_ids.min().item())
        tmax = int(target_ids.max().item())

        if imin < 0 or tmin < 0:
            raise ValueError(f"Found negative token ids: input_min={imin} target_min={tmin}")

        if self.expected_vocab_size is not None:
            Vexp = int(self.expected_vocab_size)
            if imax >= Vexp or tmax >= Vexp:
                raise ValueError(
                    f"Token id out of range for vocab_size={Vexp}: input_max={imax} target_max={tmax}"
                )

        # logits shape
        if logits.ndim != 3:
            raise ValueError(f"logits must be 3D (B,T,V). Got {tuple(logits.shape)}")

        Bl, Tl, V = logits.shape
        if (Bl, Tl) != (B, T):
            raise ValueError(f"logits (B,T) mismatch: logits={(Bl, Tl)} vs input={(B, T)}")

        if self.expected_vocab_size is not None and V != int(self.expected_vocab_size):
            raise ValueError(f"logits vocab dim mismatch: got V={V}, expected={self.expected_vocab_size}")

        # stats + CE sanity
        with torch.no_grad():
            logits_min = float(logits.min().item())
            logits_max = float(logits.max().item())
            logits_mean = float(logits.mean().item())
            logits_std = float(logits.std().item())

            # CE sanity en un slice
            logits_2d = logits.reshape(B * T, V)
            targets_1d = target_ids.reshape(B * T)
            n = min(1024, targets_1d.numel())
            ce_slice = float(F.cross_entropy(logits_2d[:n], targets_1d[:n], reduction="mean").item())

        print(
            "[DEBUG first batch]\n"
            f"  device: {self.device}\n"
            f"  input_ids:  shape={tuple(input_ids.shape)} dtype={input_ids.dtype} min={imin} max={imax}\n"
            f"  target_ids: shape={tuple(target_ids.shape)} dtype={target_ids.dtype} min={tmin} max={tmax}\n"
            f"  logits:     shape={tuple(logits.shape)} dtype={logits.dtype}\n"
            f"  logits stats: min={logits_min:.4f} max={logits_max:.4f} mean={logits_mean:.4f} std={logits_std:.4f}\n"
            f"  CE sanity (first {n} tokens): {ce_slice:.4f}\n"
        )

    def _step(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        out = self.model(input_ids)
        logits = self._extract_logits(out)
        self._debug_first_batch_checks(input_ids, target_ids, logits)
        loss = language_modeling_loss(logits, target_ids)
        return loss

    # ---------------------------------------------------------
    # Train step
    # ---------------------------------------------------------
    def train_step(self, input_ids: Tensor, target_ids: Tensor) -> float:
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
    # Epoch (compat)
    # ---------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for input_ids, target_ids in dataloader:
            total_loss += self.train_step(input_ids, target_ids)
            n += 1
        return total_loss / max(1, n)

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        logger=None,
        log_every: Optional[int] = None,
        max_batches: Optional[int] = None,
        average_by: str = "token",  # "batch" o "token"
    ) -> float:
        self.model.eval()
        if len(dataloader) == 0:
            return 0.0

        total_batches = len(dataloader)
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        total_loss_sum = 0.0
        total_batches_count = 0

        total_token_loss_sum = 0.0
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader, start=1):
            if batch_idx > total_batches:
                break

            input_ids, target_ids = batch
            input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

            loss = self._step(input_ids, target_ids)
            loss_val = float(loss.detach().item())

            if average_by == "token":
                num_tok = int(target_ids.numel())
                total_token_loss_sum += loss_val * num_tok
                total_tokens += num_tok
            else:
                total_loss_sum += loss_val
                total_batches_count += 1

            if logger is not None and log_every and batch_idx % log_every == 0:
                progress = 100.0 * batch_idx / total_batches
                logger.info(f"[val {batch_idx}/{total_batches} ({progress:.2f}%)] loss={loss_val:.4f}")

        if average_by == "token":
            return total_token_loss_sum / max(1, total_tokens)
        return total_loss_sum / max(1, total_batches_count)

    # ---------------------------------------------------------
    # Checkpoints
    # ---------------------------------------------------------
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        global_step: int,
        val_loss: Optional[float] = None,
    ) -> Path:
        ckpt_path = self.ckpt_dir / filename

        # ---- training_config (as dict) ----
        training_config: Dict[str, Any] = self.config.__dict__

        # ---- model_config (self-contained for inference) ----
        # We try to read a config object from the model. Common patterns:
        #   - model.config
        #   - model.cfg
        # If none exists, we gracefully store an empty dict.
        model_cfg = getattr(self.model, "config", None)
        if model_cfg is None:
            model_cfg = getattr(self.model, "cfg", None)

        model_config: Dict[str, Any] = {}
        if model_cfg is not None:
            # NOTE: init_std and tie_weights are especially useful for reproducibility and
            #       for documenting that init is linked to weight tying in GPT-style models.
            model_config = {
                "vocab_size": int(getattr(model_cfg, "vocab_size")),
                "d_model": int(getattr(model_cfg, "d_model")),
                "n_layers": int(getattr(model_cfg, "n_layers")),
                "n_heads": int(getattr(model_cfg, "n_heads")),
                "max_seq_len": int(getattr(model_cfg, "max_seq_len")),
                "dropout": float(getattr(model_cfg, "dropout")),
                "init_std": float(getattr(model_cfg, "init_std", 0.02)),
                "tie_weights": bool(getattr(model_cfg, "tie_weights", True)),
            }

        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "training_config": training_config,
            # ✅ NEW: make ckpt self-contained for inference
            "model_config": model_config,
        }

        torch.save(payload, ckpt_path)
        return ckpt_path