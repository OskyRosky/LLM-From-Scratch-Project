# src/cli/pretrain_gpt_tokens.py

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from torch.utils.data import DataLoader, random_split

from src.config.training_config import TrainingConfig
from src.data.datasets import TokenBinDataset
from src.model.gpt import GPTConfig, GPTModel
from src.training.optimizer import create_optimizer
from src.training.trainer import Trainer
from src.training.evaluation import loss_to_perplexity
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger


# -------------------------
# Helpers
# -------------------------
def load_meta(meta_path: str) -> Dict[str, Any]:
    p = Path(meta_path)
    if not p.exists():
        raise FileNotFoundError(f"meta.json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if "bin_file" not in meta:
        raise KeyError("meta.json must contain key 'bin_file'")
    if "vocab_size" not in meta:
        raise KeyError("meta.json must contain key 'vocab_size'")

    return meta


def resolve_tokens_bin(meta_path: str, bin_file_value: str) -> str:
    """
    Resuelve el path de tokens.bin:
    - Si meta['bin_file'] es absoluto -> se usa directo
    - Si es relativo -> se resuelve relativo al directorio del meta.json
    """
    meta_dir = Path(meta_path).resolve().parent
    p = Path(bin_file_value)
    if p.is_absolute():
        return str(p)
    return str((meta_dir / p).resolve())


def build_dataloaders_from_bin(
    tokens_bin: str,
    seq_len: int,
    dtype: str,
    batch_size: int,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    ds = TokenBinDataset(tokens_bin_path=tokens_bin, seq_len=seq_len, dtype=dtype)

    n_total = len(ds)
    n_val = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(
            f"Dataset demasiado pequeño para val_ratio={val_ratio}. "
            f"len(ds)={n_total} => n_train={n_train}, n_val={n_val}"
        )

    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, val_loader


def mean(values) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def is_finite_number(x: Optional[float]) -> bool:
    if x is None:
        return False
    try:
        xf = float(x)
    except Exception:
        return False
    return math.isfinite(xf)


def format_ppl(loss_value: Optional[float], ppl_threshold: float) -> str:
    """
    Devuelve un string amigable para perplexity:
    - si loss is None -> "NA"
    - si loss no es finita -> "NA (nan/inf)"
    - si loss > threshold -> "NA (loss>threshold)"
    - si loss <= threshold -> número con 2 decimales
    """
    if loss_value is None:
        return "NA"

    try:
        lf = float(loss_value)
    except Exception:
        return "NA"

    if not math.isfinite(lf):
        return "NA (nan/inf)"

    if lf > ppl_threshold:
        return f"NA (loss>{ppl_threshold:g})"

    ppl = loss_to_perplexity(lf)
    if ppl != ppl:  # NaN
        return "NA"
    if ppl == float("inf"):
        return "NA (overflow)"
    return f"{ppl:.2f}"


def dump_run_config(
    ckpt_dir: Path,
    args: argparse.Namespace,
    meta_path: str,
    tokens_bin: str,
    vocab_size: int,
    dtype: str,
    gpt_cfg: GPTConfig,
    train_cfg: TrainingConfig,
) -> None:
    payload: Dict[str, Any] = {
        "meta_path": str(Path(meta_path).resolve()),
        "tokens_bin": str(Path(tokens_bin).resolve()),
        "vocab_size": int(vocab_size),
        "dtype": dtype,
        "args": vars(args),
        "gpt_config": {
            "vocab_size": gpt_cfg.vocab_size,
            "max_seq_len": gpt_cfg.max_seq_len,
            "d_model": gpt_cfg.d_model,
            "n_heads": gpt_cfg.n_heads,
            "n_layers": gpt_cfg.n_layers,
            "dropout": gpt_cfg.dropout,
        },
        "training_config": dict(train_cfg.__dict__),
    }

    with (ckpt_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain GPT from token ids (BPE tokens.bin)."
    )
    parser.add_argument("--meta", type=str, required=True, help="Path to meta.json")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps")
    parser.add_argument("--max_epochs", type=int, default=1, help="Max epochs")
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|mps|auto")
    parser.add_argument("--ckpt_dir", type=str, default="models/checkpoints/pretrain_bpe_v4")

    # model size
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # validation
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio (0.0-0.5)")
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=300,
        help="Max validation batches (use small number for smoke tests)",
    )

    # logging stability
    parser.add_argument("--train_avg_window", type=int, default=50, help="Moving average window")
    parser.add_argument(
        "--report_ppl_when_loss_leq",
        type=float,
        default=20.0,
        help="Only report perplexity when loss <= this threshold",
    )

    args = parser.parse_args()
    logger = get_logger("pretrain_gpt_tokens")

    # Basic arg validation
    if not (0.0 <= args.val_ratio <= 0.5):
        raise ValueError("--val_ratio debe estar entre 0.0 y 0.5")
    if args.train_avg_window <= 0:
        raise ValueError("--train_avg_window debe ser >= 1")
    if args.max_val_batches <= 0:
        raise ValueError("--max_val_batches debe ser >= 1")

    # 1) Meta + paths
    meta = load_meta(args.meta)
    tokens_bin = resolve_tokens_bin(args.meta, meta["bin_file"])
    vocab_size = int(meta["vocab_size"])
    dtype = meta.get("dtype", "uint16")

    # 2) Training config
    train_cfg = TrainingConfig(batch_size=args.batch_size, device=args.device)
    set_seed(train_cfg.seed)
    logger.info(f"Using device: {train_cfg.resolved_device()}")

    # 3) Dataloaders
    logger.info(f"Loading token dataset: {tokens_bin}")
    train_loader, val_loader = build_dataloaders_from_bin(
        tokens_bin=tokens_bin,
        seq_len=args.seq_len,
        dtype=dtype,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )
    logger.info(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")

    # 4) Model
    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = GPTModel(gpt_cfg)

    # 5) Optimizer + Trainer
    optimizer = create_optimizer(model, train_cfg)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer/meta footprint
    with (ckpt_dir / "tokenizer_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Save run config (args + gpt_cfg + train_cfg)
    dump_run_config(
        ckpt_dir=ckpt_dir,
        args=args,
        meta_path=args.meta,
        tokens_bin=tokens_bin,
        vocab_size=vocab_size,
        dtype=dtype,
        gpt_cfg=gpt_cfg,
        train_cfg=train_cfg,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        ckpt_dir=str(ckpt_dir),
    )

    # 6) Train loop
    global_step = 0
    max_steps = int(args.max_steps)
    max_epochs = int(args.max_epochs)
   # logging frequency (prefer % of run)
    default_log_every = max(1, int(round(max_steps * 0.01)))  # 1% por defecto
    log_every = getattr(train_cfg, "log_every", default_log_every)

    logger.info(
        f"Starting token pretraining: vocab_size={vocab_size}, seq_len={args.seq_len}, "
        f"d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}"
    )

    recent_losses = deque(maxlen=int(args.train_avg_window))
    last_train_loss: Optional[float] = None
    aborted = False

    try:
        for epoch in range(max_epochs):
            if global_step >= max_steps:
                break

            for batch in train_loader:
                if global_step >= max_steps:
                    break

                input_ids, target_ids = batch
                global_step += 1

                loss = trainer.train_step(input_ids, target_ids)

                # Guardrail: train loss must be finite
                if not is_finite_number(loss):
                    logger.error(f"[FATAL] Non-finite train loss at step={global_step}: loss={loss!r}")
                    trainer.save_checkpoint(
                        filename=f"ckpt_nan_step_{global_step}.pt",
                        epoch=epoch,
                        global_step=global_step,
                        val_loss=None,
                    )
                    aborted = True
                    raise SystemExit(1)

                last_train_loss = float(loss)
                recent_losses.append(last_train_loss)

                if log_every and global_step % log_every == 0:
                    progress = 100.0 * global_step / max_steps

                    loss_avg = mean(recent_losses)
                    # Guardrail: loss_avg should also be finite (if computed)
                    if loss_avg is not None and not is_finite_number(loss_avg):
                        logger.error(
                            f"[FATAL] Non-finite loss_avg at step={global_step}: loss_avg={loss_avg!r}"
                        )
                        trainer.save_checkpoint(
                            filename=f"ckpt_nan_avg_step_{global_step}.pt",
                            epoch=epoch,
                            global_step=global_step,
                            val_loss=None,
                        )
                        aborted = True
                        raise SystemExit(1)

                    ppl_avg_str = format_ppl(loss_avg, args.report_ppl_when_loss_leq)

                    logger.info(
                        f"[step {global_step}/{max_steps} ({progress:.2f}%)] "
                        f"loss_last={last_train_loss:.4f} "
                        f"loss_avg_{len(recent_losses)}={loss_avg:.4f} "
                        f"train_ppl_avg_{len(recent_losses)}={ppl_avg_str}"
                    )

    finally:
        if aborted:
            logger.info("Aborted run due to non-finite values (checkpoint saved).")

    logger.info("Finished training loop, running final validation...")

    # 7) Validation (bounded)
    if len(val_loader) > 0:
        max_val_batches = min(len(val_loader), int(args.max_val_batches))
        val_log_every = max(1, max_val_batches // 10)  # más denso para smoke

        val_loss = trainer.evaluate(
            val_loader,
            logger=logger,
            log_every=val_log_every,
            max_batches=max_val_batches,
        )
    else:
        val_loss = float("nan")

    # Guardrail: val_loss must be finite
    if not is_finite_number(val_loss):
        logger.error(f"[FATAL] Non-finite val loss after evaluation: val_loss={val_loss!r}")
        trainer.save_checkpoint(
            filename=f"ckpt_nan_val_step_{global_step}.pt",
            epoch=max_epochs - 1,
            global_step=global_step,
            val_loss=None,
        )
        raise SystemExit(1)

    # 8) Final metrics
    train_loss_for_ppl = mean(recent_losses)
    if train_loss_for_ppl is None:
        train_loss_for_ppl = last_train_loss if last_train_loss is not None else val_loss

    train_ppl_str = format_ppl(train_loss_for_ppl, args.report_ppl_when_loss_leq)
    val_ppl_str = format_ppl(val_loss, args.report_ppl_when_loss_leq)

    logger.info(
        f"Final train_loss_for_ppl={float(train_loss_for_ppl):.4f} train_ppl={train_ppl_str} "
        f"val_loss={float(val_loss):.4f} val_ppl={val_ppl_str}"
    )

    # 9) Always save final checkpoint
    final_ckpt = trainer.save_checkpoint(
        filename=f"ckpt_final_step_{global_step}.pt",
        epoch=max_epochs - 1,
        global_step=global_step,
        val_loss=float(val_loss),
    )
    logger.info(f"Saved final checkpoint: {str(final_ckpt)}")
    logger.info(f"Checkpoint dir: {str(ckpt_dir)}")


if __name__ == "__main__":
    main()