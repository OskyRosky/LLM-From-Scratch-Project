# src/cli/pretrain_gpt_tokens.py

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain GPT from token ids (BPE tokens.bin)."
    )
    parser.add_argument(
        "--meta",
        type=str,
        required=True,
        help="Path to meta.json (from build_token_ids)",
    )
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Max epochs (stops earlier if max_steps hit)",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|mps|auto")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="models/checkpoints/pretrain_bpe_v4",
        help="Checkpoint dir",
    )

    # model size
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # validation
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio (0.0-0.5)")
    args = parser.parse_args()

    logger = get_logger("pretrain_gpt_tokens")

    # Validación de val_ratio
    if not (0.0 <= args.val_ratio <= 0.5):
        raise ValueError("--val_ratio debe estar entre 0.0 y 0.5")

    # 1) Cargar meta
    meta = load_meta(args.meta)
    tokens_bin = resolve_tokens_bin(args.meta, meta["bin_file"])
    vocab_size = int(meta["vocab_size"])
    dtype = meta.get("dtype", "uint16")

    # 2) Training config
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        device=args.device,
    )
    set_seed(train_cfg.seed)
    logger.info(f"Using device: {train_cfg.resolved_device()}")

    # 3) Dataloaders desde tokens.bin
    logger.info(f"Loading token dataset: {tokens_bin}")
    train_loader, val_loader = build_dataloaders_from_bin(
        tokens_bin=tokens_bin,
        seq_len=args.seq_len,
        dtype=dtype,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )
    logger.info(f"train_batches={len(train_loader)} val_batches={len(val_loader)}")

    # 4) Modelo GPT token-level
    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = GPTModel(gpt_cfg)

    # 5) Optimizador + Trainer
    optimizer = create_optimizer(model, train_cfg)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Guardar una "huella" del tokenizer/meta asociado al pretraining
    with (ckpt_dir / "tokenizer_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        ckpt_dir=str(ckpt_dir),
    )

    # 6) Loop de training por steps
    global_step = 0
    max_steps = int(args.max_steps)
    max_epochs = int(args.max_epochs)
    log_every = getattr(train_cfg, "log_every", 10)

    logger.info(
        f"Starting token pretraining: vocab_size={vocab_size}, seq_len={args.seq_len}, "
        f"d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}"
    )

    last_train_loss: Optional[float] = None

    for epoch in range(max_epochs):
        if global_step >= max_steps:
            break

        for batch in train_loader:
            if global_step >= max_steps:
                break

            input_ids, target_ids = batch
            global_step += 1

            loss = trainer.train_step(input_ids, target_ids)
            last_train_loss = loss

            if log_every and global_step % log_every == 0:
                progress = 100.0 * global_step / max_steps
                logger.info(f"[step {global_step}/{max_steps} ({progress:.2f}%)] loss={loss:.4f}")

    logger.info("Finished training loop, running final validation...")

    # 7) Evaluación final
    if len(val_loader) > 0:
        max_val_batches = min(len(val_loader), 300)
        val_log_every = max(1, max_val_batches // 50)
        val_loss = trainer.evaluate(
            val_loader,
            logger=logger,
            log_every=val_log_every,
            max_batches=max_val_batches,
        )
    else:
        val_loss = float("nan")

    # 8) Perplexity (blindado en evaluation.loss_to_perplexity)
    train_loss_for_ppl = last_train_loss if last_train_loss is not None else val_loss
    train_ppl = loss_to_perplexity(train_loss_for_ppl)
    val_ppl = loss_to_perplexity(val_loss)

    logger.info(f"Final train_ppl={train_ppl:.2f} val_ppl={val_ppl:.2f}")
    logger.info(f"Checkpoint dir: {str(ckpt_dir)}")


if __name__ == "__main__":
    main()