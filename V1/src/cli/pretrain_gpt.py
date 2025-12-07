# src/cli/pretrain_gpt.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.config.training_config import TrainingConfig
from src.data.text_preprocessing import load_text, basic_clean
from src.data.tokenization import CharacterTokenizer
from src.data.datasets import CharacterDataset
from src.model.gpt import GPTConfig, GPTModel
from src.training.optimizer import create_optimizer
from src.training.trainer import Trainer
from src.training.evaluation import loss_to_perplexity
from src.utils.seed import set_seed
from src.utils.logging_utils import get_logger


# ---------------------------------------------------------
#  Helpers para datos
# ---------------------------------------------------------
def build_datasets_and_tokenizer(
    text_path: str,
    seq_len: int,
) -> Tuple[CharacterTokenizer, CharacterDataset, CharacterDataset]:
    """
    Carga texto, limpia, tokeniza a nivel carácter y
    crea datasets de train / val.
    """
    # 1) Cargar texto crudo y limpiar
    raw_text = load_text(text_path)
    clean_text = basic_clean(raw_text)

    # 2) Entrenar tokenizador de caracteres
    tokenizer = CharacterTokenizer()
    tokenizer.train(clean_text)

    # 3) Convertir texto completo en ids
    ids = tokenizer.encode(clean_text, add_special_tokens=False)
    ids = torch.tensor(ids, dtype=torch.long)

    # 4) Split train/val simple (90/10)
    n_total = ids.size(0)
    n_train = int(0.9 * n_total)
    if n_train <= seq_len + 1 or n_total <= seq_len + 1:
        raise ValueError(
            f"Texto demasiado pequeño para seq_len={seq_len}. "
            f"n_total={n_total}, n_train={n_train}"
        )

    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    train_ds = CharacterDataset(train_ids.tolist(), seq_len=seq_len)
    val_ds = CharacterDataset(val_ids.tolist(), seq_len=seq_len)

    return tokenizer, train_ds, val_ds


def build_dataloaders(
    train_ds: CharacterDataset,
    val_ds: CharacterDataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def save_tokenizer(tokenizer: CharacterTokenizer, path: str) -> None:
    """
    Guarda el tokenizador de forma sencilla (stoi / itos) usando torch.save.
    """
    payload = {
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
    }
    torch.save(payload, path)


# ---------------------------------------------------------
#  Entrenamiento principal
# ---------------------------------------------------------
def run_training(args: argparse.Namespace) -> None:
    logger = get_logger("pretrain_gpt")

    # 1) Config de entrenamiento general (usa batch_size y device del CLI)
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        device=args.device,
    )
    set_seed(train_cfg.seed)

    logger.info(f"Using device: {train_cfg.resolved_device()}")

    # 2) Datos + tokenizador
    logger.info(f"Loading and preparing data from {args.data_path}")
    tokenizer, train_ds, val_ds = build_datasets_and_tokenizer(
        text_path=args.data_path,
        seq_len=args.seq_len,
    )

    train_loader, val_loader = build_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=args.batch_size,
    )

    # 3) Config y modelo GPT (pequeño)
    vocab_size = tokenizer.vocab_size
    logger.info(
        f"Vocab size = {vocab_size}, seq_len = {args.seq_len}, "
        f"d_model = {args.d_model}, n_layers = {args.n_layers}, n_heads = {args.n_heads}"
    )

    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = GPTModel(gpt_cfg)

    # 4) Optimizador
    optimizer = create_optimizer(model, train_cfg)

    # 5) Trainer
    ckpt_dir = args.ckpt_dir
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        ckpt_dir=ckpt_dir,
    )

    # 6) Entrenamiento controlado por max_steps
    global_step = 0
    best_val_loss = float("inf")  # por ahora no lo usamos, pero lo dejamos
    max_steps = args.max_steps
    max_epochs = args.max_epochs

    log_every = getattr(train_cfg, "log_every", 10)

    logger.info(
        f"Starting training for up to {max_epochs} epochs "
        f"or {max_steps} steps (lo que ocurra primero)."
    )

    last_epoch_used = 0
    last_train_loss = None

    for epoch in range(max_epochs):
        if global_step >= max_steps:
            break

        for batch in train_loader:
            if global_step >= max_steps:
                break

            input_ids, target_ids = batch
            global_step += 1

            # Un paso de entrenamiento
            loss = trainer.train_step(input_ids, target_ids)
            last_train_loss = loss
            last_epoch_used = epoch

            # Logueo periódico con porcentaje
            if log_every and global_step % log_every == 0:
                progress = 100.0 * global_step / max_steps
                logger.info(
                    f"[step {global_step}/{max_steps} "
                    f"({progress:.2f}%)] loss={loss:.4f}"
                )

        # Si ya llegamos a max_steps salimos del loop externo
        if global_step >= max_steps:
            break

    logger.info("Finished training loop, running final validation...")

    # 7) Evaluación final sobre el conjunto de validación
    #    Usamos solo una muestra de batches para no tardar horas,
    #    pero con logs más frecuentes (~0.5% de avance).
    if len(val_loader) > 0:
        max_val_batches = min(len(val_loader), 1000)  # seguimos limitando a 1000
        # Queremos ~0.5% de resolución -> ~200 logs
        target_log_percent = 0.5
        num_logs = max(1, int(100 / target_log_percent))  # 200
        val_log_every = max(1, max_val_batches // num_logs)
    else:
        max_val_batches = 0
        val_log_every = 1

    val_loss = trainer.evaluate(
        val_loader,
        logger=logger,
        log_every=val_log_every,
        max_batches=max_val_batches,
    )

    train_ppl = loss_to_perplexity(last_train_loss if last_train_loss is not None else val_loss)
    val_ppl = loss_to_perplexity(val_loss)

    logger.info(
        f"[Final Eval] train_loss={last_train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"train_ppl={train_ppl:.2f} "
        f"val_ppl={val_ppl:.2f} "
        f"global_step={global_step}"
    )

    # Guardamos un único checkpoint "best" (es el último)
    ckpt_path = trainer.save_checkpoint(
        filename="gpt_char_best.pt",
        epoch=last_epoch_used,
        global_step=global_step,
        val_loss=val_loss,
    )
    logger.info(f"Checkpoint saved at: {ckpt_path}")

    # 8) Guardar tokenizador usable luego
    tok_path = Path(ckpt_dir) / "char_tokenizer.pt"
    save_tokenizer(tokenizer, str(tok_path))
    logger.info(f"Tokenizer saved at: {tok_path}")


# ---------------------------------------------------------
#  CLI
# ---------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-entrenamiento ligero de un GPT de caracteres."
    )

    # Datos
    p.add_argument(
        "--data-path",
        type=str,
        default="data/raw/tiny_corpus.txt",
        help="Ruta al archivo de texto de entrenamiento.",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Longitud de secuencia (contexto) en tokens/caracteres.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamaño de batch.",
    )

    # Modelo
    p.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Dimensión del embedding / modelo.",
    )
    p.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Número de cabezas de atención.",
    )
    p.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Número de bloques Transformer.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout del modelo.",
    )

    # Entrenamiento
    p.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Número máximo de steps de entrenamiento.",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Número máximo de épocas.",
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default="models/checkpoints",
        help="Directorio donde guardar checkpoints y tokenizador.",
    )

    # Dispositivo
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Dispositivo para entrenar el modelo (auto|cpu|mps|cuda).",
    )

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()