import argparse
import os
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.gpt import GPTConfig, GPTModel
from src.model.classification import GPTForClassification
from src.data.datasets import ClassificationDataset, ClassificationExample


# -----------------------------
# Utilidades de dispositivo
# -----------------------------
def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")


# -----------------------------
# Tokenizer mínimo desde char_tokenizer.pt
# -----------------------------
class CharTokenizerFromState:
    """
    Pequeño wrapper alrededor de stoi para reutilizar el vocabulario
    de caracteres del pretraining.
    """

    def __init__(self, stoi):
        self.stoi = stoi

    def encode(self, text: str):
        # Usamos un ID por defecto para caracteres desconocidos
        # Si existe "<unk>", lo usamos; si no, tomamos el primer ID de stoi.
        default_id = self.stoi.get("<unk>", next(iter(self.stoi.values())))
        return [self.stoi.get(ch, default_id) for ch in text]


# -----------------------------
# Dataset de ejemplo (toy)
# -----------------------------
def build_toy_examples() -> List[ClassificationExample]:
    """
    Crea un dataset mínimo de juguete para probar el finetuning.

    Ejemplo: clasificación binaria 0/1 (negativo/positivo).
    """
    examples = [
        ClassificationExample(
            text="Odio este producto, es terrible y no funciona nada bien.",
            label=0,
        ),
        ClassificationExample(
            text="Muy mala experiencia, no lo recomiendo a nadie.",
            label=0,
        ),
        ClassificationExample(
            text="Es un servicio pésimo, me siento estafado.",
            label=0,
        ),
        ClassificationExample(
            text="Me encanta este modelo, es excelente y funciona perfecto.",
            label=1,
        ),
        ClassificationExample(
            text="La calidad es muy buena, estoy muy satisfecho.",
            label=1,
        ),
        ClassificationExample(
            text="Estoy feliz con la compra, lo volvería a comprar.",
            label=1,
        ),
        ClassificationExample(
            text="El soporte técnico fue muy amable y resolvió todo.",
            label=1,
        ),
        ClassificationExample(
            text="No me gustó tanto, pero al menos funciona.",
            label=0,
        ),
    ]
    return examples


# -----------------------------
# Train / Eval helpers
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids)  # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_idx % 10 == 0 or batch_idx == len(dataloader):
            print(
                f"[train] batch {batch_idx}/{len(dataloader)} - "
                f"loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    torch.set_grad_enabled(False)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        total_batches += 1

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)

    avg_loss = total_loss / max(total_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Finetuning de GPT de caracteres para clasificación de texto."
    )

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="models/checkpoints_oscar_long",
        help="Directorio con gpt_char_best.pt y char_tokenizer.pt",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Número de clases de salida.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Dispositivo a usar (auto|cpu|mps|cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Tamaño de batch.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Número máximo de épocas.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[INFO] Usando dispositivo: {device}")

    # -------------------------
    # 1. Paths del checkpoint y tokenizer
    # -------------------------
    ckpt_path = os.path.join(args.ckpt_dir, "gpt_char_best.pt")
    tok_path = os.path.join(args.ckpt_dir, "char_tokenizer.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No se encontró el checkpoint en {ckpt_path}")
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(f"No se encontró el tokenizer en {tok_path}")

    # -------------------------
    # 2. Cargar tokenizer y checkpoint
    # -------------------------
    print(f"[INFO] Cargando tokenizer desde: {tok_path}")
    tok_state = torch.load(tok_path, map_location="cpu")
    stoi = tok_state["stoi"]
    tokenizer = CharTokenizerFromState(stoi)
    vocab_size = len(stoi)

    print(f"[INFO] Cargando checkpoint desde: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Intentamos el formato "bonito" (config + model_state_dict)
    if isinstance(ckpt, dict) and "config" in ckpt and "model_state_dict" in ckpt:
        print("[INFO] Checkpoint con 'config' y 'model_state_dict' detectado.")
        config_dict = ckpt["config"]
        model_state = ckpt["model_state_dict"]
        config = GPTConfig(**config_dict)
    else:
        # Caso como el tuyo: checkpoint es solo state_dict
        print("[WARN] Checkpoint sin 'config'; asumimos que es un state_dict puro.")
        model_state = ckpt

        # ⚠️ IMPORTANTE: pon aquí los MISMOS hiperparámetros que usaste en el pretraining
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=256,   # <-- usa aquí tu --seq-len del pretraining
            d_model=256,      # <-- tu --d-model
            n_heads=4,        # <-- tu --n-heads
            n_layers=4,       # <-- tu --n-layers
            dropout=0.1,      # <-- tu --dropout
        )

    base_model = GPTModel(config)
    base_model.load_state_dict(model_state)
    print("[INFO] Modelo base GPT cargado.")

    # seq_len = contexto del modelo (block_size en config)
    seq_len = config.block_size
    print(f"[INFO] Usando seq_len = {seq_len} para clasificación.")

    # -------------------------
    # 3. Construir dataset de clasificación (toy)
    # -------------------------
    all_examples = build_toy_examples()

    # Split manual: 75% train, 25% val
    n_total = len(all_examples)
    n_train = int(n_total * 0.75)
    train_examples = all_examples[:n_train]
    val_examples = all_examples[n_train:]

    train_dataset = ClassificationDataset(train_examples, tokenizer, seq_len=seq_len)
    val_dataset = ClassificationDataset(val_examples, tokenizer, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"[INFO] Tamaño train: {len(train_dataset)} ejemplos")
    print(f"[INFO] Tamaño val:   {len(val_dataset)} ejemplos")

    # -------------------------
    # 4. Crear GPTForClassification y optimizador
    # -------------------------
    model = GPTForClassification(base_model=base_model, num_classes=args.num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -------------------------
    # 5. Loop de entrenamiento
    # -------------------------
    for epoch in range(1, args.max_epochs + 1):
        print(f"\n===== Época {epoch}/{args.max_epochs} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    print("\n[INFO] Finetuning de clasificación completado.")


if __name__ == "__main__":
    main()