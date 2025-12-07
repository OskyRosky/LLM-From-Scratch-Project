# src/cli/finetune_classification.py

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

        # Progreso simple por batch
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

    with torch.no_grad():
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
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Si se establece, congela el backbone GPT y entrena solo la cabeza de clasificación.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="Tipo de pooling sobre las hidden states (mean|cls).",
    )

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[INFO] Usando dispositivo: {device}")

    # -------------------------
    # 1. Cargar checkpoint y tokenizer
    # -------------------------
    ckpt_path = os.path.join(args.ckpt_dir, "gpt_char_best.pt")
    tok_path = os.path.join(args.ckpt_dir, "char_tokenizer.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No se encontró el checkpoint en {ckpt_path}")
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(f"No se encontró el tokenizer en {tok_path}")

    print(f"[INFO] Cargando tokenizer desde: {tok_path}")
    tok_state = torch.load(tok_path, map_location="cpu")
    stoi = tok_state["stoi"]
    tokenizer = CharTokenizerFromState(stoi)
    vocab_size = len(stoi)

    print(f"[INFO] Cargando checkpoint desde: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # -------------------------
    # 2. Reconstruir config + extraer model_state
    # -------------------------
    if isinstance(ckpt, dict) and "config" in ckpt and "model_state_dict" in ckpt:
        print("[INFO] Checkpoint con 'config' y 'model_state_dict' detectado.")
        config_dict = ckpt["config"]
        model_state = ckpt["model_state_dict"]
        config = GPTConfig(**config_dict)

    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print("[INFO] Checkpoint de entrenamiento detectado (sin 'config').")
        model_state = ckpt["model_state_dict"]

        training_cfg = ckpt.get("training_config", None)
        tc = {}
        if training_cfg is not None:
            if hasattr(training_cfg, "__dict__"):
                tc = vars(training_cfg)
            elif isinstance(training_cfg, dict):
                tc = training_cfg

        d_model = tc.get("d_model", 256)
        n_heads = tc.get("n_heads", 4)
        n_layers = tc.get("n_layers", 4)
        dropout = tc.get("dropout", 0.1)
        max_seq_len = tc.get("seq_len") or tc.get("max_seq_len") or 256

        we = model_state.get("tok_embedding.embedding.weight", None)
        if we is not None:
            vocab_size = we.shape[0]
            d_model = we.shape[1]

        pe = model_state.get("pos_embedding.pos_embedding.weight", None)
        if pe is not None:
            max_seq_len = pe.shape[0]

        layer_ids = {
            int(k.split(".")[1])
            for k in model_state.keys()
            if k.startswith("blocks.")
        }
        if layer_ids:
            n_layers = max(layer_ids) + 1

        print(
            f"[INFO] Inferido desde checkpoint -> "
            f"vocab_size={vocab_size}, max_seq_len={max_seq_len}, "
            f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}"
        )

        config = GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    else:
        print("[INFO] Checkpoint es un state_dict puro del modelo.")
        model_state = ckpt

        we = model_state["tok_embedding.embedding.weight"]
        pe = model_state["pos_embedding.pos_embedding.weight"]

        vocab_size = we.shape[0]
        d_model = we.shape[1]
        max_seq_len = pe.shape[0]

        layer_ids = {
            int(k.split(".")[1])
            for k in model_state.keys()
            if k.startswith("blocks.")
        }
        n_layers = max(layer_ids) + 1 if layer_ids else 0

        n_heads = 4 if d_model % 4 == 0 else 1

        print(
            f"[INFO] Inferido desde state_dict -> "
            f"vocab_size={vocab_size}, max_seq_len={max_seq_len}, "
            f"d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}"
        )

        config = GPTConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=0.1,
        )

    seq_len = config.max_seq_len
    print(f"[INFO] Usando seq_len = {seq_len} para clasificación.")
    print("[INFO] Config reconstruida; se usará como backbone preentrenado.")

    # -------------------------
    # 3. Construir dataset de clasificación (toy)
    # -------------------------
    all_examples = build_toy_examples()

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
    # 4. Crear backbone GPT preentrenado y modelo de clasificación
    # -------------------------
    print("[INFO] Instanciando backbone GPTModel y cargando pesos de pretraining...")
    backbone = GPTModel(config)
    backbone.load_state_dict(model_state)
    print("[INFO] Backbone GPTModel cargado correctamente con pesos del pretraining.")

    print("[INFO] Instanciando GPTForClassification y copiando el backbone...")
    model = GPTForClassification(config, num_classes=args.num_classes, pooling=args.pooling)
    model.gpt.load_state_dict(backbone.state_dict())
    print("[INFO] Pesos del backbone copiados a model.gpt.")

    if args.freeze_backbone:
        for p in model.gpt.parameters():
            p.requires_grad = False
        print("[INFO] Backbone congelado: solo se entrenará la cabeza de clasificación.")

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

    # -------------------------
    # 6. Guardar modelo finetuneado
    # -------------------------
    save_path = os.path.join(args.ckpt_dir, "gpt_char_cls_toy_v2.pt")
    torch.save(
        {
            "config": config.__dict__,
            "model_state_dict": model.state_dict(),
            "num_classes": args.num_classes,
            "stoi": stoi,
        },
        save_path,
    )
    print(f"[INFO] Modelo de clasificación guardado en: {save_path}")

    print("\n[INFO] Finetuning de clasificación completado.")


if __name__ == "__main__":
    main()