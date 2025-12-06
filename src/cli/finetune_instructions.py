# src/cli/finetune_instructions.py

import argparse
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from src.model.gpt import GPTConfig, GPTModel
from src.data.datasets import InstructionExample, InstructionDataset
from src.finetuning.instructions import compute_instruction_loss


# ---------------------------------------------------------------------
# Utilidades de dispositivo
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Tokenizer mínimo desde char_tokenizer.pt
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Carga del dataset de instrucciones
# ---------------------------------------------------------------------
def load_instruction_examples(path: str) -> List[InstructionExample]:
    """
    Carga ejemplos de un archivo .jsonl o .json con pares
    {"prompt": "...", "response": "..."}.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo de instrucciones en: {path}")

    examples: List[InstructionExample] = []

    if path.endswith(".jsonl"):
        # Un JSON por línea
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"]
                response = obj["response"]
                examples.append(InstructionExample(prompt=prompt, response=response))
    else:
        # Asumimos un JSON con lista de objetos
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for obj in data:
                prompt = obj["prompt"]
                response = obj["response"]
                examples.append(InstructionExample(prompt=prompt, response=response))

    return examples


# ---------------------------------------------------------------------
# Entrenamiento / evaluación
# ---------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, device, pad_token_id: int):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = compute_instruction_loss(
            model=model,
            input_ids=input_ids,
            labels=labels,
            pad_token_id=pad_token_id,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        pct = 100.0 * batch_idx / len(dataloader)
        print(
            f"[train] batch {batch_idx}/{len(dataloader)} "
            f"({pct:5.1f}%) - loss: {loss.item():.4f}"
        )

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device, pad_token_id: int):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        loss, _ = compute_instruction_loss(
            model=model,
            input_ids=input_ids,
            labels=labels,
            pad_token_id=pad_token_id,
        )

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Finetuning de GPT de caracteres para instrucciones (instruction tuning)."
    )

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="models/checkpoints_oscar_long",
        help="Directorio con gpt_char_best.pt y char_tokenizer.pt",
    )
    parser.add_argument(
        "--instructions-path",
        type=str,
        default="data/processed/instructions/instructions_tiny.jsonl",
        help="Ruta al archivo .jsonl/.json con pares {prompt, response}.",
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
        default=20,
        help="Número máximo de épocas.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--pad-id",
        type=int,
        default=0,
        help="ID de padding a ignorar en la loss.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Si se especifica, congela todos los pesos del modelo (no recomendado aquí).",
    )

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[INFO] Usando dispositivo: {device}")

    # ------------------------------------------------------------
    # 1. Cargar tokenizer
    # ------------------------------------------------------------
    tok_path = os.path.join(args.ckpt_dir, "char_tokenizer.pt")
    if not os.path.isfile(tok_path):
        raise FileNotFoundError(f"No se encontró el tokenizer en {tok_path}")

    print(f"[INFO] Cargando tokenizer desde: {tok_path}")
    tok_state = torch.load(tok_path, map_location="cpu")
    stoi = tok_state["stoi"]
    tokenizer = CharTokenizerFromState(stoi)

    # ------------------------------------------------------------
    # 2. Cargar checkpoint base (gpt_char_best.pt) y reconstruir config
    # ------------------------------------------------------------
    ckpt_path = os.path.join(args.ckpt_dir, "gpt_char_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No se encontró el checkpoint en {ckpt_path}")

    print(f"[INFO] Cargando checkpoint base desde: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    vocab_size = len(stoi)

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
    print(f"[INFO] Usando seq_len = {seq_len} para instruction tuning.")
    print("[INFO] Config reconstruida; se usará como backbone preentrenado.")

    # ------------------------------------------------------------
    # 3. Cargar ejemplos de instrucciones y construir datasets
    # ------------------------------------------------------------
    examples = load_instruction_examples(args.instructions_path)
    print(f"[INFO] Ejemplos de instrucciones cargados: {len(examples)}")

    if len(examples) < 2:
        raise ValueError("Se requieren al menos 2 ejemplos de instrucciones.")

    n_total = len(examples)
    n_train = max(1, int(n_total * 0.75))
    train_examples = examples[:n_train]
    val_examples = examples[n_train:]

    train_dataset = InstructionDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )
    val_dataset = InstructionDataset(
        examples=val_examples,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )

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

    # ------------------------------------------------------------
    # 4. Instanciar GPTModel y cargar pesos de pretraining
    # ------------------------------------------------------------
    print("[INFO] Instanciando backbone GPTModel y cargando pesos...")
    model = GPTModel(config)
    model.load_state_dict(model_state)
    model.to(device)
    print("[INFO] Backbone GPTModel cargado correctamente con pesos preentrenados.")

    if args.freeze_backbone:
        print("[INFO] Congelando todos los pesos del modelo (no se entrenarán).")
        for p in model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------
    # 5. Optimizador
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # ------------------------------------------------------------
    # 6. Loop de entrenamiento
    # ------------------------------------------------------------
    best_val_loss = float("inf")
    save_path = os.path.join(args.ckpt_dir, "gpt_char_instructions.pt")

    for epoch in range(1, args.max_epochs + 1):
        print(f"\n===== Época {epoch}/{args.max_epochs} =====")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, pad_token_id=args.pad_id
        )
        val_loss = evaluate(
            model, val_loader, device, pad_token_id=args.pad_id
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        # Guardar el mejor modelo según val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "config": config.__dict__,
                    "model_state_dict": model.state_dict(),
                    "stoi": stoi,
                },
                save_path,
            )
            print(
                f"[INFO] Nuevo mejor modelo guardado en: {save_path} "
                f"(val_loss={val_loss:.4f})"
            )

    print("\n[INFO] Finetuning de instrucciones completado.")
    print(f"[INFO] Mejor val_loss: {best_val_loss:.4f}")
    print(f"[INFO] Modelo final guardado en: {save_path}")


if __name__ == "__main__":
    main()