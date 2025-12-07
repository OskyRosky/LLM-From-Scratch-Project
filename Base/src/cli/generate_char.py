# src/cli/generate_char.py

from __future__ import annotations
import argparse
import torch
from pathlib import Path

from src.model.gpt import GPTModel, GPTConfig
from src.utils.logging_utils import get_logger

# ---------------------------------------------------------
#  Utilidades para cargar tokenizador
# ---------------------------------------------------------
def load_tokenizer(path: str):
    payload = torch.load(path, map_location="cpu")
    stoi = payload["stoi"]
    itos = payload["itos"]
    return stoi, itos


def encode(prompt: str, stoi: dict[str, int]) -> list[int]:
    ids = []
    for ch in prompt:
        if ch in stoi:
            ids.append(stoi[ch])
        # Si el caracter no está en el vocabulario, lo ignoramos.
        # También podríamos loguearlo, pero para empezar lo dejamos simple.
    if not ids:
        raise ValueError(
            "The prompt produced an empty token sequence: none of its characters are in the tokenizer vocabulary."
        )
    return ids


def decode(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------
#  Generación carácter por carácter
# ---------------------------------------------------------
@torch.no_grad()
def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    stoi: dict,
    itos: dict,
    device: torch.device,
):
    model.eval()
    for _ in range(max_new_tokens):
        # Cortar a la longitud máxima del modelo
        idx_cond = idx[:, -model.config.max_seq_len :]
        logits, _ = model(idx_cond)

        # Tomar solo el último paso
        logits = logits[:, -1, :]  # (B, vocab)
        probs = torch.softmax(logits, dim=-1)

        # Sampling
        next_id = torch.multinomial(probs, num_samples=1)

        # Append
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist(), itos)


# ---------------------------------------------------------
#  CLI principal
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generación de texto con GPT char-level.")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Directorio con checkpoint y tokenizer.")
    parser.add_argument("--prompt", type=str, default="Hola mundo", help="Texto inicial.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Tokens nuevos a generar.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Dispositivo para inferencia.",
    )
    args = parser.parse_args()

    logger = get_logger("generate_char")

    # Directorios
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = ckpt_dir / "gpt_char_best.pt"
    tok_path = ckpt_dir / "char_tokenizer.pt"

    logger.info(f"Loading checkpoint from {ckpt_path}")
    logger.info(f"Loading tokenizer from {tok_path}")

    # Load tokenizer
    stoi, itos = load_tokenizer(tok_path)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["training_config"]

    # Reconstruir config del modelo
    model_cfg = GPTConfig(
        vocab_size=len(stoi),
        max_seq_len=cfg_dict["batch_size"] * 8,  # safe fallback
        d_model=cfg_dict.get("d_model", 128),
        n_heads=cfg_dict.get("n_heads", 4),
        n_layers=cfg_dict.get("n_layers", 4),
        dropout=0.0,
    )

    model = GPTModel(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device(args.device)
    model.to(device)

    # Encode prompt
    prompt_ids = encode(args.prompt, stoi)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    logger.info("Generating text...")
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        stoi=stoi,
        itos=itos,
        device=device,
    )

    print("\n===================\nGENERATED TEXT\n===================\n")
    print(out)
    print("\n===================\n")


if __name__ == "__main__":
    main()