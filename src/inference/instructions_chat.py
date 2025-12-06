# src/inference/instructions_chat.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from torch import nn

from src.model.gpt import GPTConfig, GPTModel
from src.cli.finetune_classification import CharTokenizerFromState


# -------------------------------------------------------------------
# Dataclass para agrupar todo lo necesario del modelo
# -------------------------------------------------------------------
@dataclass
class InstructionsModelBundle:
    model: nn.Module
    tokenizer: Any
    config: GPTConfig
    device: torch.device


# -------------------------------------------------------------------
# Helpers de dispositivo
# -------------------------------------------------------------------
def get_device(device_str: str) -> torch.device:
    """
    Convierte un string ('cpu', 'mps', 'cuda') en torch.device.
    Siempre espera un string, nunca un torch.device.
    """
    if device_str is None:
        return torch.device("cpu")

    s = str(device_str).lower()
    if s.startswith("cuda") and torch.cuda.is_available():
        return torch.device(s)
    if s == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------------------------------------------
# Carga del modelo de instrucciones
# -------------------------------------------------------------------
def load_instructions_model(
    ckpt_dir: str,
    device_str: str = "cpu",
) -> InstructionsModelBundle:
    """
    Carga el checkpoint de instrucciones gpt_char_instructions.pt
    ubicado dentro de ckpt_dir.
    """
    ckpt_path = os.path.join(ckpt_dir, "gpt_char_instructions.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint de instrucciones en: {ckpt_path}"
        )

    print(f"[INFO] Cargando modelo de instrucciones desde: {ckpt_path}")

    obj: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")

    config_dict = obj["config"]
    state_dict = obj["model_state_dict"]
    stoi = obj["stoi"]

    # Reconstruir config y tokenizer
    config = GPTConfig(**config_dict)
    tokenizer = CharTokenizerFromState(stoi)

    # Instanciar modelo y cargar pesos
    model = GPTModel(config)
    model.load_state_dict(state_dict)

    device = get_device(device_str)
    model.to(device)
    model.eval()

    return InstructionsModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )


# -------------------------------------------------------------------
# Helpers de prompt / codificación
# -------------------------------------------------------------------
def build_prompt(user_prompt: str) -> str:
    """
    Construye el texto de entrada estilo instruction-tuning:
        "<instr> {user_prompt}\n<resp> "
    """
    return f"<instr> {user_prompt}\n<resp> "


def encode_prompt(
    text: str,
    tokenizer: Any,
    seq_len: int,
) -> torch.Tensor:
    """
    Tokeniza y ajusta a longitud fija seq_len (truncado + padding con 0).
    Devuelve tensor (1, T).
    """
    ids: List[int] = tokenizer.encode(text)
    ids = ids[:seq_len]
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def decode_ids(
    ids: List[int],
    tokenizer: Any,
) -> str:
    """
    Decodifica una lista de IDs a texto usando tokenizer.stoi.
    No usamos tokenizer.decode porque no existe en CharTokenizerFromState.
    """
    # Construimos itos a partir de stoi
    itos = {i: ch for ch, i in tokenizer.stoi.items()}

    chars: List[str] = []
    for idx in ids:
        ch = itos.get(int(idx), "")
        # opcional: ignorar padding si usamos "<pad>"
        if ch == "<pad>":
            continue
        chars.append(ch)

    return "".join(chars)


# -------------------------------------------------------------------
# Generación de respuesta
# -------------------------------------------------------------------
def generate_answer(
    bundle: InstructionsModelBundle,
    user_prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
) -> str:
    """
    Genera una respuesta de texto dado un prompt de usuario.

    Este es el método que llama Streamlit.
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    config = bundle.config
    device = bundle.device

    # 1) Construir prompt completo
    prompt_text = build_prompt(user_prompt)

    # 2) Codificar
    inp = encode_prompt(prompt_text, tokenizer, config.max_seq_len).to(device)

    # 3) Generar
    with torch.no_grad():
        out_ids = model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
        )

    decoded = decode_ids(out_ids[0].tolist(), tokenizer)

    # 4) Intentar separar lo que viene después de <resp>
    split_tok = "<resp>"
    if split_tok in decoded:
        after_resp = decoded.split(split_tok, 1)[1]
        answer = after_resp.strip()
    else:
        # fallback: devolvemos todo el texto generado
        answer = decoded

    return answer


# -------------------------------------------------------------------
# Pequeño test manual desde la terminal
# -------------------------------------------------------------------
if __name__ == "__main__":
    ckpt_dir = "models/checkpoints_oscar_long"
    bundle = load_instructions_model(ckpt_dir, device_str="cpu")

    questions = [
        "Un perro es un canino?",
        "Cuál es la capital de Costa Rica?",
        "Qué es un modelo de lenguaje?",
    ]

    for q in questions:
        ans = generate_answer(bundle, user_prompt=q, max_new_tokens=80, temperature=0.7)
        print("\n====================================================")
        print("Pregunta:", q)
        print("Respuesta del modelo:")
        print(repr(ans))