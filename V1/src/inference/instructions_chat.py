# src/inference/instructions_chat.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import os

import torch

from src.model.gpt import GPTModel, GPTConfig


# ---------------------------------------------------------------------
# Dataclass para agrupar modelo + tokenizer + dispositivo
# ---------------------------------------------------------------------


@dataclass
class InstructionsModelBundle:
    model: GPTModel
    tokenizer: "CharTokenizerForInstructions"
    device: torch.device
    pad_id: int
    config: GPTConfig


# ---------------------------------------------------------------------
# Tokenizer para inference de instrucciones
# ---------------------------------------------------------------------


class CharTokenizerForInstructions:
    """
    Tokenizer muy simple basado en stoi/itos del pretraining,
    reutilizado para instruction tuning.

    - encode(text): mapea cada carácter a su ID.
    - decode(ids): mapea IDs a caracteres.
    """

    def __init__(self, stoi: Dict[str, int]):
        self.stoi: Dict[str, int] = stoi
        self.itos: Dict[int, str] = {v: k for k, v in stoi.items()}
        # ID de padding (si no existe, 0)
        self.pad_id: int = self.stoi.get("<PAD>", self.stoi.get("<pad>", 0))

    def encode(self, text: str) -> List[int]:
        """
        Codifica carácter por carácter usando stoi.
        Si un carácter no existe en el vocabulario, usa <unk> o el primer ID.
        """
        default_id = self.stoi.get("<unk>", next(iter(self.stoi.values())))
        return [self.stoi.get(ch, default_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """
        Decodifica IDs a texto, concatenando los símbolos tal cual
        están definidos en itos (incluyendo tokens especiales).
        """
        return "".join(self.itos.get(i, "?") for i in ids)


# ---------------------------------------------------------------------
# Selección de dispositivo
# ---------------------------------------------------------------------


def _select_device(device_str: str) -> torch.device:
    device_str = (device_str or "auto").lower()
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
# Carga del modelo de instrucciones desde el checkpoint
#   ✅ acepta ckpt_dir o ckpt_path
# ---------------------------------------------------------------------


def load_instructions_model(
    ckpt_dir: Optional[str] = None,
    device_str: str = "auto",
    ckpt_path: Optional[str] = None,
) -> InstructionsModelBundle:
    """
    Carga el modelo fine-tuneado para instrucciones.

    Formas de uso:
      - load_instructions_model(ckpt_dir="models/checkpoints_oscar_long", device_str="mps")
      - load_instructions_model(ckpt_path="models/checkpoints_oscar_long/gpt_char_instructions.pt")
    """
    device = _select_device(device_str)

    # Resolver ruta al checkpoint
    if ckpt_path is None:
        if ckpt_dir is None:
            raise ValueError(
                "Debes pasar al menos uno de ckpt_dir o ckpt_path a load_instructions_model()."
            )
        ckpt_path = os.path.join(ckpt_dir, "gpt_char_instructions.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No se encontró el checkpoint de instrucciones en: {ckpt_path}")

    print(f"[INFO] Cargando modelo de instrucciones desde: {ckpt_path}")
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")

    if (
        not isinstance(ckpt, dict)
        or "config" not in ckpt
        or "model_state_dict" not in ckpt
        or "stoi" not in ckpt
    ):
        raise ValueError(
            "Checkpoint de instrucciones no tiene 'config', 'model_state_dict' y 'stoi'. "
            "Asegúrate de haberlo guardado con finetune_instructions.py."
        )

    # Reconstruir config y modelo
    config_dict = ckpt["config"]
    config = GPTConfig(**config_dict)

    model = GPTModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    stoi = ckpt["stoi"]
    tokenizer = CharTokenizerForInstructions(stoi)
    pad_id = tokenizer.pad_id

    print(f"[INFO] Dispositivo: {device.type}")
    print(f"[INFO] pad_id (inference): {pad_id}")

    return InstructionsModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_id=pad_id,
        config=config,
    )


# ---------------------------------------------------------------------
# Generación de respuestas (inference)
# ---------------------------------------------------------------------


@torch.no_grad()
def generate_answer(
    bundle: InstructionsModelBundle,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> Tuple[str, str]:
    """
    Genera una respuesta dado un 'prompt' de usuario.

    Construye un contexto de la forma:
        "<instr> {prompt}\\n<resp>"

    Devuelve:
      - answer_text: solo el texto posterior a <resp>, sin <PAD>/<pad>.
      - full_text: todo el texto generado (incluyendo prefijos).
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    pad_id = bundle.pad_id
    max_ctx = bundle.config.max_seq_len

    # 1) Construir prompt con los prefijos usados en el dataset
    prompt_text = f"<instr> {prompt}\n<resp>"
    input_ids = tokenizer.encode(prompt_text)

    # 🔒 Asegurar que el contexto inicial no exceda max_seq_len
    if len(input_ids) > max_ctx:
        input_ids = input_ids[-max_ctx:]

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # (1, T_in)

    generated = input_tensor

    # 2) Bucle de generación con ventana deslizante
    for _ in range(max_new_tokens):
        # Mantener solo la última ventana de contexto
        if generated.size(1) > max_ctx:
            generated = generated[:, -max_ctx:]

        logits = model(generated)           # (1, T, V)
        logits_last = logits[:, -1, :]      # (1, V)

        # Prohibir que el modelo escoja el token PAD
        if pad_id is not None:
            logits_last[:, pad_id] = -1e9

        # Muestreo greedy / con temperatura
        if temperature is None or temperature <= 0.0:
            next_token = torch.argmax(logits_last, dim=-1, keepdim=True)  # (1, 1)
        else:
            probs = torch.softmax(logits_last / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)          # (1, 1)

        generated = torch.cat([generated, next_token], dim=1)

    # 3) Decodificar todo
    full_ids = generated[0].tolist()
    full_text = tokenizer.decode(full_ids)

    # Nos quedamos solo con la parte después de "<resp>"
    if "<resp>" in full_text:
        after_resp = full_text.split("<resp>", 1)[1]
    else:
        after_resp = full_text

    # Limpiar tokens de padding del texto legible
    answer_text = (
        after_resp.replace("<PAD>", "").replace("<pad>", "").strip()
    )

    return answer_text, full_text


# ---------------------------------------------------------------------
# Pequeño test manual
# ---------------------------------------------------------------------


if __name__ == "__main__":
    bundle = load_instructions_model(
        ckpt_dir="models/checkpoints_oscar_long",
        device_str="cpu",
    )
    qs = [
        "Un perro es un canino?",
        "Un gato es un felino?",
        "Cuál es la capital de Costa Rica?",
    ]
    for q in qs:
        ans, full = generate_answer(
            bundle,
            prompt=q,
            max_new_tokens=80,
            temperature=0.0,
        )
        print("\n=======================================")
        print("Pregunta:", q)
        print("Full:", repr(full))
        print("Respuesta:", ans)