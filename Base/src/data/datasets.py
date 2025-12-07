# src/data/datasets.py

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------
# 1. CharacterDataset – pretraining (LM de caracteres)
# --------------------------------------------------------------------


class CharacterDataset(Dataset):
    """
    Dataset de nivel carácter para pretraining (next-token prediction).

    Dado un vector de IDs [t0, t1, ..., tN], construye ejemplos:

        x = [ti,     ti+1,   ..., ti+seq_len-1]
        y = [ti+1,   ti+2,   ..., ti+seq_len]

    para todos los offsets válidos i.
    """

    def __init__(self, token_ids: Sequence[int], seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        if len(token_ids) <= seq_len:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for seq_len={seq_len}. "
                "You need a longer corpus or a smaller seq_len."
            )

        self.token_ids: List[int] = list(token_ids)
        self.seq_len: int = seq_len

    def __len__(self) -> int:
        """
        Número de ejemplos de entrenamiento disponibles.

        Para una secuencia de longitud N y ventana L, hay N - L ejemplos.
        """
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve (x, y) de shape (seq_len,).

        x: token_ids[idx : idx + seq_len]
        y: token_ids[idx + 1 : idx + 1 + seq_len]
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range 0..{len(self) - 1}")

        x_ids = self.token_ids[idx : idx + self.seq_len]
        y_ids = self.token_ids[idx + 1 : idx + 1 + self.seq_len]

        x = torch.tensor(x_ids, dtype=torch.long)
        y = torch.tensor(y_ids, dtype=torch.long)

        return x, y


# --------------------------------------------------------------------
# 2. ClassificationDataset – finetuning de clasificación
# --------------------------------------------------------------------


@dataclass
class ClassificationExample:
    """
    Ejemplo de clasificación:
    - text: texto de entrada.
    - label: entero que representa la clase (0, 1, 2, ...).
    """
    text: str
    label: int


class ClassificationDataset(Dataset):
    """
    Dataset para clasificación de texto usando un tokenizer de caracteres.

    Requisitos del tokenizer:
      - método encode(text: str) -> List[int]
      - diccionario .stoi con los IDs de los tokens
        (idealmente con "<PAD>" o "<pad>" para padding).
    """

    def __init__(self, examples, tokenizer, seq_len: int):
        """
        :param examples: lista de ClassificationExample
        :param tokenizer: tokenizer de caracteres ya cargado
        :param seq_len: longitud fija de secuencia
        """
        self.examples: List[ClassificationExample] = list(examples)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Intentar encontrar un pad_id razonable.
        # Primero "<PAD>", luego "<pad>", y si no existe, 0.
        stoi: Dict[str, int] = getattr(self.tokenizer, "stoi", {})
        self.pad_id: int = stoi.get("<PAD>", stoi.get("<pad>", 0))

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Codifica un texto a IDs, truncando o rellenando con PAD
        hasta seq_len.
        """
        ids = self.tokenizer.encode(text)

        # Truncar si excede seq_len
        if len(ids) > self.seq_len:
            ids = ids[: self.seq_len]
        # Rellenar con pad_id si es más corto
        elif len(ids) < self.seq_len:
            pad_length = self.seq_len - len(ids)
            ids = ids + [self.pad_id] * pad_length

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        input_ids = self._encode_text(ex.text)
        label = torch.tensor(ex.label, dtype=torch.long)

        return {
            "input_ids": input_ids,  # (seq_len,)
            "label": label,          # entero con la clase
        }


# --------------------------------------------------------------------
# 3. InstructionDataset – instruction tuning (Fase 6 / 8)
# --------------------------------------------------------------------


@dataclass
class InstructionExample:
    """
    Ejemplo de instrucción para fine-tuning:
    - prompt: instrucción / pregunta.
    - response: respuesta objetivo.
    """
    prompt: str
    response: str


class InstructionDataset(Dataset):
    """
    Dataset para instruction tuning.

    Construye secuencias de texto de la forma:

        "<instr> {prompt}\n<resp> {response}"

    Luego las tokeniza y aplica truncado/padding a seq_len.
    """

    def __init__(
        self,
        examples: List[InstructionExample],
        tokenizer,
        seq_len: int,
        instr_prefix: str = "<instr>",
        resp_prefix: str = "<resp>",
    ):
        """
        Parameters
        ----------
        examples:
            Lista de InstructionExample.
        tokenizer:
            Objeto con:
              - encode(text: str) -> List[int]
              - atributo .stoi (dict) con vocabulario.
        seq_len:
            Longitud máxima de secuencia.
        instr_prefix:
            Prefijo para la parte de instrucción.
        resp_prefix:
            Prefijo para la parte de respuesta.
        """
        self.examples: List[InstructionExample] = list(examples)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.instr_prefix = instr_prefix
        self.resp_prefix = resp_prefix

        # Detectar pad_id coherente con el vocabulario del tokenizer
        stoi: Dict[str, int] = getattr(self.tokenizer, "stoi", {})
        pad_id = stoi.get("<PAD>", stoi.get("<pad>", 0))
        self.pad_id: int = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Texto conjunto instrucción + respuesta
        full_text = f"{self.instr_prefix} {ex.prompt}\n{self.resp_prefix} {ex.response}"

        # Tokenización -> lista de IDs
        ids = self.tokenizer.encode(full_text)

        # Truncado a seq_len
        ids = ids[: self.seq_len]

        # Padding con pad_id a la derecha
        if len(ids) < self.seq_len:
            ids = ids + [self.pad_id] * (self.seq_len - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)

        # Para LM: labels = input_ids (el shift se hace en la loss/entrenamiento)
        sample = {
            "input_ids": input_ids,          # (T,)
            "labels": input_ids.clone(),     # (T,)
        }
        return sample