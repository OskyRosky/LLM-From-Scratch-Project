# src/data/datasets.py

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------
# 1. CharacterDataset – pretraining (LM de caracteres)
# --------------------------------------------------------------------


class CharacterDataset(Dataset):
    """
    A simple character-level dataset for next-token prediction.

    Given a sequence of token ids [t0, t1, t2, ..., tN],
    we create training examples of the form:

        x = [ti,     ti+1,   ..., ti+seq_len-1]
        y = [ti+1,   ti+2,   ..., ti+seq_len]

    for all valid i.

    This is exactly what a GPT-style language model needs:
    given a prefix (x), predict the next token at each position (y).
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
        Number of training examples.

        For a sequence of length N and window size L (seq_len),
        we can start the window at positions 0 to N - L - 1 (inclusive),
        so the total number of samples is N - L.
        """
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a pair (x, y) of shape (seq_len,).

        x: token_ids[idx : idx + seq_len]
        y: token_ids[idx + 1 : idx + 1 + seq_len]
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range 0..{len(self)-1}")

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
    Estructura simple para guardar un ejemplo de clasificación:
    - text: string de entrada (ya limpio o crudo, según definas)
    - label: entero (0, 1, 2, ...) que representa la clase
    """
    text: str
    label: int


class ClassificationDataset(Dataset):
    """
    Dataset para clasificación de texto usando el tokenizer de caracteres.

    Supone que el tokenizer tiene:
      - un método `encode(text: str) -> List[int]`
      - un diccionario `stoi` con los IDs de los tokens
        (idealmente con "<pad>", pero si no existe usamos 0).
    """

    def __init__(self, examples, tokenizer, seq_len: int):
        """
        :param examples: lista de `ClassificationExample`
        :param tokenizer: tokenizer de caracteres ya cargado
        :param seq_len: longitud fija de secuencia (como en pretraining)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # ID de padding: si no hay "<pad>", usamos 0 por defecto
        self.pad_id = self.tokenizer.stoi.get("<pad>", 0)

    def __len__(self):
        return len(self.examples)

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Codifica un texto a IDs, trunca o paddea a seq_len fijo.
        """
        ids = self.tokenizer.encode(text)

        # Truncar si es más largo que seq_len
        if len(ids) > self.seq_len:
            ids = ids[: self.seq_len]
        # Pad a la derecha si es más corto
        elif len(ids) < self.seq_len:
            pad_length = self.seq_len - len(ids)
            ids = ids + [self.pad_id] * pad_length

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        input_ids = self._encode_text(ex.text)
        label = torch.tensor(ex.label, dtype=torch.long)

        return {
            "input_ids": input_ids,  # shape: (seq_len,)
            "label": label,          # entero con la clase
        }


# --------------------------------------------------------------------
# 3. InstructionDataset – instruction tuning (Fase 6)
# --------------------------------------------------------------------


@dataclass
class InstructionExample:
    """
    Ejemplo de instrucción para fine-tuning:
    - prompt: la instrucción / pregunta
    - response: la respuesta que queremos que el modelo aprenda a generar
    """
    prompt: str
    response: str


class InstructionDataset(Dataset):
    """
    Dataset para instruction tuning.

    Construye secuencias de la forma (en texto):

        "<instr> {prompt}\n<resp> {response}"

    y luego las tokeniza con un tokenizer que tenga un método .encode(text)
    que devuelve una lista de IDs de tokens/caracteres.

    Por simplicidad, aquí:
    - Usamos tokens "pseudo-especiales" como texto literal ("<instr>", "<resp>").
    - Hacemos truncado/padding a un tamaño fijo seq_len.
    - Dejamos que el script de entrenamiento se encargue de hacer el shift
      input_ids[:-1] -> labels[1:].
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
            Objeto con un método .encode(text: str) -> List[int].
            En nuestro caso será el tokenizer de caracteres cargado desde
            char_tokenizer.pt.
        seq_len:
            Longitud máxima de secuencia (se trunca o se rellena).
        instr_prefix, resp_prefix:
            Prefijos textuales para marcar instrucción y respuesta.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.instr_prefix = instr_prefix
        self.resp_prefix = resp_prefix

        # Consistencia con ClassificationDataset: si hubiera "<pad>", lo usamos.
        self.pad_id = getattr(self.tokenizer, "stoi", {}).get("<pad>", 0)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]

        # Construimos el texto completo para este ejemplo
        full_text = f"{self.instr_prefix} {ex.prompt}\n{self.resp_prefix} {ex.response}"

        # Tokenización -> lista de IDs
        ids = self.tokenizer.encode(full_text)

        # Truncado
        ids = ids[: self.seq_len]

        # Padding
        if len(ids) < self.seq_len:
            ids = ids + [self.pad_id] * (self.seq_len - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)

        # En instruction tuning estilo LM:
        #   - input_ids se usa como entrada
        #   - labels será la misma secuencia desplazada, pero
        #     haremos ese shift en el loop de entrenamiento.
        sample = {
            "input_ids": input_ids,        # (T,)
            "labels": input_ids.clone(),   # (T,)
        }
        return sample