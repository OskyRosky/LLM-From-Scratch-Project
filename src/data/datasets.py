
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from torch.utils.data import Dataset
import torch


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