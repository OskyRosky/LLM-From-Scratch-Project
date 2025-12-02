from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


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