from typing import Dict, List, Optional


class CharacterTokenizer:
    """
    A simple character-level tokenizer.

    This is our first tokenizer implementation, mainly for:
    - understanding the data flow (text -> ids -> text)
    - testing the model infrastructure on small examples

    Later, we can add more advanced tokenizers (e.g. BPE) while
    keeping the same interface (train / encode / decode).
    """

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        # We reserve a few tokens for control purposes
        if special_tokens is None:
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        self.special_tokens: List[str] = special_tokens
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        # Add special tokens first
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> None:
        """
        Internal helper to add a single token to the vocabulary.
        """
        if token in self.stoi:
            return
        idx = len(self.stoi)
        self.stoi[token] = idx
        self.itos[idx] = token

    def train(self, text: str) -> None:
        """
        Build the character vocabulary from a given text corpus.

        Every unique character in the text (including spaces and punctuation)
        becomes a token in the vocabulary, in addition to the special tokens.
        """
        unique_chars = sorted(set(text))
        for ch in unique_chars:
            # It's very unlikely but we avoid collisions with special tokens
            if ch in self.stoi:
                continue
            self._add_token(ch)

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the current vocabulary.
        """
        return len(self.stoi)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Convert a string into a list of token ids.

        Parameters
        ----------
        text:
            The input string.
        add_special_tokens:
            If True, prepend <BOS> and append <EOS> if they exist.

        Returns
        -------
        ids:
            A list of integer token ids.
        """
        ids: List[int] = []

        if add_special_tokens and "<BOS>" in self.stoi:
            ids.append(self.stoi["<BOS>"])

        unk_id = self.stoi.get("<UNK>")

        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            elif unk_id is not None:
                ids.append(unk_id)
            else:
                # In practice, this should not happen because we train on the corpus first
                raise KeyError(f"Unknown character encountered and no <UNK> token defined: {repr(ch)}")

        if add_special_tokens and "<EOS>" in self.stoi:
            ids.append(self.stoi["<EOS>"])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token ids back into a string.

        Parameters
        ----------
        ids:
            List of token ids.
        skip_special_tokens:
            If True, remove <PAD>, <UNK>, <BOS>, <EOS> from the output.

        Returns
        -------
        text:
            The reconstructed string.
        """
        tokens: List[str] = []
        specials = set(self.special_tokens) if skip_special_tokens else set()

        for idx in ids:
            if idx not in self.itos:
                raise KeyError(f"Unknown token id during decoding: {idx}")

            token = self.itos[idx]

            if skip_special_tokens and token in specials:
                continue

            tokens.append(token)

        # For a character-level tokenizer, each token is a single character
        return "".join(tokens)