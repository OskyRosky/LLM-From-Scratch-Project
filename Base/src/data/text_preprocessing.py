from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def load_text(path: PathLike, encoding: str = "utf-8") -> str:
    """
    Load a text file and return its content as a single string.

    Parameters
    ----------
    path:
        Path to the text file.
    encoding:
        Encoding used to read the file.

    Returns
    -------
    text:
        The raw text contained in the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found: {path}")

    return path.read_text(encoding=encoding)


def normalize_newlines(text: str) -> str:
    """
    Normalize newlines to '\\n' and strip leading/trailing whitespace.
    """
    # Replace Windows and old Mac newlines with Unix-style
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip leading/trailing whitespace
    return text.strip()


def collapse_whitespace(text: str) -> str:
    """
    Collapse consecutive whitespace characters into a single space,
    but keep newlines intact.

    Example:
        'Hello   world\\n\\nThis   is' -> 'Hello world\\n\\nThis is'
    """
    import re

    # We replace sequences of spaces and tabs, but we do NOT remove newlines.
    def _collapse_line(line: str) -> str:
        return re.sub(r"[ \t]+", " ", line).strip()

    lines = text.split("\n")
    cleaned_lines = [_collapse_line(line) for line in lines]
    return "\n".join(cleaned_lines)


def basic_clean(text: str) -> str:
    """
    Apply a simple, conservative cleaning pipeline:
    - normalize newlines
    - collapse extra spaces and tabs
    - strip leading/trailing whitespace

    This function intentionally does NOT:
    - lowercase the text
    - remove punctuation

    Those choices will depend on the tokenizer/model design and we
    want to keep them explicit later.
    """
    text = normalize_newlines(text)
    text = collapse_whitespace(text)
    return text