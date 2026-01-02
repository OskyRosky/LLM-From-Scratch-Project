from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import json
from tokenizers import Tokenizer


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class TokInfo:
    tokenizer: Tokenizer
    special_ids: Dict[str, int]
    pad_id: Optional[int]
    eos_id: Optional[int]
    instr_id: Optional[int]
    resp_id: Optional[int]


def load_tokenizer_and_specials(meta_path: str, tokenizer_path: str) -> TokInfo:
    meta = load_json(meta_path)
    special_ids = {k: int(v) for k, v in meta.get("special_ids", {}).items()}

    tok = Tokenizer.from_file(tokenizer_path)

    pad_id = special_ids.get("pad")
    eos_id = special_ids.get("eos")
    instr_id = special_ids.get("instr")
    resp_id = special_ids.get("resp")

    return TokInfo(
        tokenizer=tok,
        special_ids=special_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        instr_id=instr_id,
        resp_id=resp_id,
    )


def encode(tok: Tokenizer, text: str) -> List[int]:
    return tok.encode(text).ids


def decode(tok: Tokenizer, ids: List[int]) -> str:
    return tok.decode(ids)
