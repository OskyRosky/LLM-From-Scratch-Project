from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import torch

from src.model.gpt import GPTModel, GPTConfig
from src.infer.tokenizer_wrapper import load_tokenizer_and_specials

# Reusamos helpers ya validados (sin reescribir lógica)
from src.cli.generate_tokens import (
    cfg_from_ckpt_or_fallback,
    build_banned_ids,
    generate,
)

_ASSETS = None  # cache global (simple y efectivo)


@dataclass
class Assets:
    model: GPTModel
    tokenizer: object  # HF Tokenizer
    cfg: GPTConfig
    eos_id: Optional[int]
    banned_ids: List[int]
    device: torch.device


def _load_assets_once(
    meta_path: str,
    ckpt_path: str,
    tokenizer_path: str,
    device: str = "cpu",
    n_heads_legacy: int = 4,
    forbid_special: int = 1,
    ban_replacement: int = 1,
) -> Assets:
    global _ASSETS
    if _ASSETS is not None:
        return _ASSETS

    dev = torch.device(device)

    tokinfo = load_tokenizer_and_specials(meta_path, tokenizer_path)
    tokenizer = tokinfo.tokenizer
    eos_id = tokinfo.eos_id

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    cfg_dict = cfg_from_ckpt_or_fallback(ckpt, sd, legacy_n_heads=n_heads_legacy)
    cfg = GPTConfig(**cfg_dict)

    model = GPTModel(cfg).to(dev)
    model.load_state_dict(sd, strict=False)
    model.eval()

    banned_ids = build_banned_ids(
        tokenizer=tokenizer,
        special_ids={k: int(v) for k, v in tokinfo.special_ids.items()},
        forbid_special=bool(int(forbid_special)),
        ban_replacement=bool(int(ban_replacement)),
        vocab_size=int(cfg.vocab_size),
    )

    _ASSETS = Assets(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        eos_id=int(eos_id) if eos_id is not None else None,
        banned_ids=banned_ids,
        device=dev,
    )
    return _ASSETS


def answer(
    user_prompt: str,
    *,
    meta_path: str,
    ckpt_path: str,
    tokenizer_path: str,
    device: str = "cpu",
    max_new_tokens: int = 64,
    min_new_tokens: int = 2,
    stop_at_period: int = 1,
    period_id: int = 19,
    top_k: int = 0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
) -> str:
    assets = _load_assets_once(
        meta_path=meta_path,
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        device=device,
    )

    prompt = f"<instr> {user_prompt}<resp>"
    enc = assets.tokenizer.encode(prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=assets.device)

    # Greedy => temperatura no aplica (igual que en CLI)
    greedy = (int(top_k) == 0)
    temp = 1.0 if greedy else float(temperature)

    out_ids = generate(
        model=assets.model,
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        block_size=int(assets.cfg.max_seq_len),
        temperature=float(temp),
        top_k=int(top_k),
        eos_id=assets.eos_id,
        repetition_penalty=float(repetition_penalty),
        no_repeat_ngram=int(no_repeat_ngram),
        min_new_tokens=int(min_new_tokens),
        banned_ids=assets.banned_ids,
        debug_next=0,
        tokenizer=assets.tokenizer,
        stop_at_period=bool(int(stop_at_period)),
        period_id=int(period_id),
    )

    full = out_ids[0].tolist()
    gen_only = full[len(enc.ids):]
    text = assets.tokenizer.decode(gen_only)

    # limpieza mínima
    return text.strip()
