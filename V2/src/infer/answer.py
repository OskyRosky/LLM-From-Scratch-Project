# src/infer/answer.py
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List, Dict, Tuple

import torch

from src.model.gpt import GPTModel, GPTConfig
from src.infer.tokenizer_wrapper import load_tokenizer_and_specials

# Reusar helpers validados
from src.cli.generate_tokens import (
    cfg_from_ckpt_or_fallback,
    build_banned_ids,
    generate,
)

# FACTS (FAQ grounding)
from src.inference.faq_fallback import faq_fact


REFUSAL = "No tengo esa información en mi entrenamiento actual."


@dataclass(frozen=True)
class Assets:
    model: GPTModel
    tokenizer: object  # HF Tokenizer
    cfg: GPTConfig
    eos_id: Optional[int]
    banned_ids: List[int]
    device: str  # keep original string for clarity


# -------------------------
# Text cleaning
# -------------------------
def _try_fix_mojibake(s: str) -> str:
    if "Ã" not in s and "Â" not in s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        return s



def _dedupe_repeats(s: str) -> str:
    # Colapsa repeticiones exactas del mismo segmento separado por espacios.
    parts = [p.strip() for p in s.split(" ") if p.strip()]
    if len(parts) < 6:
        return s
    # Caso típico: "San José. San José. San José."
    if len(set(parts)) <= 3:
        # reconstruye con el primer segmento hasta el primer punto
        m = re.match(r"^(.+?\.)\s+\1(\s+\1)+\s*$", s.strip())
        if m:
            return m.group(1).strip()
    return s

def _clean_text(s: str) -> str:
    s = _try_fix_mojibake(s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()

    s = s.replace("..", ".")
    s = re.sub(r"\s+\.", ".", s)
    return s


# -------------------------
# Private guard (simple + efectivo)
# -------------------------
_PRIVATE_KEYWORDS = [
    "mi ", "mis ", "mío", "mía",
    "favorito", "favorita",
    "anime", "jefes", "jefe",
    "hermano", "hermana",
    "edad",
    "perros", "perro",  # ojo: aquí solo para "mis perros"/"nombre de mis perros"
    "nombre de mis", "nombres de mis",
]

def is_private_question(text: str) -> bool:
    t = (text or "").strip().lower()

    # patrones obvios de info personal
    if "mi " in t or "mis " in t:
        # "mi anime", "mis perros", "mis 3 jefes", etc.
        if any(k in t for k in ["anime", "jef", "hermano", "edad", "perro", "nombre", "nombres", "favorit"]):
            return True

    # "cuáles son los tres jefes que he tenido"
    if "que he tenido" in t and ("jefe" in t or "jefes" in t):
        return True

    # "nombre de mis X"
    if "nombre de mis" in t or "nombres de mis" in t:
        return True

    # "mi anime favorito"
    if "anime" in t and "favorit" in t:
        return True

    return False


# -------------------------
# Config defaults via env
# -------------------------
def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# -------------------------
# Asset loading (cacheado)
# -------------------------
@lru_cache(maxsize=4)
def _load_assets_once(
    meta_path: str,
    ckpt_path: str,
    tokenizer_path: str,
    device: str,
    n_heads_legacy: int,
    forbid_special: int,
    ban_replacement: int,
) -> Assets:
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

    return Assets(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        eos_id=int(eos_id) if eos_id is not None else None,
        banned_ids=banned_ids,
        device=device,
    )


def clear_cache() -> None:
    _load_assets_once.cache_clear()


# -------------------------
# Internal: build prompt
# -------------------------
def _build_prompt(user_prompt: str, fact: str) -> str:
    up = (user_prompt or "").strip()
    f = (fact or "").strip()

    if f:
        # Forzamos frase completa y “humana”
        return (
            "<instr> Responde de forma breve, clara y en una oración completa. "
            "Usa únicamente este hecho verificado como base.\n"
            f"HECHO: {f}\n"
            f"PREGUNTA: {up}\n"
            "<resp>"
        )

    return f"<instr> {up}<resp>"


# -------------------------
# Public API
# -------------------------
def answer_with_meta(
    user_prompt: str,
    *,
    meta_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    min_new_tokens: int = 2,
    stop_at_period: int = 1,
    period_id: int = 19,
    top_k: int = 0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    seed: Optional[int] = 42,
    n_heads_legacy: int = 4,
    forbid_special: int = 1,
    ban_replacement: int = 1,
) -> Tuple[str, Dict[str, object]]:
    # 0) Private guard
    if is_private_question(user_prompt):
        return REFUSAL, {
            "used_private_guard": True,
            "used_fact": False,
            "fact": "",
            "took_ms": 0.0,
        }

    # 1) Fact lookup (returns "" si no hay)
    fact = faq_fact(user_prompt) or ""
    has_fact = bool(fact.strip())

    # Defaults por env
    meta_path = meta_path or _env("LLM_META", "models/tokenized/oscar_bpe_v4/meta.json")
    tokenizer_path = tokenizer_path or _env("LLM_TOKENIZER", "models/tokenizers/oscar_bpe_v4/tokenizer.json")
    ckpt_path = ckpt_path or _env("LLM_CKPT", "models/checkpoints/instr_mini_run_masked_eos_CLOSE_v4/ckpt_instr_debug.pt")
    device = device or _env("LLM_DEVICE", "cpu")

    if seed is not None:
        torch.manual_seed(int(seed))

    assets = _load_assets_once(
        meta_path=meta_path,
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        device=device,
        n_heads_legacy=int(n_heads_legacy),
        forbid_special=int(forbid_special),
        ban_replacement=int(ban_replacement),
    )

    prompt = _build_prompt(user_prompt, fact=fact)
    enc = assets.tokenizer.encode(prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=torch.device(assets.device))

    greedy = (int(top_k) == 0)
    temp = 1.0 if greedy else float(temperature)

    t0 = time.perf_counter()

    out_ids = generate(
        model=assets.model,
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        block_size=int(assets.cfg.max_seq_len),
        temperature=float(temp),
        top_k=int(top_k),
        eos_id=assets.eos_id,
        repetition_penalty=float(max(repetition_penalty, 1.15) if has_fact else repetition_penalty),
        no_repeat_ngram=int(max(no_repeat_ngram, 3) if has_fact else no_repeat_ngram),
        min_new_tokens=int(max(min_new_tokens, 8) if has_fact else min_new_tokens),
        banned_ids=assets.banned_ids,
        debug_next=0,
        tokenizer=assets.tokenizer,
        stop_at_period=(False if has_fact else bool(int(stop_at_period))),
        period_id=int(period_id),
    )

    took_ms = (time.perf_counter() - t0) * 1000.0

    full = out_ids[0].tolist()
    gen_only = full[len(enc.ids):]
    text = assets.tokenizer.decode(gen_only)
    text = _clean_text(text)
    text = _dedupe_repeats(text)

    # Si hay HECHO y el modelo se desvía (tiny model), forzamos respuesta grounded
    if has_fact:
        low = text.lower()
        if ("no tengo esa información" in low) or (len(text.strip()) < 5):
            text = fact.strip()

    return text, {
        "used_private_guard": False,
        "used_fact": has_fact,
        "fact": fact.strip(),
        "took_ms": round(float(took_ms), 2),
    }


def answer(*args, **kwargs) -> str:
    text, _meta = answer_with_meta(*args, **kwargs)
    return text
