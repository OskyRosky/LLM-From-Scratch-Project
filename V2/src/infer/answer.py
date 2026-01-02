# src/infer/answer.py
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List, Tuple, Dict, Any

import torch

from src.model.gpt import GPTModel, GPTConfig
from src.infer.tokenizer_wrapper import load_tokenizer_and_specials

# Hechos verificados (FAQ -> FACT, no respuesta final directa)
from src.inference.faq_fallback import faq_fact

# Reusar helpers validados
from src.cli.generate_tokens import (
    cfg_from_ckpt_or_fallback,
    build_banned_ids,
    generate,
)


# -----------------------------------------------------------------------------
# Assets (cache)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Assets:
    model: GPTModel
    tokenizer: object  # HF Tokenizer
    cfg: GPTConfig
    eos_id: Optional[int]
    banned_ids: List[int]
    device: str


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


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

    cfg_dict = cfg_from_ckpt_or_fallback(ckpt, sd, legacy_n_heads=int(n_heads_legacy))
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


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
def _try_fix_mojibake(s: str) -> str:
    if "Ã" not in s and "Â" not in s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        return s


def _clean_text(s: str) -> str:
    s = _try_fix_mojibake(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()

    # arreglos típicos
    s = s.replace("..", ".")
    s = re.sub(r"\s+\.", ".", s)

    return s.strip()


# -----------------------------------------------------------------------------
# Guards: PRIVATE
# -----------------------------------------------------------------------------
_PRIVATE_PATTERNS = [
    r"\bmi\b", r"\bmis\b", r"\bmío\b", r"\bmía\b",
    r"\bfavorito\b", r"\bfavorita\b",
    r"\banime\b", r"\bjefes\b", r"\bjefe\b",
    r"\bhermano\b", r"\bhermana\b", r"\bgemelo\b", r"\bgemela\b",
    r"\bedad\b", r"\bnombre\b", r"\bnombres\b",
    r"\bmis\s+\d+\b",
]


def is_private_question(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in _PRIVATE_PATTERNS)


# Mensajes de rechazo (humanos y consistentes)
_REFUSE_PRIVATE = "No tengo información personal tuya en mi entrenamiento, así que no puedo responder eso."
_REFUSE_UNKNOWN = "No tengo suficiente base en mi entrenamiento para responder eso con precisión."

# Marcadores para detectar que “ya es rechazo”
_REFUSE_MARKERS = [
    "no tengo información",              # cubre private
    "no tengo esa información",          # texto típico del modelo
    "en mi entrenamiento actual",        # refuerzo del mismo caso
    "no tengo suficiente base",          # tu unknown unificado
    "no tengo suficiente información",   # por compatibilidad con runs anteriores
    "no puedo responder",
]


def _contains_refuse_marker(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _REFUSE_MARKERS)


# -----------------------------------------------------------------------------
# Unknown guard (anti-derrail en preguntas sin FACT)
# -----------------------------------------------------------------------------
_ANCHOR_WORDS = [
    "gato", "gatos", "felino", "felinos", "félido", "félidos",
    "perro", "perros", "canino", "caninos", "cánido", "cánidos",
    "capital", "costa rica", "francia",
]

_TOPIC_HINTS = [
    "fotosíntesis", "relatividad", "cuántica", "quantica",
    "sistema solar", "planeta", "luna",
    "machine learning", "aprendizaje automático", "llm",
    "física", "fisica",
]


def _unknown_guard(question: str, answer: str) -> bool:
    q = question.lower()
    a = answer.lower()

    # Si ya es rechazo (cualquier variante), no tocar
    if any(m in a for m in _REFUSE_MARKERS):
        return False

    has_topic = any(t in q for t in _TOPIC_HINTS)
    anchor_hits = sum(1 for w in _ANCHOR_WORDS if w in a)

    return bool(has_topic and anchor_hits >= 2)


# -----------------------------------------------------------------------------
# Fact validation (anti-alucinación)
# -----------------------------------------------------------------------------
_JUNK_PATTERNS = [
    r"enlaces externos", r"sitio web oficial", r"sitio oficial",
    r"\bcánid", r"\bfélid",
]


def _fact_key(fact: str) -> str:
    m = re.search(r"\bes\s+(.+?)\.\s*$", fact.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return fact.strip().rstrip(".").split()[-1].strip()


def _looks_bad(answer: str) -> bool:
    a = answer.lower()
    return any(re.search(p, a) for p in _JUNK_PATTERNS)


def _validate_against_fact(answer: str, fact: str) -> bool:
    key = _fact_key(fact)
    if not key:
        return False
    return key.lower() in answer.lower() and not _looks_bad(answer)


# -----------------------------------------------------------------------------
# Core generation
# -----------------------------------------------------------------------------
def _generate_only(
    user_prompt: str,
    *,
    assets: Assets,
    max_new_tokens: int,
    min_new_tokens: int,
    stop_at_period: int,
    period_id: int,
    top_k: int,
    temperature: float,
    repetition_penalty: float,
    no_repeat_ngram: int,
) -> str:
    prompt = f"<instr> {user_prompt}<resp>"
    enc = assets.tokenizer.encode(prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=torch.device(assets.device))

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
    return _clean_text(assets.tokenizer.decode(gen_only))


# -----------------------------------------------------------------------------
# Public APIs
# -----------------------------------------------------------------------------
def answer(
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
) -> str:
    a, _ = answer_with_meta(
        user_prompt,
        meta_path=meta_path,
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        device=device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        stop_at_period=stop_at_period,
        period_id=period_id,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram=no_repeat_ngram,
        seed=seed,
        n_heads_legacy=n_heads_legacy,
        forbid_special=forbid_special,
        ban_replacement=ban_replacement,
    )
    return a


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
) -> Tuple[str, Dict[str, Any]]:
    meta_path = meta_path or _env("LLM_META", "models/tokenized/oscar_bpe_v4/meta.json")
    tokenizer_path = tokenizer_path or _env("LLM_TOKENIZER", "models/tokenizers/oscar_bpe_v4/tokenizer.json")
    ckpt_path = ckpt_path or _env(
        "LLM_CKPT",
        "models/checkpoints/instr_mini_run_masked_eos_CLOSE_v4/ckpt_instr_debug.pt",
    )
    device = device or _env("LLM_DEVICE", "cpu")

    if seed is not None:
        torch.manual_seed(int(seed))

    # 1) Private guard
    if is_private_question(user_prompt):
        return _REFUSE_PRIVATE, {
            "used_private_guard": True,
            "used_fact": False,
            "fact": "",
            "fact_validation_fallback": False,
            "unknown_guard_triggered": False,
            "refuse_reason": "private",
            "took_ms": 0.0,
        }

    # 2) Load assets
    assets = _load_assets_once(
        meta_path=meta_path,
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        device=device,
        n_heads_legacy=int(n_heads_legacy),
        forbid_special=int(forbid_special),
        ban_replacement=int(ban_replacement),
    )

    # 3) Fact
    fact = faq_fact(user_prompt) or ""
    used_fact = bool(fact.strip())

    t0 = time.perf_counter()

    # 3A) Con FACT: anclado y seguro
    if used_fact:
        guided_prompt = (
            "Responde con UNA sola oración, breve y correcta. "
            "No agregues información extra.\n"
            f"HECHO: {fact}\n"
            f"PREGUNTA: {user_prompt}"
        )

        ans = _generate_only(
            guided_prompt,
            assets=assets,
            max_new_tokens=min(int(max_new_tokens), 32),
            min_new_tokens=int(min_new_tokens),
            stop_at_period=1,
            period_id=int(period_id),
            top_k=int(top_k),
            temperature=float(temperature),
            repetition_penalty=max(float(repetition_penalty), 1.15),
            no_repeat_ngram=max(int(no_repeat_ngram), 3),
        )

        took_ms = (time.perf_counter() - t0) * 1000.0

        fact_ok = _validate_against_fact(ans, fact)
        too_short = (ans.strip().rstrip(".") == _fact_key(fact).strip()) or (len(ans.strip()) <= 8)

        if (not fact_ok) or too_short:
            return _clean_text(fact), {
                "used_private_guard": False,
                "used_fact": True,
                "fact": fact,
                "fact_validation_fallback": True,
                "unknown_guard_triggered": False,
                "refuse_reason": "",
                "took_ms": round(took_ms, 2),
            }

        return ans, {
            "used_private_guard": False,
            "used_fact": True,
            "fact": fact,
            "fact_validation_fallback": False,
            "unknown_guard_triggered": False,
            "refuse_reason": "",
            "took_ms": round(took_ms, 2),
        }

    # 4) LLM normal (sin fact)
    ans = _generate_only(
        user_prompt,
        assets=assets,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(min_new_tokens),
        stop_at_period=int(stop_at_period),
        period_id=int(period_id),
        top_k=int(top_k),
        temperature=float(temperature),
        repetition_penalty=float(repetition_penalty),
        no_repeat_ngram=int(no_repeat_ngram),
    )
    took_ms = (time.perf_counter() - t0) * 1000.0

    # 4A) Si el modelo devolvió un “No tengo...” por sí solo:
    # - unificamos el texto a _REFUSE_UNKNOWN
    # - seteamos refuse_reason para UI
    if _contains_refuse_marker(ans):
        return _REFUSE_UNKNOWN, {
            "used_private_guard": False,
            "used_fact": False,
            "fact": "",
            "fact_validation_fallback": False,
            "unknown_guard_triggered": False,
            "refuse_reason": "unknown_no_knowledge",
            "took_ms": round(took_ms, 2),
        }

    # 4B) Unknown guard SOLO aquí (cuando se descarrila)
    if _unknown_guard(user_prompt, ans):
        return _REFUSE_UNKNOWN, {
            "used_private_guard": False,
            "used_fact": False,
            "fact": "",
            "fact_validation_fallback": False,
            "unknown_guard_triggered": True,
            "refuse_reason": "unknown_derail",
            "took_ms": round(took_ms, 2),
        }

    return ans, {
        "used_private_guard": False,
        "used_fact": False,
        "fact": "",
        "fact_validation_fallback": False,
        "unknown_guard_triggered": False,
        "refuse_reason": "",
        "took_ms": round(took_ms, 2),
    }