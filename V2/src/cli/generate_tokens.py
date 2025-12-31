# src/cli/generate_tokens.py

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from src.model.gpt import GPTModel, GPTConfig

try:
    from tokenizers import Tokenizer  # HF tokenizers
except ImportError as e:
    raise ImportError("Missing dependency: tokenizers. Install with: pip install tokenizers") from e


# -------------------------
# IO helpers
# -------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_tokenizer_path(meta_path: str, tokenizer_path: str) -> str:
    """
    meta.json may store an absolute path (works on your Mac, breaks in Docker/other machines).

    We try, in order:
      1) tokenizer_path as-is
      2) relative to meta.json directory
      3) PROJECT_ROOT/models/tokenizers/oscar_bpe_v4/<basename>
      4) meta_dir/<basename>
    """
    if os.path.exists(tokenizer_path):
        return tokenizer_path

    meta_dir = os.path.dirname(os.path.abspath(meta_path))

    candidate = os.path.join(meta_dir, tokenizer_path)
    if os.path.exists(candidate):
        return candidate

    base = os.path.basename(tokenizer_path)
    project_root = os.path.abspath(os.path.join(meta_dir, "..", "..", ".."))  # best-effort
    candidate2 = os.path.join(project_root, "models", "tokenizers", "oscar_bpe_v4", base)
    if os.path.exists(candidate2):
        return candidate2

    candidate3 = os.path.join(meta_dir, base)
    if os.path.exists(candidate3):
        return candidate3

    raise FileNotFoundError(
        "Tokenizer file not found. Tried:\n"
        f"- {tokenizer_path}\n"
        f"- {candidate}\n"
        f"- {candidate2}\n"
        f"- {candidate3}\n"
        "Tip: store tokenizer_path as a relative path in meta.json for portability, "
        "or pass --tokenizer_path to override."
    )


def resolve_pack(pack_dir: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Resolve portable inference pack.

    Expected layout:
      <pack_dir>/
        meta.json
        tokenizer.json
        config.json   (optional)
      <pack_dir>/../
        ckpt_final_step_5000.pt  (default)

    Returns:
      meta_path, ckpt_path, tokenizer_path, config_path_or_None
    """
    pdir = Path(pack_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"--pack not found: {pdir}")

    meta_path = pdir / "meta.json"
    tok_path = pdir / "tokenizer.json"
    cfg_path = pdir / "config.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {pdir}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {pdir}")

    ckpt_path = pdir.parent / "ckpt_final_step_5000.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Tip: ensure ckpt_final_step_5000.pt is in {pdir.parent}"
        )

    config_path = str(cfg_path) if cfg_path.exists() else None
    return str(meta_path), str(ckpt_path), str(tok_path), config_path


# -------------------------
# Model helpers
# -------------------------

def infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    vocab_size, d_model = sd["tok_embedding.embedding.weight"].shape
    max_seq_len = sd["pos_embedding.pos_embedding.weight"].shape[0]

    layer_ids = set()
    pat = re.compile(r"^blocks\.(\d+)\.")
    for k in sd.keys():
        m = pat.match(k)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers = (max(layer_ids) + 1) if layer_ids else 0

    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "max_seq_len": int(max_seq_len),
        "n_layers": int(n_layers),
    }


def cfg_from_ckpt_or_fallback(
    ckpt: Dict[str, Any],
    sd: Dict[str, torch.Tensor],
    legacy_n_heads: int,
) -> Dict[str, Any]:
    """
    Prefer ckpt['model_config'] (new checkpoints), but filter to keys
    that GPTConfig actually supports. Otherwise infer from state_dict.
    """
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and len(model_cfg) > 0:
        allowed = {"vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "dropout"}
        cfg_dict = {k: model_cfg[k] for k in allowed if k in model_cfg}
        return cfg_dict

    arch = infer_arch_from_state_dict(sd)
    return {
        "vocab_size": arch["vocab_size"],
        "d_model": arch["d_model"],
        "n_layers": arch["n_layers"],
        "n_heads": int(legacy_n_heads),
        "max_seq_len": arch["max_seq_len"],
        "dropout": 0.0,
    }


@torch.no_grad()
def generate(
    model: GPTModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_id: Optional[int] = None,
    repetition_penalty: float = 1.0,   # ✅ NEW
) -> torch.Tensor:
    """
    Decoding policy:
      - top_k == 0  -> GREEDY (deterministic). temperature ignored.
      - top_k > 0   -> top-k sampling. temperature applies.

    Anti-loop (deterministic):
      - repetition_penalty > 1.0 penalizes already-generated tokens.
        Works in greedy and sampling.
    """
    model.eval()
    greedy = (top_k == 0)

    rep_pen = float(repetition_penalty) if repetition_penalty is not None else 1.0
    use_rep_pen = rep_pen > 1.0

    for _ in range(max_new_tokens):
        idx = input_ids[:, -block_size:]
        logits = model(idx)[:, -1, :]  # (B, V)

        # ✅ Repetition penalty (per sample)
        # Penalize tokens that already appeared in the generated sequence so far.
        if use_rep_pen:
            for b in range(logits.size(0)):
                prev = input_ids[b].unique()
                logits[b, prev] = logits[b, prev] / rep_pen

        if not greedy:
            logits = logits / max(float(temperature), 1e-6)

            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k=k, dim=-1)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    return input_ids


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate text from a token-level GPT checkpoint (BPE tokenizer)."
    )

    # Pack mode
    ap.add_argument(
        "--pack",
        type=str,
        default=None,
        help="Path to inference_pack directory. If set, auto-resolves meta/ckpt/tokenizer.",
    )

    # Legacy inputs (still supported)
    ap.add_argument("--meta", type=str, help="Path to meta.json for the tokenized dataset.")
    ap.add_argument("--ckpt", type=str, help="Path to checkpoint .pt file.")

    ap.add_argument("--prompt", type=str, required=True, help="Prompt text to generate from.")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="How many new tokens to generate.")

    # Decoding controls (default greedy)
    ap.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="0 = greedy (deterministic, recommended for eval). >0 = top-k sampling.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used when top_k > 0). For greedy eval keep 1.0.",
    )

    # ✅ NEW: deterministic anti-loop
    ap.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Deterministic anti-loop. 1.0 = off. Try 1.1–1.2 for Spanish to reduce repetitions.",
    )

    # Only used if ckpt lacks model_config
    ap.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Legacy only: used when checkpoint lacks model_config. Must match training.",
    )

    ap.add_argument("--seed", type=int, default=42, help="Only affects sampling (top_k > 0).")

    # Tokenizer override
    ap.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional override path to tokenizer.json. If not set, uses meta.json tokenizer_path.",
    )

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # Resolve pack
    if args.pack:
        meta_path, ckpt_path, tok_path, _cfg_path = resolve_pack(args.pack)
        args.meta = meta_path
        args.ckpt = ckpt_path
        args.tokenizer_path = tok_path

    if not args.meta or not args.ckpt:
        raise ValueError("Either use --pack OR provide --meta and --ckpt.")

    # Load meta + tokenizer
    meta = load_json(args.meta)
    tok_path_raw = args.tokenizer_path or meta["tokenizer_path"]
    tok_path = resolve_tokenizer_path(args.meta, tok_path_raw)
    tokenizer = Tokenizer.from_file(tok_path)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"]

    # Prefer ckpt model_config; fallback to inference from sd
    cfg_dict = cfg_from_ckpt_or_fallback(ckpt, sd, legacy_n_heads=args.n_heads)

    cfg = GPTConfig(**cfg_dict)

    # Build model
    device = torch.device(args.device)
    model = GPTModel(cfg).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("WARNING: load_state_dict not strict.")
        if missing:
            print("  Missing keys (sample):", missing[:10])
        if unexpected:
            print("  Unexpected keys (sample):", unexpected[:10])

    model.eval()

    # Greedy guard: temperature irrelevant
    if args.top_k == 0 and args.temperature != 1.0:
        print(
            f"[INFO] Greedy mode (top_k=0): ignoring temperature={args.temperature}. "
            "Use --top_k > 0 to enable sampling."
        )
        temperature = 1.0
    else:
        temperature = float(args.temperature)

    # Tokenize + generate
    enc = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

    eos_id = meta.get("special_ids", {}).get("eos", None)

    out_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        block_size=int(cfg.max_seq_len),
        temperature=temperature,
        top_k=int(args.top_k),
        eos_id=eos_id,
        repetition_penalty=float(args.repetition_penalty),
    )

    print(tokenizer.decode(out_ids[0].tolist()))


if __name__ == "__main__":
    main()