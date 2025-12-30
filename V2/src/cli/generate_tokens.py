# src/cli/generate_tokens.py

import argparse
import json
import os
import re
from typing import Any, Dict, Optional

import torch

from src.model.gpt import GPTModel, GPTConfig

try:
    from tokenizers import Tokenizer  # HF tokenizers
except ImportError as e:
    raise ImportError(
        "Missing dependency: tokenizers. Install with: pip install tokenizers"
    ) from e


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_tokenizer_path(meta_path: str, tokenizer_path: str) -> str:
    """
    Resolve tokenizer.json path robustly.

    We try, in order:
      1) tokenizer_path as-is (absolute or existing relative path)
      2) relative to the meta.json directory
      3) PROJECT_ROOT/models/tokenizers/oscar_bpe_v4/<basename>
      4) meta_dir/<basename>

    Note: meta.json currently stores an absolute path (works on your Mac),
    but will break in Docker/other machines. This resolver makes it portable.
    """
    # 1) As-is
    if tokenizer_path and os.path.exists(tokenizer_path):
        return tokenizer_path

    meta_dir = os.path.dirname(os.path.abspath(meta_path))

    # 2) Relative to meta.json directory (if meta stored a relative path)
    if tokenizer_path:
        candidate = os.path.join(meta_dir, tokenizer_path)
        if os.path.exists(candidate):
            return candidate

    # 3) Fallback by basename to a common project location
    base = os.path.basename(tokenizer_path) if tokenizer_path else "tokenizer.json"
    project_root = os.path.abspath(os.path.join(meta_dir, "..", "..", ".."))  # best-effort
    candidate2 = os.path.join(project_root, "models", "tokenizers", "oscar_bpe_v4", base)
    if os.path.exists(candidate2):
        return candidate2

    # 4) Last resort: meta_dir + basename
    candidate3 = os.path.join(meta_dir, base)
    if os.path.exists(candidate3):
        return candidate3

    raise FileNotFoundError(
        "Tokenizer file not found. Tried:\n"
        f"- {tokenizer_path}\n"
        f"- {os.path.join(meta_dir, tokenizer_path) if tokenizer_path else '(meta_dir + <none>)'}\n"
        f"- {candidate2}\n"
        f"- {candidate3}\n"
        "Tip: pass --tokenizer_path explicitly or store tokenizer_path as a relative path in meta.json."
    )


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


@torch.no_grad()
def generate(
    model: GPTModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int = 0,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    model.eval()

    for _ in range(max_new_tokens):
        idx = input_ids[:, -block_size:]
        logits = model(idx)  # (B, T, V)
        logits = logits[:, -1, :]  # (B, V)

        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature

        if top_k and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        # Note: EOS check assumes batch=1 (your CLI uses a single prompt).
        if eos_id is not None and int(next_id.item()) == int(eos_id):
            break

    return input_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text from a token-level GPT checkpoint.")
    ap.add_argument("--meta", type=str, required=True, help="Path to meta.json for the tokenized dataset.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file.")
    ap.add_argument("--prompt", type=str, required=True, help="Prompt text to generate from.")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0, help="0 = greedy. >0 enables top-k sampling.")
    ap.add_argument("--n_heads", type=int, default=4, help="Must match training (default=4 based on your logs).")
    ap.add_argument("--seed", type=int, default=42)

    # ✅ NEW: optional tokenizer override (for Docker/portability)
    ap.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional override path to tokenizer.json. If not set, uses meta.json['tokenizer_path'].",
    )

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    meta = load_json(args.meta)

    # Choose tokenizer path: override > meta.json
    tok_path_raw = args.tokenizer_path or meta["tokenizer_path"]
    tok_path = resolve_tokenizer_path(args.meta, tok_path_raw)
    tokenizer = Tokenizer.from_file(tok_path)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"]

    arch = infer_arch_from_state_dict(sd)

    # Safety
    if arch["d_model"] % args.n_heads != 0:
        raise ValueError(f"d_model={arch['d_model']} must be divisible by n_heads={args.n_heads}")

    # Build config to match training as closely as possible.
    cfg = GPTConfig(
        vocab_size=arch["vocab_size"],
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=args.n_heads,
        max_seq_len=arch["max_seq_len"],
        dropout=0.0,
    )

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

    # Tokenize prompt
    enc = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

    eos_id = meta.get("special_ids", {}).get("eos", None)

    out_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        block_size=arch["max_seq_len"],
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=eos_id,
    )

    text = tokenizer.decode(out_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()