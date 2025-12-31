import argparse
import json
from typing import Dict, Any, List, Tuple

import torch
from tokenizers import Tokenizer

from src.model.gpt import GPTModel, GPTConfig
from src.cli.generate_tokens import (
    load_json,
    resolve_tokenizer_path,
    cfg_from_ckpt_or_fallback,
    build_banned_ids,
    generate,
)


def load_eval_jsonl(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "expected" not in obj:
                raise ValueError(f"Linea {i}: falta 'prompt' o 'expected' -> {obj}")
            items.append({"prompt": obj["prompt"], "expected": obj["expected"]})
    if not items:
        raise ValueError("eval_jsonl está vacío.")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate instruction-tuned checkpoint on a JSONL (prompt/expected).")

    ap.add_argument("--eval_jsonl", type=str, required=True, help="Path to eval JSONL with {prompt, expected}.")
    ap.add_argument("--meta", type=str, required=True, help="Path to meta.json")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--tokenizer_path", type=str, default=None, help="Override tokenizer.json path")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")

    # Generation params (match your working setup)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--min_new_tokens", type=int, default=2)
    ap.add_argument("--stop_at_period", type=int, default=1)
    ap.add_argument("--period_id", type=int, default=19)

    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)

    ap.add_argument("--forbid_special", type=int, default=1)
    ap.add_argument("--ban_replacement", type=int, default=1)

    # Legacy only (if ckpt lacks model_config)
    ap.add_argument("--n_heads", type=int, default=4)

    # Output verbosity
    ap.add_argument("--show_all", type=int, default=0, help="1=print every case. 0=print only failures + summary.")

    args = ap.parse_args()
    torch.manual_seed(int(args.seed))

    # Load meta + tokenizer
    meta = load_json(args.meta)
    tok_path_raw = args.tokenizer_path or meta["tokenizer_path"]
    tok_path = resolve_tokenizer_path(args.meta, tok_path_raw)
    tok = Tokenizer.from_file(tok_path)

    special_ids = meta.get("special_ids", {})
    eos_id = special_ids.get("eos", None)

    # Load ckpt + model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"]

    cfg_dict = cfg_from_ckpt_or_fallback(ckpt, sd, legacy_n_heads=int(args.n_heads))
    cfg = GPTConfig(**cfg_dict)

    device = torch.device(args.device)
    model = GPTModel(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Build banned ids
    banned_ids = build_banned_ids(
        tokenizer=tok,
        special_ids={k: int(v) for k, v in special_ids.items()},
        forbid_special=bool(int(args.forbid_special)),
        ban_replacement=bool(int(args.ban_replacement)),
        vocab_size=int(cfg.vocab_size),
    )

    # Load eval items
    items = load_eval_jsonl(args.eval_jsonl)

    ok = 0
    failures: List[Tuple[str, str, str]] = []

    for it in items:
        user_q = it["prompt"]
        expected = it["expected"].strip()

        full_prompt = f"<instr> {user_q}<resp>"
        enc = tok.encode(full_prompt)
        x = torch.tensor([enc.ids], dtype=torch.long, device=device)

        out = generate(
            model=model,
            input_ids=x,
            max_new_tokens=int(args.max_new_tokens),
            block_size=int(cfg.max_seq_len),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            eos_id=(int(eos_id) if eos_id is not None else None),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram=int(args.no_repeat_ngram),
            min_new_tokens=int(args.min_new_tokens),
            banned_ids=banned_ids,
            debug_next=0,
            tokenizer=tok,
            stop_at_period=bool(int(args.stop_at_period)),
            period_id=int(args.period_id),
        )

        full_ids = out[0].tolist()
        gen_only_ids = full_ids[len(enc.ids):]
        pred = tok.decode(gen_only_ids).strip()

        match = (pred == expected)
        if match:
            ok += 1
        else:
            failures.append((user_q, expected, pred))

        if int(args.show_all) == 1:
            print("-" * 70)
            print("PROMPT   :", user_q)
            print("EXPECTED :", expected)
            print("PRED     :", pred)
            print("MATCH    :", match)

    total = len(items)
    acc = ok / total

    print("\n" + "=" * 70)
    print(f"Eval: {ok}/{total} = {acc:.3f}")

    if failures:
        print("\nFailures:")
        for q, exp, pred in failures:
            print("-" * 70)
            print("PROMPT   :", q)
            print("EXPECTED :", exp)
            print("PRED     :", pred)


if __name__ == "__main__":
    main()
