# src/cli/finetune_instructions_tokens_debug.py

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.model.gpt import GPTModel, GPTConfig
from src.data.instruction_token_dataset import InstructionTokenDataset
from src.training.losses import language_modeling_loss


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cycle(dataloader):
    """Infinite dataloader iterator (never raises StopIteration)."""
    while True:
        for batch in dataloader:
            yield batch


def infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer minimal GPTConfig fields from a checkpoint state_dict (legacy ckpts)."""
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


def main():
    ap = argparse.ArgumentParser("Debug instruction-tuning (token-level)")
    ap.add_argument("--meta", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=256)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # ✅ only used if base_ckpt is legacy (missing model_config)
    ap.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Legacy only: used when base_ckpt lacks model_config.",
    )

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_json(args.meta)
    pad_id = int(meta["special_ids"]["pad"])

    # -----------------------
    # Dataset + DataLoader
    # -----------------------
    ds = InstructionTokenDataset(
        jsonl_path=args.jsonl,
        tokenizer_json_path=args.tokenizer_path,
        seq_len=int(args.seq_len),
        pad_id=pad_id,
    )

    if len(ds) == 0:
        raise ValueError(
            f"Instruction dataset is empty: {args.jsonl}. "
            "Check file path and JSONL format."
        )

    # ✅ critical: drop_last=False so small datasets still yield batches
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )

    if len(dl) == 0:
        raise ValueError(
            f"DataLoader produced 0 batches. len(ds)={len(ds)} batch_size={args.batch_size}. "
            "Fix by lowering --batch_size (<= len(dataset)) or keep drop_last=False."
        )

    it = cycle(dl)

    # -----------------------
    # Load base checkpoint
    # -----------------------
    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"]

    # Prefer model_config if present (new ckpts)
    model_cfg = ckpt.get("model_config", None)

    if isinstance(model_cfg, dict) and len(model_cfg) > 0:
        # ✅ Filter to keys GPTConfig actually accepts
        allowed = {"vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "dropout"}
        cfg_dict = {k: model_cfg[k] for k in allowed if k in model_cfg}

        # Safety defaults
        cfg_dict.setdefault("dropout", 0.0)

        # If ckpt forgot n_heads for any reason, fall back to CLI
        cfg_dict["n_heads"] = int(cfg_dict.get("n_heads", args.n_heads))
    else:
        # ✅ Legacy fallback (your ckpt_final_step_5000.pt)
        arch = infer_arch_from_state_dict(sd)
        cfg_dict = {
            "vocab_size": int(arch["vocab_size"]),
            "d_model": int(arch["d_model"]),
            "n_layers": int(arch["n_layers"]),
            "n_heads": int(args.n_heads),
            "max_seq_len": int(arch["max_seq_len"]),
            "dropout": 0.0,
        }

    # Override seq len for training if you want, but keep it consistent
    cfg_dict["max_seq_len"] = int(args.seq_len)

    # Quick sanity
    if int(cfg_dict["d_model"]) % int(cfg_dict["n_heads"]) != 0:
        raise ValueError(
            f"d_model={cfg_dict['d_model']} must be divisible by n_heads={cfg_dict['n_heads']}. "
            f"Try --n_heads 4 (or match training)."
        )

    cfg = GPTConfig(**cfg_dict)

    device = torch.device(args.device)
    model = GPTModel(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------
    # Train loop (debug)
    # -----------------------
    for step in range(1, int(args.steps) + 1):
       x, y, mask = next(it)
       x = x.to(device)
       y = y.to(device)
       mask = mask.to(device)
        # mask currently not used by language_modeling_loss unless you implemented it; ok for debug.

        opt.zero_grad(set_to_none=True)

        logits = model(x)  # expected (B,T,V)
        loss = language_modeling_loss(logits, y, loss_mask=mask, pad_id=pad_id)

        loss.backward()
        opt.step()

        if step == 1 or step % 10 == 0 or step == int(args.steps):
            # optional ppl (helps to see if it's learning)
            ppl = float(torch.exp(loss.detach()).item())
            print(f"[step {step:>3}/{args.steps}] loss={float(loss.detach().item()):.4f} ppl={ppl:.2f}")

    # -----------------------
    # Save debug checkpoint
    # -----------------------
    out_ckpt = out_dir / "ckpt_instr_debug.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": 0,
        "global_step": int(args.steps),
        "val_loss": None,
        "training_config": {
            "task": "instruction_tuning_debug",
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": str(device),
            "seed": int(args.seed),
            "seq_len": int(args.seq_len),
        },
        "model_config": {
            "vocab_size": int(cfg.vocab_size),
            "d_model": int(cfg.d_model),
            "n_layers": int(cfg.n_layers),
            "n_heads": int(cfg.n_heads),
            "max_seq_len": int(cfg.max_seq_len),
            "dropout": float(cfg.dropout),
        },
    }

    torch.save(payload, out_ckpt)
    print(f"\n✅ Saved instruction debug ckpt: {out_ckpt}")


if __name__ == "__main__":
    main()