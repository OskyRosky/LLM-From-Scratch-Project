# src/cli/finetune_instructions_tokens_debug.py

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.gpt import GPTModel, GPTConfig
from src.data.instruction_token_dataset import InstructionTokenDataset
from src.training.losses import language_modeling_loss


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cycle(dataloader):
    """Infinite dataloader iterator (never raises StopIteration)."""
    while True:
        for batch in dataloader:
            yield batch


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

    # Prefer model_config if present
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and len(model_cfg) > 0:
        allowed = {"vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "dropout"}
        cfg_dict = {k: model_cfg[k] for k in allowed if k in model_cfg}
    else:
        raise ValueError("Base checkpoint missing model_config. (Expected after your trainer patch.)")

    # Override seq len for training if you want, but keep it consistent
    cfg_dict["max_seq_len"] = int(args.seq_len)

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
        x, y, mask = next(it)  # never stops
        x = x.to(device)
        y = y.to(device)
        # mask currently not used by language_modeling_loss unless you implemented it; ok for debug.

        opt.zero_grad(set_to_none=True)

        logits = model(x)  # expected (B,T,V)
        loss = language_modeling_loss(logits, y)

        loss.backward()
        opt.step()

        if step == 1 or step % 10 == 0 or step == int(args.steps):
            print(f"[step {step}/{args.steps}] loss={float(loss.detach().item()):.4f}")

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