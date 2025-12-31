# src/cli/finetune_instructions_tokens_debug.py

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.gpt import GPTModel, GPTConfig
from src.data.instruction_token_dataset import InstructionTokenDataset


def lm_loss_ignore_pad(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    logits:  (B,T,V)
    targets: (B,T)
    Ignora pad_id usando ignore_index=-100.
    """
    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)

    targets_1d = targets_1d.clone()
    targets_1d[targets_1d == pad_id] = -100

    return F.cross_entropy(logits_2d, targets_1d, ignore_index=-100)


def main():
    ap = argparse.ArgumentParser("Debug instruction fine-tune (token-level)")
    ap.add_argument("--meta", required=True, help="meta.json (for special_ids)")
    ap.add_argument("--tokenizer_path", required=True, help="tokenizer.json")
    ap.add_argument("--base_ckpt", required=True, help="Base pretrain checkpoint (.pt)")
    ap.add_argument("--jsonl", required=True, help="Instruction data jsonl")
    ap.add_argument("--out_dir", required=True, help="Output checkpoint dir")

    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    pad_id = int(meta["special_ids"]["pad"])

    # Dataset
    ds = InstructionTokenDataset(
        jsonl_path=args.jsonl,
        tokenizer_json_path=args.tokenizer_path,
        seq_len=args.seq_len,
        pad_id=pad_id,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Load base ckpt
    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"]

    # Config: prefer ckpt model_config if present, else infer minimal
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and len(model_cfg) > 0:
        allowed = {"vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "dropout"}
        cfg_dict = {k: model_cfg[k] for k in allowed if k in model_cfg}
        # Para debug: no dropout (más estable)
        cfg_dict["dropout"] = 0.0
    else:
        # fallback hard (tu arquitectura conocida)
        cfg_dict = {
            "vocab_size": 4096,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 4,
            "max_seq_len": args.seq_len,
            "dropout": 0.0,
        }

    cfg = GPTConfig(**cfg_dict)
    model = GPTModel(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    it = iter(dl)
    for step in range(1, args.steps + 1):
        try:
            x, y, _mask = next(it)
        except StopIteration:
            it = iter(dl)
            x, y, _mask = next(it)

        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)  # (B,T,V)
        loss = lm_loss_ignore_pad(logits, y, pad_id=pad_id)
        loss.backward()
        opt.step()

        if step == 1 or step % 10 == 0:
            ppl = torch.exp(loss.detach()).item()
            print(f"[step {step:>3}/{args.steps}] loss={loss.item():.4f} ppl={ppl:.2f}")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / "ckpt_instr_debug.pt"

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": 0,
        "global_step": int(args.steps),
        "val_loss": None,
        "training_config": {
            "task": "instruction_tuning_debug",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "seed": args.seed,
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