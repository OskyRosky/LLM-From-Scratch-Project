import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build a local text corpus from OSCAR (streaming).")
    parser.add_argument("--lang", type=str, default="es", help="Language code (default: es).")
    parser.add_argument("--dataset", type=str, default="oscar-corpus/OSCAR-2301", help="HF dataset repo.")
    parser.add_argument("--split", type=str, default="train", help="Split to stream (default: train).")
    parser.add_argument("--max_samples", type=int, default=200_000, help="Max documents to write.")
    parser.add_argument("--min_chars", type=int, default=200, help="Filter out short samples.")
    parser.add_argument("--out", type=str, default="data/raw/oscar_corpus.txt", help="Output txt path.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Missing dependency 'datasets'. Run: pip install datasets") from e

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # OSCAR configs are typically language-coded (e.g., 'es')
    ds = load_dataset(args.dataset, args.lang, split=args.split, streaming=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            # OSCAR usually exposes text under 'text'
            text = row.get("text", "")
            if not isinstance(text, str):
                continue
            text = text.strip().replace("\r", " ")
            if len(text) < args.min_chars:
                continue

            f.write(text)
            f.write("\n")
            written += 1

            if written >= args.max_samples:
                break

    print(f"[OK] Wrote {written:,} samples to: {out_path.resolve()}")

if __name__ == "__main__":
    main()