# src/data/convert_parquet_to_txt.py

import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def parquet_to_txt(
    parquet_path: str,
    txt_path: str,
    text_column: str = "text",
    max_rows: Optional[int] = None,
) -> None:
    """
    Convierte un archivo .parquet de OSCAR a un corpus .txt concatenado.

    Parameters
    ----------
    parquet_path:
        Ruta al archivo .parquet de entrada.
    txt_path:
        Ruta al archivo .txt de salida.
    text_column:
        Nombre de la columna que contiene el texto (en OSCAR suele ser 'text').
    max_rows:
        Si se indica, limita el número de filas leídas (útil para pruebas).
    """
    parquet_path = Path(parquet_path)
    txt_path = Path(txt_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    print(f"[INFO] Loading parquet from: {parquet_path}")

    # Leemos con pandas (requiere pyarrow o fastparquet instalado)
    df = pd.read_parquet(parquet_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in parquet. "
            f"Available columns: {list(df.columns)}"
        )

    if max_rows is not None:
        df = df.head(max_rows)
        print(f"[INFO] Using only first {len(df)} rows for the corpus.")
    else:
        print(f"[INFO] Using all {len(df)} rows for the corpus.")

    texts = df[text_column].astype(str).tolist()

    print("[INFO] Concatenating rows into a single text corpus...")
    corpus = "\n\n".join(t.strip() for t in texts if t and not t.isspace())

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(corpus, encoding="utf-8")

    size_mb = txt_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Wrote corpus to {txt_path} (size: {size_mb:.2f} MB)")


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  python -m src.data.convert_parquet_to_txt "
            "INPUT.parquet OUTPUT.txt [max_rows]"
        )
        raise SystemExit(1)

    parquet_path = sys.argv[1]
    txt_path = sys.argv[2]

    max_rows: Optional[int]
    if len(sys.argv) >= 4:
        max_rows = int(sys.argv[3])
    else:
        max_rows = None

    parquet_to_txt(parquet_path, txt_path, max_rows=max_rows)


if __name__ == "__main__":
    main()