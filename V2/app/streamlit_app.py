# app/streamlit_app.py
import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------
# Asegurar imports "src..."
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # V2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer.answer import answer_with_meta, clear_cache  # noqa: E402

# ---------------------------------------------------------------------
# Config por env (recomendado)
# ---------------------------------------------------------------------
DEFAULT_META = os.getenv("LLM_META", "models/tokenized/oscar_bpe_v4/meta.json")
DEFAULT_TOK = os.getenv("LLM_TOKENIZER", "models/tokenizers/oscar_bpe_v4/tokenizer.json")
DEFAULT_CKPT = os.getenv(
    "LLM_CKPT",
    "models/checkpoints/instr_mini_run_masked_eos_CLOSE_v4/ckpt_instr_debug.pt",
)
DEFAULT_DEVICE = os.getenv("LLM_DEVICE", "mps")

# ---------------------------------------------------------------------
# Streamlit basic page
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LLM From Scratch – Token Chat (BPE)",
    page_icon="💬",
    layout="wide",
)

st.title("💬 LLM From Scratch – Token Chat (BPE)")

st.markdown(
    """
Modelo **token-level (BPE)** entrenado desde cero y luego **instruction-tuned**.

Flujo:
1) Si hay **hecho verificado** (FAQ -> FACT), el LLM responde **anclado** al hecho (y si hace falta, cae al hecho exacto).
2) Si no hay hecho, el LLM responde normal.
3) Si la pregunta es **privada** o el modelo se **descarrila / no tiene base**, se rechaza de forma honesta.
"""
)

# ---------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración")

meta_path = st.sidebar.text_input("meta.json", value=DEFAULT_META)
tokenizer_path = st.sidebar.text_input("tokenizer.json", value=DEFAULT_TOK)
ckpt_path = st.sidebar.text_input("checkpoint (.pt)", value=DEFAULT_CKPT)
device = st.sidebar.selectbox(
    "device",
    options=["mps", "cpu"],
    index=0 if DEFAULT_DEVICE == "mps" else 1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Decoding")

max_new_tokens = st.sidebar.slider("max_new_tokens", 10, 200, 60, 5)
min_new_tokens = st.sidebar.slider("min_new_tokens", 1, 40, 2, 1)

stop_at_period = st.sidebar.checkbox("stop_at_period (.)", value=True)
period_id = st.sidebar.number_input("period_id", value=19, step=1)

# Defaults recomendados para mejorar “sin fact”
top_k = st.sidebar.slider("top_k (0 = greedy)", 0, 100, 30, 1)
temperature = st.sidebar.slider("temperature (solo si top_k>0)", 0.0, 1.5, 0.8, 0.05)

repetition_penalty = st.sidebar.slider("repetition_penalty", 1.0, 2.0, 1.0, 0.05)
no_repeat_ngram = st.sidebar.slider("no_repeat_ngram", 0, 6, 0, 1)

st.sidebar.markdown("---")
if st.sidebar.button("🔁 Clear cache (recargar modelo)"):
    clear_cache()
    st.sidebar.success("Cache limpiado. La próxima respuesta recarga assets.")

# ---------------------------------------------------------------------
# Main: test questions
# ---------------------------------------------------------------------
st.markdown("### Pregunta de prueba")

opciones = [
    "¿Cuál es la capital de Costa Rica?",
    "¿Cuál es la capital de Francia?",
    "Los perros pertenecen a qué familia",
    "¿Cuál es el 5 planeta del sistema solar?",
    "¿Cuál es la capital de Argentina?",
    "¿Cuál es mi anime favorito?",
    "¿Cuáles son los tres jefes que he tenido?",
    "Explica la fotosíntesis en una frase.",
    "Explica la relatividad en una frase.",
    "¿Qué es un LLM?",
    "¿Qué es machine learning?",
]

pregunta_base = st.radio("Elige una pregunta:", opciones, index=0)

prompt = st.text_area(
    "Puedes editar la pregunta:",
    value=pregunta_base,
    height=90,
)

# ---------------------------------------------------------------------
# Helpers UI
# ---------------------------------------------------------------------
def _badge(meta: dict) -> str:
    """
    Decide el “estado” usando flags + refuse_reason (para cubrir
    casos donde el modelo ya devolvió “No tengo...” y lo normalizamos).
    """
    reason = (meta.get("refuse_reason") or "").strip()

    if meta.get("used_private_guard") or reason == "private":
        return "🔒 No tengo info personal"

    if meta.get("unknown_guard_triggered") or reason in ("unknown_derail", "unknown_no_knowledge"):
        return "⚠️ No tengo base suficiente para responder con precisión"

    if meta.get("used_fact"):
        if meta.get("fact_validation_fallback"):
            return "✅ Hecho verificado (fallback exacto)"
        return "✅ Hecho verificado + respuesta generada"

    return "🤖 Respuesta generada por el modelo"


def _status_help(meta: dict) -> str:
    reason = (meta.get("refuse_reason") or "").strip()
    if meta.get("used_private_guard") or reason == "private":
        return "La pregunta pide información personal del usuario, y este sistema no la tiene."
    if meta.get("unknown_guard_triggered") or reason == "unknown_derail":
        return "El modelo se descarriló en un tema general; se bloquea para evitar alucinaciones."
    if reason == "unknown_no_knowledge":
        return "El modelo no tiene base suficiente para responder eso con precisión."
    if meta.get("used_fact") and meta.get("fact_validation_fallback"):
        return "Se devolvió el hecho exacto para asegurar 0 alucinación."
    if meta.get("used_fact"):
        return "Respuesta generada, anclada a un hecho verificado."
    return "Respuesta generada por el modelo."


# ---------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------
if st.button("Generar respuesta"):
    if not prompt.strip():
        st.warning("Por favor escribe una pregunta.")
    else:
        with st.spinner("Generando..."):
            ans, meta = answer_with_meta(
                prompt,
                meta_path=meta_path,
                ckpt_path=ckpt_path,
                tokenizer_path=tokenizer_path,
                device=device,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                stop_at_period=1 if stop_at_period else 0,
                period_id=int(period_id),
                top_k=int(top_k),
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram=int(no_repeat_ngram),
            )

        st.markdown("### 🟢 Respuesta")
        st.write(ans)

        st.markdown("### Estado")
        st.success(_badge(meta))
        st.caption(_status_help(meta))

        st.caption(
            f"took_ms={meta.get('took_ms', 0.0)} | device={device} | "
            f"top_k={top_k} | temp={temperature} | max_new_tokens={max_new_tokens}"
        )

        # Mostrar FACT si existe (útil para demo y verificación)
        if meta.get("used_fact") and meta.get("fact"):
            st.markdown("### Hecho verificado (ancla)")
            st.code(meta["fact"])

        with st.expander("Debug (meta)"):
            st.write(meta)

else:
    st.info("Elige una pregunta y pulsa **Generar respuesta**.")