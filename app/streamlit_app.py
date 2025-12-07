# app/streamlit_app.py

import os
import sys

import streamlit as st

# ---------------------------------------------------------------------
# Asegurar que podamos hacer "import src..."
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.inference.instructions_chat import (  # noqa: E402
    load_instructions_model,
    generate_answer,
    InstructionsModelBundle,
)
from src.inference.faq_fallback import faq_match  # noqa: E402


# ---------------------------------------------------------------------
# Configuración básica de la página
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LLM From Scratch - Instruction Chat",
    page_icon="💬",
    layout="wide",
)


# ---------------------------------------------------------------------
# Función cacheada para cargar el modelo UNA sola vez
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo (solo la primera vez)...")
def get_model_bundle(device_str: str = "mps") -> InstructionsModelBundle:
    """
    Carga el modelo de instrucciones y lo cachea.

    Usamos la interfaz:
        load_instructions_model(ckpt_dir=..., device_str=...)
    """
    ckpt_dir = "models/checkpoints_oscar_long"

    st.write(f"[DEBUG] Cargando modelo en dispositivo: {device_str}")
    bundle = load_instructions_model(
        ckpt_dir=ckpt_dir,
        device_str=device_str,
    )
    return bundle


# ---------------------------------------------------------------------
# Sidebar: parámetros de generación
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros de generación")

max_new_tokens = st.sidebar.slider(
    "max_new_tokens",
    min_value=10,
    max_value=200,
    value=80,
    step=5,
)

temperature = st.sidebar.slider(
    "temperature",
    min_value=0.0,
    max_value=1.5,
    value=0.0,  # igual que en eval_instructions_mini
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Backend: `gpt_char_instructions.pt` en "
    "`models/checkpoints_oscar_long/`."
)


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------
st.title("💬 LLM From Scratch – Instruction Chat (tiny)")

st.markdown(
    """
Modelo **carácter a carácter** entrenado desde cero sobre `oscar_corpus.txt`  
y luego *instruction-tuned* con un conjunto mínimo de pares (instrucción → respuesta).

⚠️ **Este modelo es muy pequeño y educativo**, no esperes respuestas tipo ChatGPT.
"""
)

st.markdown("---")

st.markdown("### Pregunta de prueba")

opciones = [
    "Los perros son caninos?",
    "Los gatos son felinos?",
    "Cuál es la capital de Costa Rica?",
]

pregunta_base = st.radio(
    "Elige una de las preguntas de test:",
    opciones,
    index=0,
)

prompt = st.text_area(
    "Puedes ajustar la pregunta si quieres:",
    value=pregunta_base,
    height=100,
)

if st.button("Generar respuesta"):
    if not prompt.strip():
        st.warning("Por favor escribe una instrucción o pregunta.")
    else:
        with st.spinner("Generando respuesta..."):
            # 1) Intentar primero el fallback perfecto (FAQ)
            faq_answer = faq_match(prompt)

            if faq_answer is not None:
                # Respuesta perfecta, sin usar el modelo
                answer_text = faq_answer
                full_text = f"(fallback) {faq_answer}"
                st.success("✔ Respuesta obtenida vía fallback (respuesta perfecta)")
            else:
                # 2) Si no coincide, usamos el modelo tiny
                bundle = get_model_bundle(device_str="mps")

                answer_text, full_text = generate_answer(
                    bundle=bundle,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

        st.markdown("### 🟢 Respuesta procesada (solo después de `<resp>`)")
        st.write(answer_text)

        st.markdown("### 📜 Texto completo generado")
        st.code(repr(full_text), language="python")

        st.markdown("---")
        st.markdown(
            "_Recuerda: este es un modelo tiny para fines educativos; "
            "las respuestas pueden ser incoherentes._"
        )
else:
    st.info("Elige una pregunta y pulsa **Generar respuesta**.")