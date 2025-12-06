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
    OJO: aquí pasamos device_str como string, no un torch.device,
    y pasamos el DIRECTORIO donde está el checkpoint.
    """
    # 👈 IMPORTANTE: aquí va el directorio, no el archivo .pt
    ckpt_dir = "models/checkpoints_oscar_long"

    st.write(f"[DEBUG] Cargando modelo en dispositivo: {device_str}")
    # load_instructions_model espera (ckpt_dir: str, device_str: str)
    bundle = load_instructions_model(ckpt_dir, device_str=device_str)
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
    value=0.7,
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
Modelo **caracter a caracter** entrenado desde cero sobre `oscar_corpus.txt`  
y luego *instruction-tuned* con un conjunto mínimo de pares (instrucción → respuesta).

⚠️ **Este modelo es muy pequeño y educativo**, no esperes respuestas tipo ChatGPT.
"""
)

st.markdown("---")

prompt = st.text_area(
    "Escribe una instrucción o pregunta:",
    value="Un perro es un canino?",
    height=100,
)

if st.button("Generar respuesta"):
    if not prompt.strip():
        st.warning("Por favor escribe una instrucción o pregunta.")
    else:
        with st.spinner("Cargando modelo (si es la primera vez) y generando respuesta..."):
            # Aquí NO pasamos torch.device, solo el string "mps"
            bundle = get_model_bundle(device_str="mps")

            answer = generate_answer(
                bundle=bundle,
                user_prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        st.markdown("### Respuesta del modelo")
        st.write(answer)

        st.markdown("---")
        st.markdown(
            "_Recuerda: este es un modelo tiny para fines educativos; "
            "las respuestas pueden ser incoherentes._"
        )
else:
    st.info("Escribe una pregunta y pulsa **Generar respuesta**.")