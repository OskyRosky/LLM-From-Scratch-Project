# src/inference/faq_fallback.py

from difflib import SequenceMatcher

FAQ = {
    "perro_canino": "Sí, los perros son caninos. Pertenecen a la familia de los cánidos.",
    "gato_felino": "Sí, los gatos son felinos. Pertenecen a la familia de los félidos.",
    "capital_cr": "La capital de Costa Rica es San José.",
}

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def faq_match(prompt: str):
    p = prompt.lower().strip()

    # -----------------------------
    # 1. PREGUNTA DE PERROS
    # -----------------------------
    perro_keywords = ["perro", "perros", "canino", "caninos", "cánido", "cánidos"]
    if any(k in p for k in perro_keywords) and similar(p, "los perros son caninos?") > 0.45:
        return FAQ["perro_canino"]

    # -----------------------------
    # 2. PREGUNTA DE GATOS
    # -----------------------------
    gato_keywords = ["gato", "gatos", "felino", "felinos", "félido", "félidos"]
    if any(k in p for k in gato_keywords) and similar(p, "los gatos son felinos?") > 0.45:
        return FAQ["gato_felino"]

    # -----------------------------
    # 3. CAPITAL DE COSTA RICA
    # -----------------------------
    if "capital" in p and ("costa rica" in p or "costarricense" in p):
        return FAQ["capital_cr"]

    return None