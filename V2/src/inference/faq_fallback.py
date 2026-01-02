from difflib import SequenceMatcher

# FACTS: devolver SIEMPRE una frase completa (hecho verificado)
FAQ_FACTS = {
    "perro_canino": "Los perros pertenecen a la familia de los cánidos.",
    "gato_felino": "Los gatos pertenecen a la familia de los félidos.",
    "capital_cr": "La capital de Costa Rica es San José.",
    "capital_fr": "La capital de Francia es París.",
}

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def faq_fact(prompt: str) -> str:
    """
    Devuelve un FACT (string) si hay match.
    Si no hay match, devuelve string vacío "" (nada de None).
    """
    p = prompt.lower().strip()

    # 1) Perros / cánidos
    perro_keywords = ["perro", "perros", "canino", "caninos", "cánido", "cánidos"]
    if any(k in p for k in perro_keywords):
        if similar(p, "los perros son caninos?") > 0.45 or "familia" in p:
            return FAQ_FACTS["perro_canino"]

    # 2) Gatos / félidos
    gato_keywords = ["gato", "gatos", "felino", "felinos", "félido", "félidos"]
    if any(k in p for k in gato_keywords):
        if similar(p, "los gatos son felinos?") > 0.45 or "familia" in p:
            return FAQ_FACTS["gato_felino"]

    # 3) Capital Costa Rica
    if "capital" in p and ("costa rica" in p or "costarricense" in p):
        return FAQ_FACTS["capital_cr"]

    # 4) Capital Francia
    if "capital" in p and ("francia" in p or "francesa" in p):
        return FAQ_FACTS["capital_fr"]

    return ""
