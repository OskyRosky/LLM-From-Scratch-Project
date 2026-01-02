# **LLM From Scratch (V2) — Token Chat (BPE)**

A minimal **token-level (BPE) language model** trained from scratch and then **instruction-tuned**, exposed through a **Streamlit chat demo**.

This repository focuses on:
- Proving the **end-to-end pipeline works** (tokenization → model → inference).
- Showing a clear inference flow with **guards** and **traceable metadata**.
- Keeping the app small, reproducible, and easy to run locally or via Docker.

---

## **What’s inside (V2)**

### **Core components**
- **Tokenizer (BPE)**: custom tokenizer and meta assets.
- **GPT-style model**: lightweight architecture for token generation.
- **Inference layer**: `answer_with_meta()` returns the answer plus metadata:
  - `used_fact`, `fact_validation_fallback`
  - `used_private_guard`
  - `unknown_guard_triggered`
  - `refuse_reason`
  - `took_ms`

### **Safety & correctness**
This project intentionally prioritizes **honest behavior** over “guessing”:
- **Private guard**: blocks personal questions (e.g., “my favorite anime”, “my bosses”).
- **Unknown / derail guard**: blocks topics where the tiny model tends to derail.
- **Fact anchoring**: if a verified fact exists, the model answers anchored to it, with validation + fallback to the exact fact if needed.

---

## **Project structure**
Typical relevant paths:

- `app/streamlit_app.py` — Streamlit UI
- `src/infer/answer.py` — inference entrypoint (`answer_with_meta`)
- `src/inference/faq_fallback.py` — verified facts lookup (`faq_fact`)
- `models/`
  - `tokenizers/.../tokenizer.json`
  - `tokenized/.../meta.json`
  - `checkpoints/.../*.pt`
- `scripts/smoke_suite.py` — quick regression suite (10 questions)

---

## **Quickstart (local)**

### 1) Create and activate venv
```bash


cd V2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```









https://poloclub.github.io/transformer-explainer/

https://github.com/poloclub/transformer-explainer

https://github.com/rasbt/LLMs-from-scratch

https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up

https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11


