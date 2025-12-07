# LLM From Scratch

 ![class](/ima/ima1.png)

---------------------------------------------

**Repository summary**

1.  **Intro** 🧳

2.  **Tech Stack** 🤖


3.  **Features** 🤳🏽


4.  **Process** 👣


5.  **Learning** 💡


6.  **Improvement** 🔩


7.  **Running the Project** ⚙️


8 .  **More** 🙌🏽

For collaboration, discussion, or improvements:

•	GitHub Issues: for bugs or feature requests.

•	Pull Requests: for contributions or new examples.

•	Contact: open an issue or connect via LinkedIn / email (author info in profile).

If this project helps you learn or build better models, consider starring ⭐ the repository —
it’s the simplest way to support continued open knowledge sharing.


---------------------------------------------

# :computer: LLM From Scratch  :computer:

---------------------------------------------

# 0. Why build an LLM from Scratch ? 

Building a Large Language Model from scratch is not about competing with industrial-scale systems like GPT-4 or Claude. Instead, it is about understanding—truly understanding—how modern language models work at the lowest level. When you assemble every component by hand, from tokenization to attention mechanisms to pretraining loops, the architecture stops being a black box and becomes an engineered system whose internal logic you can navigate with confidence.

Most practitioners work with high-level libraries that abstract away the essential mechanics of LLMs. That abstraction is convenient, but it hides the fundamental transformations that give these models their reasoning capabilities: how text becomes numbers, how attention selects and weights context, how residual blocks accumulate knowledge, or how predictions are optimized over billions of tokens. By re-implementing these ideas, you gain a concrete mental model of what is happening at every step of the pipeline.

 ![class](/ima/ima2.png)

This project exists for that purpose. It aims to reconstruct the core pieces of a GPT-style model with clarity and intention, taking nothing for granted. Every function, every shape transformation, and every training step is exposed. You are able to see how a single character or token propagates through embeddings, how attention scores are computed and normalized, how the model predicts the next element in a sequence, and why gradient updates slowly sculpt its internal representation of language.

Once you build an LLM from scratch, you earn three key advantages.
First, you gain the ability to debug and modify any part of the architecture without relying on hidden implementations.
Second, you develop intuition for why certain design decisions—such as positional encodings or normalization layers—matter for stability and performance.
And third, you become independent: capable of extending, compressing, adapting or even inventing new architectures instead of depending solely on external frameworks.

This repository is the outcome of that philosophy. It walks through the entire lifecycle of a modern language model—from raw text to a trained model capable of following simple instructions and interacting through a custom interface—while keeping the system transparent and accessible. The result is not a production-ready LLM, but rather a rigorous educational framework: a place where the fundamentals can be learned, inspected, improved and expanded.


# 1. Basic Infraestructure.

Building an LLM from scratch requires a clean, modular, and scalable codebase. The goal is not only to train a model, but to create an environment where experimentation is safe, reproducible, and easy to extend. This project follows a structure inspired by modern ML engineering practices (NLP labs, production AI teams, and open-source frameworks like nanoGPT, PyTorch Lightning, and HuggingFace).

At its core, the infrastructure is designed around four principles:
	1.	Separation of concerns — preprocessing logic, model code, training loops, evaluation utilities, and UI demos all live in dedicated modules.
	2.	Reproducibility — every step (tokenization, sampling, training configuration) is deterministic and version-controlled.
	3.	Incremental extensibility — the project supports a natural evolution:
character-level GPT → instruction tuning → UI deployment → word-level GPT (future V2).
	4.	Ease of experimentation — everything can be run via CLI tools or notebooks, without rewriting boilerplate.

The base directory structure is:

```Python
LLM-From-Scratch-Project/
│
├── data/                 # raw and processed datasets
├── models/               # pretrained weights and fine-tuned checkpoints
├── src/                  # model architecture, training pipeline, utilities
├── app/                  # Streamlit UI and demo interfaces
├── notebooks/            # exploratory analysis, prototyping
└── scripts/              # CLI tools for training, preprocessing, evaluation
```

Each directory exists for a reason.
The src/model module isolates the GPT architecture so it can be reused across pretraining and fine-tuning tasks. The src/data module handles tokenization, batching and sequence preparation—the backbone of any efficient training loop. The CLI tools abstract training into reproducible commands, while the app/ folder provides a simple but complete interface to interact with the model.

Another essential component of the infrastructure is the virtual environment. By isolating dependencies inside a .venv, the project becomes portable and behaves identically on any machine. This also prevents version conflicts, especially with libraries such as PyTorch, which evolve quickly and require careful compatibility control. Installing dependencies through requirements.txt completes this reproducibility cycle.

GPU detection and device routing also form part of the infrastructure layer. The model must gracefully select between CPU, CUDA or Apple Silicon’s MPS backend depending on what is available. This ensures that both training and inference remain accessible, whether on a high-end workstation or a simple laptop.

At this stage, nothing “intelligent” has happened yet—but everything necessary for intelligence to emerge has been prepared. The infrastructure defines how data flows, how models are instantiated, how experiments are executed, and how results are consumed. Without this foundation, even a well-designed architecture would collapse under complexity. With it, the subsequent stages—tokenization, attention mechanisms, GPT blocks, pretraining, and fine-tuning—become coherent parts of a larger, well-engineered system.

Environment Setup

The project is fully isolated inside a Python virtual environment:

```Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This ensures:
	•	consistent dependencies,
	•	clean execution across machines,
	•	and easy transition into Docker environments (used later for deployment).

Core Components

src/model/

Contains the complete GPT implementation:
	•	multi-head self-attention
	•	feedforward blocks
	•	residual connections
	•	positional embeddings
	•	autoregressive masking
	•	generation utilities

The entire model is written from first principles in pure PyTorch, making it fully transparent and easy to extend.

src/data/

Includes:
	•	the character-level tokenizer
	•	dataset loaders (pretraining, classification, and instruction tuning)
	•	deterministic batching utilities

This abstraction keeps data mechanics independent from model mechanics.

src/finetuning/

Implements:
	•	masked loss computation
	•	instruction-tuning objective
	•	classification head training
	•	freezing/unfreezing strategies

Each finetuning stage reuses the pretrained GPT backbone.

src/cli/

Command-line tools that orchestrate training:

```Python
python -m src.cli.pretrain_char
python -m src.cli.finetune_classification
python -m src.cli.finetune_instructions
```

This modularity allows you to re-run individual stages without recomputing everything.

app/

Houses the Streamlit interface that lets users interact with the tiny LLM:
	•	character-level generation
	•	instruction-based Q&A
	•	FAQ fallback mechanism
	•	model selection (future V2)

This folder connects research to a real UI.

# 2. Working with Text Data: Tokenization & Dataset Preparation.




# 3. Coding Attention Mechanisms.

# 4. GPT Model.


# 5. Pretraining 

# 6. Finetuning for Text Classification (hidden states + backbone preentrenado real)

# 7. Finetuning to Follow Instructions - instruction tuning.

# 8. UI: App Streamlit

# 9. Improvements.

# 10. Examples
 
