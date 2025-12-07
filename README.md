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

Before training anything, an LLM project must be built on solid infrastructure. Large models are not created by a single script; they emerge from a reproducible environment, a clear project layout, and a modular codebase where each component can evolve independently. The goal of this stage is to establish the foundation on which every later experiment will depend.

The first step is defining the structure of the repository itself. A well-organised project separates raw data from processed datasets, model checkpoints from source code, training scripts from inference utilities, and experiments from the application layer. This avoids chaos later, when multiple models, datasets and fine-tuning runs coexist. In this project, the repository is intentionally structured to mirror the workflow of a modern LLM engineering team:

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

# 2. Texto y tokenización (Working with Text Data).

# 3. Coding Attention Mechanisms.

# 4. GPT Model.


# 5. Pretraining 

# 6. Finetuning for Text Classification (hidden states + backbone preentrenado real)

# 7. Finetuning to Follow Instructions - instruction tuning.

# 8. UI: App Streamlit

# 9. Improvements.

# 10. Examples
 
