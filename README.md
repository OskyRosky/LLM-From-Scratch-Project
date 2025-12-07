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

Before building any language model—from the smallest educational prototype to a large-scale LLM—the first challenge is always the same: how to represent text in a way a model can understand.

Version 1 of this project intentionally uses a character-level tokenizer. This is not how modern LLMs operate, but it is the simplest and most transparent way to understand the foundations of autoregressive modeling.

## 2.1 Why Character-Level Tokenization?

For an instructional project, character-level tokenization offers several advantages:

	•	There is no vocabulary to maintain; every visible character becomes a token.
	•	The model never encounters unknown words; it learns to construct them from letters.
	•	The tokenizer is trivial to implement and completely transparent.
	•	It allows you to focus on the mechanics of transformers without relying on external tooling.

Of course, character modeling has limitations: longer sequences, slower convergence, and weaker semantic structure. But it is ideal for understanding how a GPT learns from raw text.

## 2.2 Building the Vocabulary

During preprocessing, the system scans the corpus and extracts every unique character.
Each character is mapped to an integer, forming the vocabulary the model uses.

Alongside the natural characters, several special tokens are added:

	•	a padding token for aligning sequences,
	•	a token for unknown characters,
	•	markers used during instruction tuning (e.g., one to start the instruction and another to start the response).

These special markers allow the same GPT backbone to support pretraining, classification, and instruction-following tasks using a unified token space.

## 2.3 Turning Text Into Model-Ready Sequences

Once the vocabulary exists, every dataset follows the same pipeline:
	1.	Text is converted into a sequence of numerical IDs.
	2.	Sequences are padded or truncated to a fixed maximum length.
	3.	Batches are constructed so the model always receives uniform tensors.

This ensures stability during training and mirrors the requirements of real transformer implementations.


## 2.4 Pretraining Dataset

The backbone GPT is trained using the classic next-token prediction task.
For character-level models, this means predicting the next character in the sequence.

This stage teaches the model:
	•	how text flows,
	•	how characters relate to each other,
	•	how structure emerges from sequential prediction.

Even a tiny model learns patterns such as word boundaries, punctuation, and common character combinations.

## 2.5 Instruction Dataset

For instruction tuning, the data uses simple prompt-response pairs.
Each one is rewritten internally into a structured format that clearly separates:
	•	the instruction,
	•	the model’s expected response.

This structure helps the small GPT backbone transition from “predict the next character” to “follow a command,” even with minimal data.


## 2.6 Summary

This stage lays the groundwork for everything that follows:
	•	A clean and interpretable vocabulary.
	•	A transparent tokenization strategy.
	•	Unified datasets for pretraining, classification, and instruction tuning.
	•	A clear structure for instruction-style interactions.

Even though later versions of the project will shift toward modern tokenization approaches (BPE or WordPiece), V1 deliberately stays simple so every step is visible and understandable.


# 3. Coding attention mechanisms.

Attention is the central innovation that made modern Large Language Models possible.
In this project, the goal was not just to use attention, but to understand it deeply by implementing it from scratch.

Our Version 1 model uses a minimal and educational version of the attention modules found in a GPT architecture. This helps reveal how transformers truly operate behind the scenes.

## 3.1  Motivation: Why Attention?

Traditional sequence models (RNNs, LSTMs, GRUs) struggle with long-range dependencies.
Transformers replaced recurrence with a mechanism that:
	•	looks at all positions in a sequence simultaneously,
	•	weighs how important each token is relative to all others,
	•	and does this efficiently using matrix operations.

This mechanism—self-attention—is what allows models like GPT to generate coherent text, learn context, and reason over long spans.

## 3.2 The Core Idea: Queries, Keys, and Values.

Every position in a sequence is transformed into three vectors:
	•	Query (Q) – represents the question a token is asking.
	•	Key (K) – represents how relevant a token is to others.
	•	Value (V) – carries the actual information to be aggregated.

Self-attention computes similarity between each query and key pair.
These similarities determine how much each value contributes to the final representation.

Even at the character level, this mechanism enables the model to:
	•	understand repeated patterns,
	•	capture local and global structure,
	•	build internal representations far richer than raw tokens.

## 3.3 Multi-Head Attention: Learning Multiple Patterns at Once

A single attention head learns only one type of relationship.
GPT uses multiple heads, each focusing on different aspects of the sequence.

In this project:
	•	the number of heads is small,
	•	each head has reduced dimensionality,
	•	and the design follows the original GPT architecture.

Each head analyzes the input differently, and their outputs are combined to produce a more expressive representation.

Even a tiny model benefits from this diversity of viewpoints.

## 3.4 Causal Masking: Enforcing Autoregressive Behavior

GPT models predict the next token.
To ensure the model cannot cheat by looking ahead, the attention mechanism applies a causal mask that blocks future positions.

This ensures the model behaves like a real autoregressive generator:
	•	token t can only attend to positions ≤ t,
	•	future characters are never visible during training.

This simple constraint is what makes generation deterministic and left-to-right.

## 3.5 Residual Connections and Layer Normalization

Each attention block is wrapped inside two critical architectural components:
	•	Residual connections, which stabilize learning and allow gradients to flow easily.
	•	Layer normalization, which ensures numerical stability and consistent scaling.

While Version 1 keeps these modules minimal, they replicate the functional structure of production LLM architectures.

## 3.6 Feedforward Network (MLP Block)

Following attention, the transformer includes a positionwise feed-forward network.

In this implementation, it consists of:
	•	a linear expansion,
	•	a non-linear activation,
	•	a linear projection back to the model dimension.

This MLP acts like a local reasoning module, helping the model interpret contextual information and refine its internal representation.
 
## 3.7 Putting It All Together

 A full transformer block in this project includes:
	1.	Multi-head self-attention (with causal masking)
	2.	Residual and normalization layers
	3.	Feedforward MLP
	4.	Another residual and normalization pass

Stacking these blocks produces the GPT backbone used across:
	•	Pretraining
	•	Classification finetuning
	•	Instruction finetuning
	•	Streamlit inference

Even though this model is tiny and character-based, the underlying architecture mirrors that of GPT-2 and GPT-Neo, simply scaled down for educational purposes.

## 3.8  Why Build Attention by Hand?

Implementing these modules manually:

	•	demystifies how LLMs process information,
	•	provides intuition for how patterns emerge,
	•	prepares you for more advanced architectures (GPT-2/3/NeoX),
	•	is the foundation for Version 2, where we transition to word-level or BPE tokenization.

This section is the “engine room” of the project—understanding it means understanding how every modern transformer-based model works internally.

# 4. GPT Model.


# 5. Pretraining 

# 6. Finetuning for Text Classification (hidden states + backbone preentrenado real)

# 7. Finetuning to Follow Instructions - instruction tuning.

# 8. UI: App Streamlit

# 9. Improvements.

# 10. Examples
 
