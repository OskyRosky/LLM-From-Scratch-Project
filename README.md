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

The GPT model implemented in this project follows the original decoder-only Transformer design introduced by OpenAI. Although our Version 1 model is intentionally small and character-based, the underlying architecture mirrors the essential structure of real GPT systems.

This section explains how the full model fits together and why each component matters.

## 4.1 A Decoder-Only Transformer

GPT belongs to the family of autoregressive language models:
	•	It receives a sequence of tokens.
	•	It predicts the next token.
	•	It repeats this process step by step.

To support this behavior, the model uses multiple stacked Transformer blocks, each containing:
	1.	Masked Multi-Head Self-Attention
	2.	Feedforward (MLP) sublayer
	3.	Residual connections
	4.	Layer normalization

Even a tiny model benefits from these architectural principles.
They provide stability, expressive power, and the capacity to learn long-range structure.

## 4.2 Token & Position Embeddings

Transformers have no inherent sense of order, so we give them two forms of input representation:

Token embeddings

Every character in the vocabulary is converted into a dense vector.
In Version 1, the model learns to express relationships between characters—punctuation, spacing, frequent subpatterns, and more.

Positional embeddings

Since characters appear in sequences, positional embeddings encode the position of each token.
GPT learns how the meaning of a token depends on where it occurs.

Together, token + position embeddings form the input to the first attention block.

## 4.3 Stacked Transformer Blocks

Each GPT block applies the full Transformer logic:
	•	The attention sublayer determines which previous characters are relevant.
	•	The MLP sublayer interprets the aggregated information.
	•	Residual pathways allow information to flow cleanly through the stack.

Increasing the number of blocks increases model depth and reinforces the ability to model patterns.

Although our model is intentionally small, the design aligns with real GPT architectures—and extending it (more layers, wider embeddings, more heads) becomes straightforward.

## 4.4 Causal Attention: Enforcing Autoregression

GPT is designed to generate text one token at a time.

To enforce this, the attention mechanism includes a strict lower-triangular mask:
	•	A token can attend only to earlier positions.
	•	The model never sees future tokens during training.
	•	Predictions remain consistent with real-world generation.

This property distinguishes GPT from bidirectional models like BERT and is fundamental to sequence generation.

## 4.5 Language Modeling Head (Output Layer)

The final representation of each position passes through a linear projection that maps hidden states to the vocabulary size.

The model outputs a probability distribution over all tokens.
During training, the objective is simple:

Predict the next token in the sequence as accurately as possible.

This is the foundation for:
	•	pretraining on large corpora,
	•	finetuning for classification,
	•	finetuning for instruction following.

Everything else—the emergent reasoning, structure, coherence—arises from maximizing this next-token objective across many examples.

## 4.6 Why Build GPT From Scratch?

Manually implementing GPT demystifies how modern LLMs operate:
	•	You can inspect every tensor and operation.
	•	You understand what is learned and why.
	•	You gain full control over architecture modifications.
	•	You build intuition for scaling laws, training dynamics, and efficiency constraints.

This foundation becomes crucial when transitioning to Version 2, where we incorporate word/BPE tokenization and scale the architecture to handle richer language patterns.

## 4.7 A Minimal but Complete GPT Engine

While Version 1 is small enough to run on a laptop, it includes all essential architectural components:

	•	Learnable embeddings
	•	Multi-Head Self-Attention
	•	Causal masking
	•	Transformer blocks
	•	Autoregressive decoding
	•	Full training loop for pretraining and finetuning

This provides a clean, understandable base that is both educational and extensible.

In short: Version 1 is a real GPT, just scaled down to reveal how everything works.

# 5. Pretraining 

Pretraining is the foundational stage of any GPT-like model. It is where the model learns the fundamental structure of language by repeatedly predicting the next token in a sequence. Even in our small character-level version, this process is what gives the model its initial “knowledge” before any specialized finetuning.

## 5.1 Objective: Next-Token Prediction

The core idea is simple:
	•	Take a long sequence of text.
	•	Feed it into the model.
	•	Ask the model to predict the next character at every position.

Every prediction produces a loss value.
By minimizing that loss across millions of predictions, the model gradually discovers:
	•	which characters or patterns commonly follow others,
	•	how sentences and structures flow,
	•	how context influences meaning.

Even without explicit linguistic rules, the model learns statistical regularities purely from exposure.

This mirrors how real GPT models are trained at scale—just with much smaller datasets and capacity.


## 5.2 Training on the OSCAR Corpus

To pretrain the tiny model, we use a subset of the OSCAR dataset—a large multilingual web corpus.
From it, we derive a simplified Spanish text file (oscar_corpus.txt) suitable for a character-level language model.

Key preprocessing steps:
	•	cleaning non-printable characters
	•	lowercasing or normalization
	•	removing extremely long lines
	•	preparing a contiguous training stream

The model does not learn “facts” at this stage; it only learns patterns of text.

This is intentional: pretraining provides general linguistic ability, while finetuning injects task-specific knowledge.

## 5.3 Training Dynamics

During pretraining:
	•	The model reads sequences of fixed length.
	•	It shifts the same sequence by one character to build the “labels.”
	•	The loss function compares the predicted character to the actual next character.
	•	Gradients update all model parameters, including attention, embeddings and MLP layers.

As training progresses, the model develops:
	•	stronger character-level coherence,
	•	better attention focusing on relevant context,
	•	improved ability to complete words or phrases.

Although Version 1 is intentionally tiny, it still demonstrates the essential learning behavior that powers large-scale LLMs.

## 5.4 What Pretraining Enables

Once the model is pretrained, several capabilities emerge:

Generalization

The model can continue text sequences with plausible structure.

Knowledge Transfer

The pretrained backbone serves as a foundation for downstream tasks.
Instead of starting from random weights, finetuning begins from a partially trained model that already “understands” syntax and text flow.

Instruction Tuning

In later stages, we teach the model to follow commands using curated prompt → response pairs.
Without pretraining, this would be impossible.

## 5.5 Why Pretraining Matters in a From-Scratch Project

Building this stage manually contains immense learning value:
	•	You understand the full data pipeline and why sequence preparation matters.
	•	You see how a Transformer learns from raw text rather than predefined rules.
	•	You experience the computational constraints that guide architecture and dataset choices.
	•	You gain intuition about how scaling up affects behavior and performance.

Pretraining is not just the beginning of the project—it is the foundation on which every later improvement is built.

## 5.6 Limitations of a Tiny Character-Level Pretrained Model

It is important to acknowledge the constraints:
	•	Character-level modeling is expressive but inefficient.
	•	The model capacity is extremely limited.
	•	Learning semantic knowledge is nearly impossible at this scale.
	•	Sequence length boundaries restrict long-context behaviors.

These limitations motivate the shift to Version 2, where we adopt subword tokenization, longer context windows, and larger model capacity.

# 6. Finetuning for Text Classification (hidden states + backbone preentrenado real)

Once the base model has been pretrained on raw text, it becomes a reusable backbone capable of understanding linguistic patterns. Finetuning leverages this backbone to solve downstream supervised tasks—in our case, a simple text classification problem designed to show how hidden states can be adapted for real tasks.

This step demonstrates how pretrained representations can be repurposed efficiently, even when the underlying model is tiny.

##  6.1 Why Classification Finetuning Matters

Classification is one of the most common applications of language models:
	•	sentiment analysis
	•	spam detection
	•	topic identification
	•	intent recognition
	•	document routing

Even large-scale models often rely on similar techniques internally.

In our project, classification finetuning serves two purposes:
	1.	Educational — it teaches how to adapt a pretrained transformer to a supervised objective.
	2.	Practical — it validates that the model’s learned representations contain usable information beyond simple next-token prediction.

This stage confirms that pretraining truly creates reusable knowledge.

## 6.2 Using the Pretrained Backbone

Instead of training the classifier from scratch, we reuse:
	•	the tokenizer
	•	the embedding layer
	•	the multi-head attention blocks
	•	the feed-forward layers
	•	positional encodings
	•	the overall GPT architecture

All parameters learned during pretraining remain intact.

A lightweight classification head is then added on top of the transformer’s hidden states, usually:
	•	a small linear layer,
	•	optionally followed by dropout or normalization,
	•	projecting to the number of target classes.

This approach mirrors modern NLP pipelines such as BERT or GPT-style adapters.

## 6.3 Training the Classifier

During finetuning:
	1.	Text samples are fed into the model using the same tokenization as pretraining.
	2.	The transformer produces contextual embeddings for every token.
	3.	A pooled representation (often the final token, or an average) is passed to the classification head.
	4.	The model is optimized using supervised labels and cross-entropy loss.
	5.	Only the classification head may be trained, or optionally the entire backbone may be unfrozen.

Both strategies are valuable:
	•	Frozen backbone → fast training, stable results.
	•	Full finetuning → higher accuracy but more computational cost.

In our educational setup, either approach is possible.

## 6.4 What the Model Learns

Although the model is small, finetuning enables it to develop:
	•	sensitivity to patterns in the training labels,
	•	ability to classify short inputs reliably,
	•	task-specific decision boundaries based on learned embeddings.

It is remarkable how much performance improvement can arise from a pretrained backbone, even at this tiny scale.

This mirrors the behavior of large models:
pretraining provides general linguistic knowledge, finetuning specializes it.

## 6.5 Interpreting Results

Given the intentionally limited size of our model and dataset, the goal is not cutting-edge accuracy.
Instead, we evaluate:
	•	whether the loss decreases consistently,
	•	whether the classifier separates classes with minimal overfitting,
	•	whether the model relies on meaningful patterns instead of memorization.

A successful finetuning experiment demonstrates that:
	1.	the backbone representations are useful,
	2.	the architecture supports downstream tasks,
	3.	pretraining meaningfully accelerates supervised learning.

## 6.6 Why This Stage Is Crucial for the Full Project

This finetuning step bridges the gap between:
	•	generic language modeling (pretraining), and
	•	task-driven outputs (classification, instruction following).

It also prepares the intuition required for the next stage—instruction tuning, where the model transitions from classification to behaving like a conversational assistant.

# 7. Finetuning to Follow Instructions (Instruction Tuning).

After pretraining the model on raw text and validating its backbone through classification, the next major evolutionary step is teaching the model to respond to natural-language instructions.
This is the foundation of modern conversational systems—ChatGPT, Claude, Gemini, and virtually all instruction-following LLMs rely on this technique.

Instruction tuning transforms a generic language model into an assistant.

## 7.1 What Instruction Tuning Means

A pretrained model knows language patterns, grammar, and relationships between words.
But it does not inherently understand that:

	•	a question expects an answer,
	•	a request expects execution,
	•	a command expects compliance,
	•	or that a prompt defines a task.

Instruction tuning explicitly teaches the model these behaviors.

It does so by exposing it to many examples of the form:

Instruction → Ideal Response

This creates alignment between what the user wants and what the model outputs.

Even small models—like ours—benefit significantly from this approach.

## 7.2 Why We Use a Custom Format (<instr> / <resp>)

Large LLMs use sophisticated prompting conventions, but for a character-level educational model, we need something simple, reproducible, and explicit.

This teaches the model a deterministic conversational protocol:

	1.	When it sees <instr>, it expects a question or command.
	2.	When it sees <resp>, it should begin generating an answer.
	3.	It learns to stop hallucinating metadata and focus on the content.

This structured prompting is critical, especially for tiny models.

## 7.3 Building the Instruction Dataset

Our dataset contains multiple natural variants of the same three facts:

	•	Dogs are canines.
	•	Cats are felines.
	•	San José is the capital of Costa Rica.

Although small, the dataset covers:

	•	paraphrasing,
	•	variations in phrasing,
	•	alternative sentence structures,
	•	different levels of formality.

This exposes the model to the idea that different inputs can map to the same correct answer.

This concept—semantic equivalence—is central to everything from chatbots to retrieval-augmented models.

## 7.4 Finetuning the Model with Paired Examples

During instruction tuning:

	1.	The pretrained backbone processes the full <instr> … <resp> sequence.
	2.	The model is trained to predict the answer text character-by-character.
	3.	It learns to associate instructions with appropriate outputs.
	4.	The optimization objective remains next-token prediction, but conditioned on structured inputs.

This teaches the model not only what to answer, but when and why.

Instruction tuning essentially “reshapes” the probability distribution of the model such that:

	•	Answers become more concise.
	•	Irrelevant generation decreases.
	•	Responses follow user intent.

## 7.5 Limitations of a Tiny Character-Level Model

Even after instruction tuning, our model has structural limitations:

	•	It predicts characters, not words.
	•	It has a very short context window.
	•	Its capacity is orders of magnitude smaller than real LLMs.
	•	It cannot generalize beyond taught facts.

Therefore, we complement the system with a FAQ fallback, ensuring perfect outputs for known queries.

This combination—tiny LLM + deterministic layer—is exactly how many production systems handle critical queries.

## 7.6 What We Achieved

By the end of instruction tuning, the model successfully:

	•	recognizes question patterns it never saw verbatim,
	•	answers with consistent factual information,
	•	separates instruction from response,
	•	behaves predictably when interacting with the Streamlit interface.

While small, this stage demonstrates one of the most important ideas in modern AI:

**Pretraining gives the model a brain.

Instruction tuning gives the model a purpose.**

## 7.7 Preparing for the Next Stage

Instruction tuning leads naturally into:

	•	prompt engineering,
	•	evaluation frameworks,
	•	safety layers,
	•	RAG-based augmentation,
	•	multi-task finetuning.

And eventually, the much more ambitious Stage C, where we will replace character-level models with word-level or subword-level transformers, enabling far more realistic capabilities.

# 8. Streamlit UI – Building a Minimal LLM Interface.

A language model is most valuable when it becomes interactive—when users can test ideas, ask questions, and explore the model’s behavior.
To enable this, we built a simple but effective Streamlit-based UI that wraps our tiny LLM into a user-friendly chat-like interface.

Even though the model is small and educational, the frontend reflects the same principles used in real LLM applications.

⸻

## 8.1 Purpose of the UI

The Streamlit interface serves several goals:
	•	Make model behavior observable in a clean, visual environment.
	•	Allow quick experimentation with prompts and generation parameters.
	•	Provide a practical demonstration of instruction tuning and fallback mechanisms.
	•	Enable reproducible testing for the three facts we fine-tuned.

The UI transforms a research artifact into an interactive tool.

⸻

## 8.2 Main Features

The interface includes:

1. Prompt Input Panel

A text area where the user can type or modify any instruction.

2. Predefined Test Questions

A set of radio-button presets for the three core evaluation prompts, making it easy to compare model behavior over time.

3. Generation Controls

Two adjustable parameters exposed in the sidebar:
	•	max_new_tokens
	•	temperature

These allow you to observe how small models behave under different sampling settings—an essential part of LLM experimentation.

4. Integrated FAQ Fallback (Rule-Based Layer)

Before calling the model, the UI checks whether the input matches one of the known facts.
	•	If it matches → A perfect, deterministic answer is returned.
	•	If not → The tiny LLM generates a response.

This hybrid pipeline reflects how many production AI systems operate:
models + rules = reliability.

5. Full Generation Trace

The UI displays:
	•	the cleaned, processed answer, and
	•	the entire raw model output, including <instr> and <resp> tokens.

This transparency helps you understand the generation process and diagnose errors.

⸻

## 8.3 Architecture of the UI

The UI sits in the app/ directory and interacts with:
	•	the instruction model loader,
	•	the generation engine,
	•	the tokenizer,
	•	the fallback matcher,
	•	and the Streamlit runtime.

Even though the project is lightweight, the organization reflects a clean separation of concerns:
	•	Frontend logic (Streamlit)
	•	Inference logic (Python modules in src/inference)
	•	Model architecture (in src/model)

This makes it easy to:
	•	externalize the UI into Docker,
	•	replace the model later (e.g., word-level GPT),
	•	add features like temperature sweeps or latency metrics,
	•	integrate logging or analytics.

⸻

## 8.4 Running the Model Locally

The UI was explicitly designed to run without complex infrastructure.
With Python installed, launching the app is as simple as:
	•	activating the local environment,
	•	running Streamlit,
	•	and loading the model checkpoint.

Once launched, the UI runs entirely on your machine and does not require GPUs or external services.

This approach is perfect for educational projects and quick prototyping.

⸻

## 8.5 Dockerized Version

To make the UI reproducible and portable, we containerized the Streamlit application.
The Docker version:
	•	packages all dependencies,
	•	includes the trained checkpoint,
	•	exposes port 8501,
	•	and can be deployed anywhere with Docker installed.

This simulates real-world deployment workflows, where LLM services are often shipped as containerized inference microservices.

⸻

## 8.6 Why This UI Matters

While simple, the Streamlit app is an essential part of the project:
	•	It connects theory with practice.
	•	It allows iterative evaluation as we expand the project (Phase C and beyond).
	•	It demonstrates user-facing LLM interaction even with a tiny model.
	•	It provides a framework for future additions such as:
	•	multi-turn dialogue,
	•	memory,
	•	richer prompt formatting,
	•	RAG-based lookups,
	•	or switching between multiple checkpoints.

The UI makes the LLM real.


# 9. Improvements.

Building an LLM from scratch is an iterative journey.
Version 1 of the project demonstrates the full pipeline—from raw text to a functional (yet tiny) instruction-tuned language model—but it also reveals several structural limitations that naturally arise when training small models on limited data.

This section outlines the most impactful improvements planned for the next iterations of the project.
They are grouped into architectural, data-centric, training-centric, and product-centric enhancements.

⸻

## 9.1 Architectural Improvements

1. Move from character-level to word- or subword-level tokenization

The most influential change for V2.

Character-level models are easy to implement but severely limited:
	•	slow to learn long-range dependencies,
	•	inefficient (long sequences → high cost),
	•	poor semantic representation capacity,
	•	prone to repetitive or fragmented outputs.

Upgrading to WordPiece, BPE, or SentencePiece fundamentally transforms the model’s expressiveness.
This alone will produce a dramatic jump in coherence, accuracy, and generalization.

⸻

2. Expand the GPT architecture

Version 1 uses a tiny model for educational clarity.
Version 2 will strengthen it by:
	•	increasing the number of layers,
	•	widening attention heads,
	•	extending embedding sizes,
	•	using RMSNorm or LayerNorm improvements,
	•	adding rotary positional embeddings (RoPE),
	•	exploring FlashAttention for performance.

These enhancements bring the architecture closer to modern small-scale LLMs.

⸻

## 9.2 Data Improvements

1. Build a larger, cleaner corpus

V1 relies on a small locally curated text file.
For meaningful scaling, we will introduce:
	•	more diverse Spanish text sources,
	•	strict deduplication,
	•	document segmentation,
	•	long-context formatting,
	•	heuristic filtering for quality.

Better data = better model.
Even small autoregressive models benefit enormously from thoughtful curation.

⸻

2. Expand the instruction-tuning set

The V1 instruction dataset is intentionally tiny (just enough to demonstrate the pipeline).
Improvements include:
	•	increasing the number of instructions,
	•	diversifying phrasing,
	•	adding different task types (reasoning, extraction, rephrasing, classification),
	•	ensuring balanced examples per instruction family,
	•	adopting self-instruct or synthetic generation techniques.

A richer instruction set enables the model to respond in a more natural, generalizable way.

⸻

## 9.3 Training Pipeline Improvements

1. Better optimization techniques

Future versions will incorporate:
	•	cosine learning rate scheduling,
	•	warmup steps,
	•	gradient clipping,
	•	improved weight initialization,
	•	mixed-precision training,
	•	checkpoint averaging.

These refinements stabilize training and help the model converge to stronger minima.

⸻

2. Scaling the context window

The current model uses a short context (128 tokens).
Version 2 will explore:
	•	256, 512, and possibly 1024-token contexts,
	•	ALiBi or RoPE for scalable positional reasoning.

This massively improves tasks requiring multi-sentence coherence.

⸻

3. More robust validation and evaluation

Rather than relying on a few fixed test questions, we will introduce:
	•	automated evaluation suites,
	•	perplexity tracking,
	•	held-out datasets,
	•	instruction-based benchmarks.

This transforms the project from “does it run?” to “how well does it perform?”.

## 9.4 Product and Interface Improvements

1. Improve the Streamlit UI

Planned additions:
	•	chat-style conversation history,
	•	adjustable system prompts,
	•	token usage visualization,
	•	latency monitoring,
	•	multiple model checkpoints selection.

These features bring the interface closer to a real LLM playground.

2. API layer for deployment

To support external integrations, a FastAPI inference service is planned:
	•	clean REST endpoints,
	•	lightweight load balancing,
	•	Docker-ready deployment,
	•	option to run CPU/GPU versions.

This bridges the project into more production-oriented territory.

3. Modularity for experimenting with different models

The repository will support “plug-and-play” models:
	•	character-level GPT (current V1)
	•	word-level GPT (V2)
	•	distilled versions
	•	quantized versions (int8, int4)

This modularity is key for ongoing research and experimentation.

⸻

## 9.5 Vision for V2 and Beyond

The aim of these improvements is not just incremental refinement—it is to transform the project from a proof of concept into a mini research playground for LLM construction.

Version 2 will represent:
	•	a more modern architecture,
	•	a more realistic dataset pipeline,
	•	a more expressive tokenizer,
	•	a more capable model,
	•	and a significantly more powerful instruction-tuned system.

And because every component remains fully transparent and hand-built, the educational value increases rather than decreases.

# 10. Examples

This section showcases a few screenshots of the working application built in Version 1.
The images illustrate how the Streamlit interface loads the instruction-tuned model, processes user questions, and combines both inference and fallback logic to provide responses.

These examples demonstrate the end-to-end functionality of the system:
from raw text → pretrained model → instruction tuning → interactive UI.
