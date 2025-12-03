# src/model/gpt.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import TokenEmbedding, PositionalEmbedding, LayerNorm
from .transformer_block import TransformerBlock
from .attention import create_causal_mask


@dataclass
class GPTConfig:
    """
    Configuración del modelo GPT (versión pequeña para experimentos).

    - vocab_size: tamaño del vocabulario (número de tokens distintos)
    - max_seq_len: longitud máxima de secuencia en tokens
    - d_model: dimensión de las representaciones internas
    - n_heads: número de cabezas de atención
    - n_layers: número de bloques Transformer
    - dropout: prob. de dropout
    - layer_norm_eps: epsilon numérico para LayerNorm
    """
    vocab_size: int
    max_seq_len: int

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class GPTModel(nn.Module):
    """
    Modelo tipo GPT (decoder-only) mínimo.

    Estructura (pre-norm):
        input_ids -> token + pos embeddings
                  -> N x TransformerBlock
                  -> LayerNorm
                  -> Linear a vocab (logits)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.tok_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.d_model,
        )
        self.pos_embedding = PositionalEmbedding(
            max_seq_len=config.max_seq_len,
            embed_dim=config.d_model,
        )

        # Dropout aplicado tras sumar token + posición
        self.dropout = nn.Dropout(config.dropout)

        # Bloques Transformer
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.d_model,
                    num_heads=config.n_heads,
                    ff_hidden_dim=4 * config.d_model,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # LayerNorm final
        self.ln_f = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Proyección a vocabulario
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: compartir pesos entre embedding de entrada y proyección de salida
        self.lm_head.weight = self.tok_embedding.embedding.weight

    def forward(
        self,
        input_ids: Tensor,          # (batch_size, seq_len)
        return_attn: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """
        Forward del GPT.

        Parameters
        ----------
        input_ids:
            Tensor de enteros (batch_size, seq_len) con índices de tokens.
        return_attn:
            Si True, devuelve también una lista con los mapas de atención
            de cada bloque.

        Returns
        -------
        logits:
            Tensor de forma (batch_size, seq_len, vocab_size)
        all_attn (opcional):
            Lista de tensores de forma (batch_size, n_heads, seq_len, seq_len),
            uno por cada bloque Transformer, o None si return_attn=False.
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        device = input_ids.device

        # Embeddings
        tok_emb = self.tok_embedding(input_ids)       # (B, T, d_model)
        pos_emb = self.pos_embedding(input_ids)       # (B, T, d_model)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Máscara causal compartida por todos los bloques
        mask = create_causal_mask(seq_len, device=device)

        all_attn: Optional[List[Tensor]] = [] if return_attn else None

        # Pasar por los bloques Transformer
        for block in self.blocks:
            x, attn = block(x, mask=mask, return_attn=return_attn)
            if return_attn and all_attn is not None:
                all_attn.append(attn)

        # LayerNorm final
        x = self.ln_f(x)

        # Logits a vocabulario
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, all_attn

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,          # (batch_size, seq_len_inicial)
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Generación autoregresiva simple (greedy / sampling).

        Parameters
        ----------
        input_ids:
            Tensor de enteros (batch_size, seq_len).
        max_new_tokens:
            Cuántos tokens nuevos queremos generar.
        temperature:
            Escala los logits antes del softmax (temperature > 1 => más plano).
            Si temperature == 0, se ignora y se hace greedy puro.
        top_k:
            Si no es None, mantiene solo los top_k logits más altos
            antes del softmax (sampling más enfocado).

        Returns
        -------
        Tensor (batch_size, seq_len + max_new_tokens)
            Secuencias originales + tokens generados.
        """
        self.eval()  # modo evaluación

        out_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Recortar contexto a max_seq_len si hace falta
            if out_ids.size(1) > self.config.max_seq_len:
                context = out_ids[:, -self.config.max_seq_len :]
            else:
                context = out_ids

            # Forward: obtenemos logits para todos los pasos,
            # nos interesa solo el último token
            logits, _ = self(context)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)

            if temperature > 0.0:
                logits = logits / temperature

            # Opcional: top-k truncation
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, k=top_k, dim=-1)
                # umbral = menor logit dentro de top_k para cada fila
                min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top_k,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            # Convertir a probabilidades
            probs = torch.softmax(logits, dim=-1)

            # Estrategia de muestreo:
            #   - si temperature == 0 -> greedy (argmax)
            #   - si temperature > 0 -> sampling multinomial
            if temperature == 0.0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # Concatenar nuevo token
            out_ids = torch.cat([out_ids, next_token], dim=1)

        return out_ids