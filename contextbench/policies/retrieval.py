"""Retrieval context policy — a baseline that retrieves relevant past turns.

This baseline simulates a vector-database memory. It embeds all conversational
turns as they arrive. When asked to build a context with a specific query,
it embeds the query, computes cosine similarity against all past turns, and
spends a fraction of the budget retrieving the top-K most similar turns.
The remainder of the budget is filled with the most recent contiguous tail.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from pydantic import BaseModel, Field

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import ContextResult, Message, TokenCounter
from contextbench.policies.base import ContextPolicy, register_policy
from contextbench.context.state import SentenceTransformerEmbedder


class RetrievalConfig(BaseModel):
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_device: str | None = None
    retrieval_fraction: float = Field(default=0.5, ge=0.0, le=1.0)


@register_policy("retrieval")
class RetrievalPolicy(ContextPolicy):
    """Retrieves top-k historical turns based on query similarity."""

    def __init__(self, config: PolicyConfig, token_counter: TokenCounter) -> None:
        super().__init__(config, token_counter)
        self._cfg = RetrievalConfig(**self._config.model_extra)
        
        self._embedder = SentenceTransformerEmbedder(
            model_name=self._cfg.embedder_model, device=self._cfg.embedder_device
        )
        self._embeddings: list[np.ndarray] = []
        self._count_cache: dict[Message, int] = {}

    def reset(self) -> None:
        self._embeddings.clear()
        self._count_cache.clear()

    def observe(self, message: Message) -> None:
        """Embed the turn as it arrives to build the search index."""
        text = f"{message.speaker or message.role}: {message.content}"
        emb = self._embedder.embed(text)
        self._embeddings.append(emb)

    def build_context(
        self, history: Sequence[Message], query: str | None, budget: int
    ) -> ContextResult:
        if budget <= 0 or not history:
            return ContextResult(messages=(), token_count=0, dropped_messages=len(history))

        # Robustness: if history is longer than observed embeddings (e.g. observe wasn't called properly),
        # fallback to no retrieval or dynamic embedding. We will assume observe was called correctly.
        # Ensure we only consider embeddings up to len(history).
        max_idx = min(len(self._embeddings), len(history))

        retrieval_budget = int(budget * self._cfg.retrieval_fraction)
        
        selected_retrieved: list[Message] = []
        retrieved_indices = set()
        retrieval_tokens = 0

        # Only perform retrieval if we have a query and a non-zero retrieval budget
        if query and retrieval_budget > 0 and max_idx > 0:
            query_emb = self._embedder.embed(query)
            
            # Compute similarities
            similarities = []
            for i in range(max_idx):
                e = self._embeddings[i]
                norm_e = np.linalg.norm(e)
                norm_q = np.linalg.norm(query_emb)
                if norm_e > 0 and norm_q > 0:
                    sim = np.dot(e, query_emb) / (norm_e * norm_q)
                else:
                    sim = 0.0
                similarities.append((sim, i))
                
            # Sort descending by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Pick top-K until budget is filled
            for sim, i in similarities:
                msg = history[i]
                cost = self._count_message(msg)
                if retrieval_tokens + cost <= retrieval_budget:
                    selected_retrieved.append(msg)
                    retrieval_tokens += cost
                    retrieved_indices.add(i)
                else:
                    # We might skip large messages and try smaller ones, but usually top-k is strictly rank ordered.
                    # We break to preserve strict rank ordering.
                    break

        # Fill remainder with recent tail
        tail_budget = budget - retrieval_tokens
        selected_tail: list[Message] = []
        tail_tokens = 0

        for turn_idx in range(len(history) - 1, -1, -1):
            if turn_idx in retrieved_indices:
                continue
            
            msg = history[turn_idx]
            cost = self._count_message(msg)
            if tail_tokens + cost <= tail_budget:
                selected_tail.append(msg)
                tail_tokens += cost
            else:
                break
                
        # Combine and sort chronologically
        combined = selected_retrieved + selected_tail
        combined.sort(key=lambda m: m.turn_id)
        
        # Exact-fit guarantee
        exact_tokens = self._token_counter.count_messages(combined)
        while combined and exact_tokens > budget:
            # Pop oldest. Since retrieval mixes old and new, this might drop a retrieved turn.
            combined.pop(0)
            exact_tokens = self._token_counter.count_messages(combined)

        dropped_count = len(history) - len(combined)
        
        meta = {
            "retrieved_count": len(selected_retrieved),
            "tail_count": len(selected_tail),
            "retrieval_budget": retrieval_budget,
            "tail_budget": tail_budget,
            "retrieval_tokens": retrieval_tokens,
            "tail_tokens": tail_tokens,
        }

        return ContextResult(
            messages=tuple(combined),
            token_count=exact_tokens,
            dropped_messages=dropped_count,
            policy_meta=meta,
        )

    def _count_message(self, message: Message) -> int:
        cached = self._count_cache.get(message)
        if cached is None:
            cached = self._token_counter.count_message(message)
            self._count_cache[message] = cached
        return cached
