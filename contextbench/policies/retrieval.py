"""Retrieval context policy — a baseline that retrieves relevant past turns.

This baseline simulates a vector-database memory. It embeds all conversational
turns as they arrive. When asked to build a context with a specific query,
it embeds the query, computes cosine similarity against all past turns, and
spends a fraction of the budget retrieving the top-K most similar turns.
The remainder of the budget is filled with the most recent contiguous tail.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import ContextResult, Message, TokenCounter
from contextbench.policies.base import ContextPolicy, register_policy
from contextbench.context.state import SentenceTransformerEmbedder
from contextbench.policies.ranking import rank_topic_first, rank_turns_with_segment_prior


class RetrievalConfig(BaseModel):
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_device: str | None = None
    embedding_cache: str | None = None
    retrieval_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    # "break" stops at the first ranked turn that doesn't fit (strict top-k);
    # "skip" passes over it and keeps filling with lower-ranked turns, the
    # packing ΔContext uses -- so the two can be compared on ranking alone.
    packing: Literal["break", "skip"] = "break"
    # Control for ΔContext's topic-segment prior: the same scoring
    # (turn similarity + weight x segment similarity), over segments that carry
    # no semantic signal -- fixed runs of `segment_turns` turns, or the
    # dataset's own session boundaries. 0 disables it.
    segment_prior_weight: float = Field(default=0.0, ge=0.0)
    segment_source: Literal["fixed", "session"] = "fixed"
    segment_turns: int = Field(default=20, ge=1)
    # Topic-first control: the same two-stage search ΔContext's topic store
    # uses (segments first, then turns inside the best `topic_first_k`), over
    # fixed-size or session segments. 0 disables it.
    topic_first_k: int = Field(default=0, ge=0)


@register_policy("retrieval")
class RetrievalPolicy(ContextPolicy):
    """Retrieves top-k historical turns based on query similarity."""

    def __init__(self, config: PolicyConfig, token_counter: TokenCounter) -> None:
        super().__init__(config, token_counter)
        self._cfg = RetrievalConfig(**self._config.model_extra)
        
        self._embedder = SentenceTransformerEmbedder(
            model_name=self._cfg.embedder_model, device=self._cfg.embedder_device,
            cache_path=self._cfg.embedding_cache,
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
        topic = None

        # Only perform retrieval if we have a query and a non-zero retrieval budget
        if query and retrieval_budget > 0 and max_idx > 0:
            query_emb = self._embedder.embed(query)
            
            if self._cfg.topic_first_k > 0:
                topic = rank_topic_first(
                    {i: self._embeddings[i] for i in range(max_idx)},
                    self._segment_boundaries(history, max_idx),
                    query_emb,
                    self._cfg.topic_first_k,
                )
                ranked = topic.order
            elif self._cfg.segment_prior_weight > 0.0:
                ranked = rank_turns_with_segment_prior(
                    {i: self._embeddings[i] for i in range(max_idx)},
                    self._segment_boundaries(history, max_idx),
                    query_emb,
                    self._cfg.segment_prior_weight,
                )
            else:
                ranked = self._rank_by_similarity(query_emb, max_idx)

            # Pick top-K until budget is filled
            for i in ranked:
                msg = history[i]
                cost = self._count_message(msg)
                if retrieval_tokens + cost <= retrieval_budget:
                    selected_retrieved.append(msg)
                    retrieval_tokens += cost
                    retrieved_indices.add(i)
                elif self._cfg.packing == "break":
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
        if topic is not None:
            meta.update(topic_spans=topic.top_spans, topic_comparisons=topic.comparisons,
                        topic_segments=topic.segments)

        return ContextResult(
            messages=tuple(combined),
            token_count=exact_tokens,
            dropped_messages=dropped_count,
            policy_meta=meta,
        )

    def _rank_by_similarity(self, query_emb: np.ndarray, max_idx: int) -> list[int]:
        """Turn indices by cosine similarity to the query, most similar first."""
        similarities = []
        norm_q = np.linalg.norm(query_emb)
        for i in range(max_idx):
            e = self._embeddings[i]
            norm_e = np.linalg.norm(e)
            if norm_e > 0 and norm_q > 0:
                sim = np.dot(e, query_emb) / (norm_e * norm_q)
            else:
                sim = 0.0
            similarities.append((sim, i))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in similarities]

    def _segment_boundaries(self, history: Sequence[Message], max_idx: int) -> list[int]:
        """Last turn index of each segment for the segment-prior control."""
        if self._cfg.segment_source == "session":
            return [i for i in range(max_idx - 1)
                    if history[i + 1].session_id != history[i].session_id]
        k = self._cfg.segment_turns
        return list(range(k - 1, max_idx, k))

    def _count_message(self, message: Message) -> int:
        cached = self._count_cache.get(message)
        if cached is None:
            cached = self._token_counter.count_message(message)
            self._count_cache[message] = cached
        return cached
