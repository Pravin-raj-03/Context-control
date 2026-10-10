r"""Semantic state construction — the foundation of the ΔContext trajectory.

A conversation is modelled as a sequence of semantic states

.. math::  C_0 \rightarrow C_1 \rightarrow \dots \rightarrow C_t,

where each :class:`SemanticState` summarizes "where the conversation is" after
turn :math:`t`. Everything downstream (change metric, accumulator, controller)
consumes these states, so *how a state is built* is the single most consequential
modelling choice. It is therefore isolated behind :class:`StateConstructor`.

Version 1 uses a **windowed centroid**: the state embedding is the mean of the
last :math:`k` turn embeddings,

.. math::  C_t = \frac{1}{|W_t|} \sum_{i \in W_t} e_i,
           \qquad W_t = \{\max(0, t-k+1), \dots, t\},

where :math:`e_i` is the embedding of turn :math:`i`. With :math:`k = 1` this
degenerates to the last-turn embedding; larger :math:`k` yields a smoother state
that is more robust to single noisy turns. The centroid is intentionally left
un-normalized because the cosine change metric is scale-invariant.

Intended future extensions: role-weighted or recency-weighted centroids, EMA
states, or learned state encoders — all drop in as new :class:`StateConstructor`
implementations without touching any consumer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """A text-to-vector encoder. Injected so the embedding model is a fixed,
    reported factor rather than a hidden confound, and swappable for tests."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D ``float32`` embedding of ``text`` (shape ``(dim,)``)."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 2-D ``float32`` array of embeddings (shape ``(n, dim)``)."""
        ...


@dataclass(frozen=True, slots=True)
class SemanticState:
    r"""The semantic state :math:`C_t` after a turn.

    Attributes:
        embedding: The state vector (shape ``(dim,)``, ``float32``). Excluded from
            equality/hashing; compare with ``np.array_equal`` when needed.
        token_count: Aggregate token count of the turns currently summarized by
            this state (a rough size proxy). The controller receives the
            authoritative full-context size separately.
        turn_index: 0-based index of the turn this state is the result of.
        timestamp: A monotonic timestamp (wall-clock or logical) for the turn.
    """

    embedding: np.ndarray = field(compare=False)
    token_count: int
    turn_index: int
    timestamp: float


class StateConstructor(ABC):
    """Builds :class:`SemanticState` objects incrementally, one turn at a time.

    Stateful by design: it maintains whatever it needs (here, a fixed-size window)
    so that ``update`` is amortized cheap rather than recomputed over full history.
    """

    def __init__(self, embedder: Embedder) -> None:
        """Store the injected embedder."""
        self._embedder = embedder

    @abstractmethod
    def update(
        self, text: str, *, turn_index: int, token_count: int, timestamp: float,
        embedding: np.ndarray | None = None,
    ) -> SemanticState:
        """Ingest one turn and return the new semantic state.

        Args:
            text: The turn's text (already rendered; e.g. ``"Caroline: ..."``).
            turn_index: 0-based turn index.
            token_count: Token count of this turn (from the shared TokenCounter).
            timestamp: Monotonic timestamp for the turn.
            embedding: A precomputed single-turn embedding for ``text``, to avoid
                re-embedding when the caller already needed one. Implementations
                that don't support reuse may ignore this and recompute.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all per-conversation state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def current(self) -> SemanticState | None:
        """The most recently constructed state, or ``None`` before the first turn."""
        raise NotImplementedError

    @property
    def embedder(self) -> Embedder:
        """The injected embedder."""
        return self._embedder


class WindowedCentroidConstructor(StateConstructor):
    r"""Windowed-centroid state constructor (Version 1 default).

    The state is the mean of the last :math:`k` turn embeddings.

    Complexity: :math:`O(k \cdot d)` time and :math:`O(k \cdot d)` memory per
    ``update`` (``d`` = embedding dim). With ``window=1`` this is the pure
    last-turn representation.
    """

    def __init__(self, embedder: Embedder, *, window: int = 3) -> None:
        """Initialize the constructor.

        Args:
            embedder: The text encoder.
            window: Number of most-recent turns averaged into the state (``k``).
                Must be >= 1.

        Raises:
            ValueError: If ``window < 1``.
        """
        super().__init__(embedder)
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        self._window = window
        self._buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=window)
        self._current: SemanticState | None = None

    def update(
        self, text: str, *, turn_index: int, token_count: int, timestamp: float,
        embedding: np.ndarray | None = None,
    ) -> SemanticState:
        """Ingest one turn and return the new windowed-centroid state.

        Args:
            embedding: A precomputed single-turn embedding for ``text``, when the
                caller already needs it for something else (e.g.
                :class:`~contextbench.policies.delta_context.DeltaContextPolicy`'s
                per-turn embedding cache for query-aware search) and wants to
                avoid embedding the same text twice. Recomputed from ``text`` if
                omitted.
        """
        if embedding is None:
            embedding = np.asarray(self._embedder.embed(text), dtype=np.float32)
        else:
            embedding = np.asarray(embedding, dtype=np.float32)
        self._buffer.append((embedding, token_count))

        stacked = np.stack([e for e, _ in self._buffer])
        centroid = stacked.mean(axis=0).astype(np.float32)
        centroid.flags.writeable = False  # treat the state vector as immutable
        window_tokens = sum(t for _, t in self._buffer)

        self._current = SemanticState(
            embedding=centroid,
            token_count=window_tokens,
            turn_index=turn_index,
            timestamp=timestamp,
        )
        return self._current

    def reset(self) -> None:
        self._buffer.clear()
        self._current = None

    @property
    def current(self) -> SemanticState | None:
        return self._current

    @property
    def window(self) -> int:
        """The window size ``k``."""
        return self._window


class SentenceTransformerEmbedder:
    """An :class:`Embedder` backed by a SentenceTransformers model.

    ``sentence_transformers`` is imported lazily so the ΔContext package (and its
    hermetic tests, which use a fake embedder) stay importable without the heavy
    dependency. Embeddings are L2-normalized by default, which makes cosine
    similarity equal to a dot product downstream.

    An optional precomputed cache (``scripts/precompute_embeddings.py``) maps
    each text to its vector, so datasets whose histories repeat the same turns
    (LongMemEval reuses sessions across questions) are embedded once in batches
    instead of once per policy run, one turn at a time. Texts missing from the
    cache are embedded normally.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize: bool = True,
        cache_path: str | None = None,
    ) -> None:
        """Load the embedding model.

        Args:
            model_name: HuggingFace SentenceTransformers model id.
            device: Torch device string, or ``None`` to auto-select.
            normalize: L2-normalize embeddings.
            cache_path: Optional ``.npz`` written by
                ``scripts/precompute_embeddings.py`` for this same model.

        Raises:
            ValueError: If the cache was built with a different model.
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._cache: dict[str, np.ndarray] = {}
        if cache_path is not None:
            self._cache = load_embedding_cache(cache_path, model_name)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not self._cache:
            return self._encode(texts)
        keys = [text_key(t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._cache]
        fresh = iter(self._encode(missing)) if missing else iter(())
        return np.stack([
            self._cache[k] if k in self._cache else next(fresh) for k in keys
        ]).astype(np.float32)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def text_key(text: str) -> str:
    """Stable cache key for a text (SHA-1 of its UTF-8 bytes)."""
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_embedding_cache(path: str, model_name: str) -> dict[str, np.ndarray]:
    """Load a precomputed text -> vector cache, checking it matches ``model_name``."""
    with np.load(path, allow_pickle=False) as data:
        cached_model = str(data["model"])
        if cached_model != model_name:
            raise ValueError(
                f"Embedding cache {path} was built with {cached_model!r}, "
                f"not {model_name!r}."
            )
        return dict(zip(data["keys"].tolist(), data["vectors"]))
