r"""Checkpoint (snapshot) manager — Module 6.

When the controller decides a checkpoint should occur, the current semantic state
is stored here as a :class:`Checkpoint`. Version 1 keeps checkpoints in a Python
list; the algorithm never depends on the storage backend, so SQLite, JSON, or a
vector database can replace :class:`InMemoryCheckpointManager` behind the same
interface. **Retrieval is intentionally out of scope** — this manager stores and
enumerates checkpoints but does not rank them against a query.

Complexity: ``add`` and ``latest`` are :math:`O(1)`; ``all`` returns a snapshot
tuple. Intended future backends: persistent stores and ANN indexes for
retrieval-augmented policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from contextbench.context.state import SemanticState
from contextbench.core.registry import Registry


@dataclass(frozen=True, slots=True)
class Checkpoint:
    r"""A stored semantic checkpoint.

    Attributes:
        checkpoint_id: Monotonic 0-based id in creation order.
        state: The semantic state :math:`C_t` at checkpoint time.
        turn_index: The turn at which the checkpoint fired.
        token_count: Tokens covered by the state's window at checkpoint time.
        timestamp: The turn's timestamp.
        accumulated_change: The accumulator value :math:`A_t` that triggered it.
        turn_span: ``(start_turn, end_turn)`` this checkpoint summarizes since the
            previous checkpoint.
        meta: Free-form provenance (e.g. a future summary or message references).
    """

    checkpoint_id: int
    state: SemanticState
    turn_index: int
    token_count: int
    timestamp: float
    accumulated_change: float
    turn_span: tuple[int, int]
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)


class CheckpointManager(ABC):
    """Stores and enumerates checkpoints. Storage backend is swappable."""

    @abstractmethod
    def add(self, checkpoint: Checkpoint) -> None:
        """Append a checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def all(self) -> tuple[Checkpoint, ...]:
        """Return all checkpoints in creation order."""
        raise NotImplementedError

    @abstractmethod
    def latest(self) -> Checkpoint | None:
        """Return the most recent checkpoint, or ``None`` if empty."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Drop all checkpoints (new conversation)."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def next_id(self) -> int:
        """The id the next added checkpoint should use."""
        return len(self)


CHECKPOINT_REGISTRY: Registry[CheckpointManager] = Registry("checkpoint manager")
"""Global checkpoint-manager registry, keyed by ``delta.checkpoint_manager``."""

register_checkpoint_manager = CHECKPOINT_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the checkpoint registry."""


@register_checkpoint_manager("in_memory")
class InMemoryCheckpointManager(CheckpointManager):
    """A list-backed checkpoint store (Version 1)."""

    def __init__(self) -> None:
        self._checkpoints: list[Checkpoint] = []

    def add(self, checkpoint: Checkpoint) -> None:
        self._checkpoints.append(checkpoint)

    def all(self) -> tuple[Checkpoint, ...]:
        return tuple(self._checkpoints)

    def latest(self) -> Checkpoint | None:
        return self._checkpoints[-1] if self._checkpoints else None

    def reset(self) -> None:
        self._checkpoints.clear()

    def __len__(self) -> int:
        return len(self._checkpoints)
