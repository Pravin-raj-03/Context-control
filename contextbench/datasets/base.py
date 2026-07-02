"""The dataset abstraction.

A dataset is an iterable of :class:`ConversationSample` — a fully parsed
conversation plus its QA probes — behind a common interface, so datasets are
interchangeable. Loaders normalize raw, dataset-specific JSON into the shared
:class:`~contextbench.core.types.Message` / :class:`~contextbench.core.types.QAItem`
types; everything downstream is dataset-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from contextbench.core.config_schema import DatasetConfig
from contextbench.core.registry import Registry
from contextbench.core.types import Message, QAItem


@dataclass(frozen=True, slots=True)
class ConversationSample:
    """One conversation and the questions asked about it.

    Attributes:
        sample_id: Stable identifier for the conversation.
        messages: All turns in chronological order (across all sessions).
        qa_items: The QA probes evaluated against this conversation.
        speakers: The distinct speaker names, in a stable order.
        meta: Auxiliary conversation-level provenance (session dates, counts).
    """

    sample_id: str
    messages: tuple[Message, ...]
    qa_items: tuple[QAItem, ...]
    speakers: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def num_turns(self) -> int:
        """Total number of conversation turns."""
        return len(self.messages)

    @property
    def num_qa(self) -> int:
        """Number of QA probes."""
        return len(self.qa_items)


class Dataset(ABC):
    """Abstract base class for dataset loaders.

    A loader parses its source in ``__init__`` and then yields
    :class:`ConversationSample` objects. Selected by name via
    :data:`DATASET_REGISTRY`.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Store the validated dataset configuration.

        Args:
            config: The dataset section of the resolved experiment config.
        """
        self._config = config

    @abstractmethod
    def __iter__(self) -> Iterator[ConversationSample]:
        """Iterate over the conversations in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of conversations."""
        raise NotImplementedError

    @property
    def config(self) -> DatasetConfig:
        """The validated dataset configuration."""
        return self._config

    @property
    def name(self) -> str:
        """The dataset's configured name."""
        return self._config.name


DATASET_REGISTRY: Registry[Dataset] = Registry("dataset")
"""Global dataset registry. Decorate a loader with
``@DATASET_REGISTRY.register("name")`` (or :data:`register_dataset`)."""

register_dataset = DATASET_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the dataset registry."""
