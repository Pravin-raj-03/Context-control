"""The context-policy abstraction — the one swappable component on the
critical path, and therefore the sole independent variable of every experiment.

A policy answers one question: *given the full conversation so far, a token
budget, and (optionally) the current query, which messages should the model
see?* It knows nothing about datasets, prompting, models, or evaluation.

Contract and invariants
------------------------
* ``build_context`` is the only required method. It MUST return a
  :class:`~contextbench.core.types.ContextResult` whose ``token_count`` is
  ``<= budget`` and whose ``messages`` are a chronologically ordered subset of
  ``history`` with no message truncated (messages are atomic).
* ``observe`` is an optional incremental-state hook. Stateless policies (e.g.
  sliding window) leave it as the inherited no-op; stateful policies
  (summarization, retrieval, ΔContext) override it to update derived state as
  each turn arrives, avoiding an O(n) recomputation on every ``build_context``.
* ``reset`` MUST clear all per-conversation state so one policy instance can be
  reused across conversations in a sweep.

Budget semantics: ``budget`` bounds the tokens of the *selected context only*.
The system prompt and question are added by the prompt builder on top and are
accounted for separately by the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.registry import Registry
from contextbench.core.types import ContextResult, Message, TokenCounter

POLICY_REGISTRY: Registry[ContextPolicy] = Registry("policy")
"""Global policy registry. Decorate a subclass with
``@POLICY_REGISTRY.register("name")`` (or :data:`register_policy`) to make it
selectable by config."""

register_policy = POLICY_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the policy registry."""


class ContextPolicy(ABC):
    """Abstract base class for all context-management policies.

    Subclasses receive their validated config and the shared token counter via
    dependency injection, so a policy is model-agnostic yet counts tokens with
    the exact tokenizer the model uses.
    """

    def __init__(self, config: PolicyConfig, token_counter: TokenCounter) -> None:
        """Initialize the policy.

        Args:
            config: The validated policy configuration. Policy-specific
                hyper-parameters live in ``config.model_extra``.
            token_counter: The shared token counter used to respect the budget.
        """
        self._config = config
        self._token_counter = token_counter

    def observe(self, message: Message) -> None:
        """Ingest one newly appended history turn (optional hook).

        The default implementation is a no-op, which is correct for stateless
        policies. Override to maintain incremental derived state (a rolling
        summary, a vector index, running change scores, ...).

        Args:
            message: The turn just appended to the conversation history.
        """

    @abstractmethod
    def build_context(
        self, history: Sequence[Message], query: str | None, budget: int
    ) -> ContextResult:
        """Select the messages the model should see for the next generation.

        Args:
            history: The full conversation so far, in chronological order. Owned
                by the runner; policies must not mutate it.
            query: The current question, or ``None`` when no query conditions the
                selection. Query-agnostic policies ignore it.
            budget: Maximum number of tokens the returned context may occupy.

        Returns:
            A :class:`~contextbench.core.types.ContextResult` with
            ``token_count <= budget`` and atomic, chronologically ordered
            messages drawn from ``history``.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all per-conversation state before replaying a new conversation."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """The policy's configured name."""
        return self._config.name

    @property
    def config(self) -> PolicyConfig:
        """The validated policy configuration."""
        return self._config

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
