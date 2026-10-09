"""Tokenizer wrapper and the shared token counter.

:class:`TokenizerWrapper` wraps a HuggingFace ``AutoTokenizer`` and implements
the :class:`~contextbench.core.types.TokenCounter` protocol. A single instance is
constructed per benchmark run and injected into every policy, so all policies
measure their budgets with the same ruler — the exact tokenizer the model uses.

Two counting granularities are provided:

* :meth:`count_message` counts one message as its transcript line. It is additive
  and cheap, so greedy budget-filling policies (e.g. sliding window) use it.
* :meth:`count_messages` counts a whole rendered transcript block exactly. Because
  subword tokenization is not perfectly additive across line boundaries, this is
  the authoritative figure recorded in a
  :class:`~contextbench.core.types.ContextResult`.

``transformers`` is imported lazily inside the constructor so the rest of the
framework (policies, datasets, config) stays importable without the heavy
dependency installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from contextbench.chatbot.formatting import TranscriptFormatter
from contextbench.core.config_schema import ModelConfig
from contextbench.core.types import Message

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# A chat message as consumed by ``apply_chat_template``.
ChatMessage = dict[str, str]


class TokenizerWrapper:
    """Wraps an ``AutoTokenizer`` and counts tokens the way the model sees them."""

    def __init__(
        self,
        hf_id: str,
        *,
        formatter: TranscriptFormatter | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> None:
        """Load a tokenizer from the Hub (or local cache).

        Args:
            hf_id: HuggingFace model/tokenizer id.
            formatter: Transcript formatter used for message-level counting. The
                *same* instance should be given to the prompt builder so counts
                and prompts cannot diverge. Defaults to a fresh
                :class:`TranscriptFormatter`.
            trust_remote_code: Passed through to ``from_pretrained``.
            revision: Optional pinned model revision for reproducibility.
        """
        from transformers import AutoTokenizer

        self._hf_id = hf_id
        self._formatter = formatter or TranscriptFormatter()
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code, revision=revision, use_fast=False
        )

    @classmethod
    def from_config(
        cls, config: ModelConfig, *, formatter: TranscriptFormatter | None = None
    ) -> TokenizerWrapper:
        """Construct from a validated :class:`ModelConfig`."""
        return cls(
            config.hf_id,
            formatter=formatter,
            trust_remote_code=config.trust_remote_code,
        )

    # -- encode / decode -------------------------------------------------------

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode raw text to token ids (special tokens off by default)."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # -- TokenCounter protocol -------------------------------------------------

    def count_text(self, text: str) -> int:
        """Return the token count of a raw string, excluding special tokens."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def count_message(self, message: Message) -> int:
        """Return the token count of one message rendered as a transcript line."""
        return self.count_text(self._formatter.format_line(message))

    def count_messages(self, messages: Sequence[Message]) -> int:
        """Return the exact token count of a rendered transcript block."""
        if not messages:
            return 0
        return self.count_text(self._formatter.render(messages))

    # -- chat template ---------------------------------------------------------

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """Apply the model's chat template to a list of chat-role messages."""
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )

    def count_chat(self, messages: Sequence[ChatMessage]) -> int:
        """Return the exact token count of a full chat-templated prompt.

        This includes the template's structural tokens (role markers, BOS, the
        trailing generation prompt) and is the true prompt-token figure for a
        generation.
        """
        ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return len(ids)

    # -- properties ------------------------------------------------------------

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The underlying HuggingFace tokenizer."""
        return self._tokenizer

    @property
    def formatter(self) -> TranscriptFormatter:
        """The transcript formatter used for message-level counting."""
        return self._formatter

    @property
    def hf_id(self) -> str:
        """The HuggingFace id this tokenizer was loaded from."""
        return self._hf_id

    @property
    def model_max_length(self) -> int:
        """The tokenizer's reported maximum sequence length.

        Note: some tokenizers report a sentinel value here; the authoritative
        budget bound is ``ModelConfig.context_window``, not this property.
        """
        return self._tokenizer.model_max_length

    def __repr__(self) -> str:
        return f"TokenizerWrapper(hf_id={self._hf_id!r})"
