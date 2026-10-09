"""Model backends: the isolated, swappable text-generation component.

A :class:`ModelBackend` turns a chat-role prompt into a
:class:`RawResponse`. The model is one of the *fixed factors* of an
experiment — loaded once by the engine and reused across every policy and budget
in a sweep, which is what makes "the model is held constant" a defensible claim.

Phase 1 ships :class:`HFCausalLM` (HuggingFace ``AutoModelForCausalLM``) with
deterministic greedy decoding. The registry lets future backends (vLLM, hosted
APIs) be selected purely by ``model.backend`` in config.

``torch``/``transformers`` are imported lazily inside the backend so the rest of
the framework stays importable without them.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from contextbench.chatbot.tokenizer import ChatMessage, TokenizerWrapper
from contextbench.core.config_schema import ModelConfig
from contextbench.core.registry import Registry

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class RawResponse:
    """The result of one generation.

    Attributes:
        text: The decoded response (newly generated tokens only), stripped.
        prompt_tokens: Number of tokens in the prompt fed to the model.
        response_tokens: Number of tokens generated.
        latency_s: Wall-clock seconds spent in the generation call itself
            (excludes prompt construction and tokenization).
    """

    text: str
    prompt_tokens: int
    response_tokens: int
    latency_s: float


class ModelBackend(ABC):
    """Abstract text-generation backend, selected by name via :data:`MODEL_REGISTRY`."""

    def __init__(self, config: ModelConfig, tokenizer: TokenizerWrapper) -> None:
        """Store config and the paired tokenizer.

        Args:
            config: The validated model configuration.
            tokenizer: The tokenizer wrapper for the *same* ``hf_id``; used for
                chat templating and decoding so the backend and the token counter
                agree.
        """
        self._config = config
        self._tokenizer = tokenizer

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage]) -> RawResponse:
        """Generate a response for a chat-role prompt."""
        raise NotImplementedError

    @property
    def config(self) -> ModelConfig:
        """The validated model configuration."""
        return self._config

    @property
    def name(self) -> str:
        """The model's configured name."""
        return self._config.name


MODEL_REGISTRY: Registry[ModelBackend] = Registry("model backend")
"""Global model-backend registry, keyed by ``model.backend``."""

register_model = MODEL_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the model registry."""


@register_model("hf_causal_lm")
class HFCausalLM(ModelBackend):
    """A HuggingFace ``AutoModelForCausalLM`` backend with deterministic decoding."""

    def __init__(self, config: ModelConfig, tokenizer: TokenizerWrapper) -> None:
        """Load the model onto its device(s).

        Args:
            config: Model configuration (``hf_id``, ``dtype``, ``device_map``,
                generation settings).
            tokenizer: The paired tokenizer wrapper.
        """
        super().__init__(config, tokenizer)
        import torch
        from transformers import AutoModelForCausalLM

        self._torch = torch
        dtype = self._resolve_dtype(config.dtype)

        # On a CPU-only host, `device_map="auto"` lets accelerate offload weights
        # to disk when RAM is tight, which silently cripples throughput. Prefer a
        # full in-RAM CPU load and reserve accelerate dispatch for when a GPU is
        # actually present.
        device_map = config.device_map
        if device_map == "auto" and not torch.cuda.is_available():
            device_map = None

        self._model = AutoModelForCausalLM.from_pretrained(
            config.hf_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if device_map is None:
            self._model.to("cpu")
        self._model.eval()
        self._device = next(self._model.parameters()).device

        # A pad token is required to silence generation warnings and to pad if
        # batching is ever added; fall back to EOS when the model defines none.
        hf_tok = tokenizer.tokenizer
        self._pad_token_id = hf_tok.pad_token_id
        if self._pad_token_id is None:
            self._pad_token_id = hf_tok.eos_token_id

    @staticmethod
    def _resolve_dtype(name: str) -> "torch.dtype":
        import torch

        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]

    def generate(self, messages: Sequence[ChatMessage]) -> RawResponse:
        """Greedily generate a response for a chat-role prompt.

        Determinism: the seed is set before each call and decoding is greedy when
        ``do_sample`` is false, so identical prompts yield identical outputs.

        Args:
            messages: Chat-role messages (``[{"role": ..., "content": ...}, ...]``).

        Returns:
            A :class:`RawResponse` with the decoded text and token/latency metrics.
        """
        torch = self._torch
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        prompt_tokens = len(input_ids)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        # Single, unpadded sequence -> every position is attended to. Passing this
        # explicitly is correct and silences the pad==eos inference warning.
        attention_mask = torch.ones_like(input_tensor)

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": self._config.max_new_tokens,
            "do_sample": self._config.do_sample,
            "pad_token_id": self._pad_token_id,
            "use_cache": True,
        }

        torch.manual_seed(0)  # decoding is greedy, but keep the RNG pinned
        start = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                input_tensor, attention_mask=attention_mask, **gen_kwargs
            )
        latency_s = time.perf_counter() - start

        new_token_ids = output[0][prompt_tokens:].tolist()
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return RawResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            response_tokens=len(new_token_ids),
            latency_s=latency_s,
        )

    @property
    def device(self) -> "torch.device":
        """The device the model's parameters live on."""
        return self._device

    def __repr__(self) -> str:
        return f"HFCausalLM(hf_id={self._config.hf_id!r}, device={self._device})"


@register_model("hf_prefix_cached")
class PrefixCachedHFCausalLM(HFCausalLM):
    """:class:`HFCausalLM` with prefix KV-cache reuse and chunked prefill.

    Consecutive prompts in a run often share a long prefix: every question about
    a conversation repeats the system prompt, and a context policy that keeps
    the same turns renders the same transcript. This backend keeps the KV cache
    of the previous prompt, crops it to the longest token prefix the new prompt
    shares with it, and only computes the rest. Long prompts are prefilled in
    ``_PREFILL_CHUNK``-token chunks, which bounds activation memory and is what
    lets a whole LoCoMo conversation (~25K tokens) run on an 8 GB GPU.

    Decoding is still greedy and the prompt is unchanged, so outputs match
    :class:`HFCausalLM` up to floating-point differences between chunked and
    single-pass attention. ``latency_s`` includes the (possibly cached) prefill,
    so it reflects serving cost with prefix caching, which differs by policy:
    a policy whose context is stable across questions gets more cache reuse.
    """

    # 512 keeps prefill attention scores small enough that a ~30K-token prompt
    # still fits beside a 3B bf16 model on 8 GB, even without a fused kernel.
    _PREFILL_CHUNK = 512

    def __init__(self, config: ModelConfig, tokenizer: TokenizerWrapper) -> None:
        super().__init__(config, tokenizer)
        self._cache = None
        self._cached_ids: list[int] = []

    def generate(self, messages: Sequence[ChatMessage]) -> RawResponse:
        torch = self._torch
        from transformers import DynamicCache

        input_ids = list(self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        ))
        prompt_tokens = len(input_ids)

        torch.manual_seed(0)
        start = time.perf_counter()
        with torch.inference_mode():
            # Reuse the shared prefix, but always leave >= 1 prompt token for
            # generate() to feed, which it needs to produce the first logits.
            shared = 0
            for cached, new in zip(self._cached_ids, input_ids):
                if cached != new:
                    break
                shared += 1
            keep = min(shared, prompt_tokens - 1)
            if self._cache is None or keep == 0:
                self._cache = DynamicCache()
                keep = 0
            else:
                self._cache.crop(keep)

            for begin in range(keep, prompt_tokens - 1, self._PREFILL_CHUNK):
                end = min(begin + self._PREFILL_CHUNK, prompt_tokens - 1)
                self._model(
                    input_ids=torch.tensor([input_ids[begin:end]], device=self._device),
                    past_key_values=self._cache,
                    cache_position=torch.arange(begin, end, device=self._device),
                    use_cache=True,
                    logits_to_keep=1,
                )

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
            output = self._model.generate(
                input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                past_key_values=self._cache,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=self._config.do_sample,
                pad_token_id=self._pad_token_id,
                use_cache=True,
            )
        latency_s = time.perf_counter() - start
        # The cache now also holds generated tokens; the next call crops them
        # away, since they are never part of a shared prompt prefix.
        self._cached_ids = input_ids

        new_token_ids = output[0][prompt_tokens:].tolist()
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return RawResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            response_tokens=len(new_token_ids),
            latency_s=latency_s,
        )
