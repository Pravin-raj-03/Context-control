"""Per-conversation orchestration.

The :class:`ConversationRunner` owns the canonical conversation history and
drives one conversation end to end: it replays turns (``append`` then
``observe``) and, at each evaluation checkpoint, answers every QA probe by asking
the active policy for context, building a prompt, and generating.

It is deliberately ignorant of two things:

* **How context is selected** — it only holds a :class:`ContextPolicy` by its
  abstract interface and calls ``build_context``; it never imports a concrete
  policy.
* **How predictions are scored** — it emits raw :class:`QAPrediction` records
  (prediction text + gold + system metrics). Scoring is a separate pass applied
  by the engine, so predictions can be re-scored with new metrics without
  re-running the model.

Run without a model (``model=None``), it still asks the policy for every
context and records evidence recall, but skips prompting and generation. That
selection-only mode measures what each policy *keeps* in minutes, with the
generator's noise taken out entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from contextbench.chatbot.conversation import Conversation, ConversationReplayer
from contextbench.chatbot.model import ModelBackend
from contextbench.chatbot.prompt_builder import PromptBuilder
from contextbench.core.types import QAItem
from contextbench.datasets.base import ConversationSample
from contextbench.evaluation.evidence import evidence_recall, evidence_turn_ids
from contextbench.policies.base import ContextPolicy


@dataclass(frozen=True, slots=True)
class QAPrediction:
    """One QA probe's raw (unscored) result and its per-generation metrics."""

    # identity / provenance
    sample_id: str
    qa_id: str | None
    budget: int
    category: int | None
    # the probe
    question: str
    gold_answer: str
    prediction: str
    # context-policy metrics
    context_tokens: int
    context_messages: int
    dropped_messages: int
    # generation metrics
    prompt_tokens: int
    response_tokens: int
    latency_s: float
    # policy-specific diagnostics
    policy_meta: Mapping[str, Any] = field(default_factory=dict, compare=False)
    # adversarial probe: the right answer is that the conversation never says
    adversarial: bool = False
    # fraction of the probe's evidence turns in the selected context
    # (None when the probe has no evidence present in the history)
    evidence_recall: float | None = None
    # turn_ids of the selected context, for offline re-analysis
    context_turn_ids: tuple[int, ...] = ()
    # turn_ids of the probe's evidence turns present in the history
    evidence_turn_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversationResult:
    """All predictions for one replayed conversation."""

    sample_id: str
    num_turns: int
    predictions: tuple[QAPrediction, ...]


class ConversationRunner:
    """Replays one conversation and answers its QA probes."""

    def __init__(
        self,
        policy: ContextPolicy,
        prompt_builder: PromptBuilder,
        model: ModelBackend | None,
        replayer: ConversationReplayer,
        budget: int,
    ) -> None:
        """Wire the (shared) components the runner drives.

        Args:
            policy: The active context policy (held by its abstract interface).
            prompt_builder: Assembles prompts from selected messages + question.
            model: The text-generation backend, or ``None`` for selection-only
                runs (no prompting or generation; predictions are empty).
            replayer: Produces replay steps and QA checkpoints.
            budget: The context token budget handed to the policy per QA.
        """
        self._policy = policy
        self._prompt_builder = prompt_builder
        self._model = model
        self._replayer = replayer
        self._budget = budget
        self._conversation = Conversation()

    def run(self, sample: ConversationSample) -> ConversationResult:
        """Replay one conversation and answer its QA probes.

        Resets the policy and history first, so a single runner/policy instance is
        safely reused across every sample in a sweep.

        Args:
            sample: The conversation and its QA probes.

        Returns:
            A :class:`ConversationResult` with one :class:`QAPrediction` per probe.
        """
        self._policy.reset()
        self._conversation.reset()

        predictions: list[QAPrediction] = []
        for step in self._replayer.iter_steps(sample):
            self._conversation.append(step.message)
            self._policy.observe(step.message)
            if step.is_checkpoint:
                for qa in step.checkpoint_qa:
                    predictions.append(self._answer(qa, sample.sample_id))

        return ConversationResult(
            sample_id=sample.sample_id,
            num_turns=len(self._conversation),
            predictions=tuple(predictions),
        )

    def _answer(self, qa: QAItem, sample_id: str) -> QAPrediction:
        history = self._conversation.history
        context = self._policy.build_context(history, qa.question, self._budget)
        if self._model is not None:
            prompt = self._prompt_builder.build(
                context.messages, qa.question, question_date=qa.meta.get("question_date")
            )
            response = self._model.generate(prompt)
            text, prompt_tokens = response.text, response.prompt_tokens
            response_tokens, latency_s = response.response_tokens, response.latency_s
        else:
            text, prompt_tokens, response_tokens, latency_s = "", 0, 0, 0.0
        return QAPrediction(
            sample_id=sample_id,
            qa_id=qa.qa_id,
            budget=self._budget,
            category=qa.category,
            question=qa.question,
            gold_answer=qa.answer,
            prediction=text,
            context_tokens=context.token_count,
            context_messages=context.num_messages,
            dropped_messages=context.dropped_messages,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            latency_s=latency_s,
            policy_meta=dict(context.policy_meta),
            adversarial=bool(qa.meta.get("adversarial", False)),
            evidence_recall=evidence_recall(qa.evidence, context.messages, history),
            context_turn_ids=tuple(m.turn_id for m in context.messages),
            evidence_turn_ids=evidence_turn_ids(qa.evidence, history),
        )
