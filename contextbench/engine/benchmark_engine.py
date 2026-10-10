"""The benchmark engine — sweep controller and composition root.

:class:`BenchmarkEngine` turns a validated :class:`BenchmarkConfig` into a run:
it constructs every component from the registries, loads the model and tokenizer
**exactly once**, seeds all RNGs, and drives a :class:`ConversationRunner` over
the dataset while capturing run-level system metrics.

Loading the model once and reusing it for every conversation is what makes "the
model is a fixed factor" a defensible claim: nothing about the model changes
between conditions, only the policy under test.

Concrete implementations are imported here for their registration side effects —
the engine is the single place that knows the full component catalogue.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator

from contextbench.chatbot.formatting import TranscriptFormatter
from contextbench.chatbot.conversation import ConversationReplayer
from contextbench.chatbot.model import MODEL_REGISTRY, ModelBackend
from contextbench.chatbot.prompt_builder import PromptBuilder
from contextbench.chatbot.tokenizer import TokenizerWrapper
from contextbench.core.config_schema import BenchmarkConfig
from contextbench.datasets.base import DATASET_REGISTRY, Dataset
from contextbench.engine.conversation_runner import (ConversationResult,
                                                     ConversationRunner, QAPrediction)
from contextbench.evaluation.evaluator import (EVALUATOR_REGISTRY, EvaluationReport,
                                               Evaluator)
from contextbench.policies.base import POLICY_REGISTRY, ContextPolicy

# Register the built-in implementations (import for side effects).
import contextbench.datasets.locomo  # noqa: F401,E402
import contextbench.datasets.longmemeval  # noqa: F401,E402
import contextbench.policies.sliding_window  # noqa: F401,E402
import contextbench.chatbot.model as _model_impls  # noqa: F401,E402
import contextbench.evaluation.evaluator as _evaluator_impls  # noqa: F401,E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    """Run-level system metrics."""

    total_wall_s: float
    num_conversations: int
    num_predictions: int
    peak_gpu_mem_mb: float | None
    cpu_mem_mb: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """The complete (unscored) output of one benchmark run."""

    config: BenchmarkConfig
    conversation_results: tuple[ConversationResult, ...]
    system: SystemMetrics

    @property
    def predictions(self) -> tuple[QAPrediction, ...]:
        """All predictions across conversations, flattened."""
        return tuple(p for cr in self.conversation_results for p in cr.predictions)


class BenchmarkEngine:
    """Builds all components once and runs a policy over a dataset."""

    def __init__(
        self,
        config: BenchmarkConfig,
        dataset: Dataset,
        policy: ContextPolicy,
        model: ModelBackend | None,
        runner: ConversationRunner,
        evaluator: Evaluator,
    ) -> None:
        """Store the config and the (already-constructed) components.

        Prefer :meth:`from_config`; this constructor exists for testing and for
        injecting alternative components.
        """
        self._config = config
        self._dataset = dataset
        self._policy = policy
        self._model = model
        self._runner = runner
        self._evaluator = evaluator

    @classmethod
    def from_config(cls, config: BenchmarkConfig) -> BenchmarkEngine:
        """Construct the full component graph from a validated config.

        The model and tokenizer are loaded here, once, and shared with the
        policy (token counting) and the runner (generation).
        """
        # HACK: Pre-load SentenceTransformer before AutoTokenizer to prevent Windows/CUDA crashes
        if hasattr(config.policy, "embedder_model") and config.policy.embedder_model:
            from sentence_transformers import SentenceTransformer
            logger.info("Pre-loading SentenceTransformer to bypass initialization crash")
            SentenceTransformer(config.policy.embedder_model, device=config.policy.embedder_device)

        formatter = TranscriptFormatter(include_dates=config.experiment.render_session_dates)
        logger.info("Loading tokenizer '%s'", config.model.hf_id)
        tokenizer = TokenizerWrapper.from_config(config.model, formatter=formatter)

        if config.experiment.generate:
            # Only generation needs the GPU. Querying the device creates a CUDA
            # context (hundreds of MB), so selection-only runs skip it: several
            # of them alongside a generation run would otherwise exhaust GPU
            # memory.
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA DETECTED: Running on GPU (%s)", torch.cuda.get_device_name(0))
            else:
                logger.warning("NO CUDA DETECTED: Running on CPU. This will be very slow.")

        prompt_builder = PromptBuilder.from_config(config.experiment, formatter)
        dataset = DATASET_REGISTRY.create(config.dataset.name, config.dataset)
        policy = POLICY_REGISTRY.create(config.policy.name, config.policy, tokenizer)

        model: ModelBackend | None = None
        if config.experiment.generate:
            logger.info(
                "Loading model backend '%s' (%s)", config.model.backend, config.model.hf_id
            )
            model = MODEL_REGISTRY.create(config.model.backend, config.model, tokenizer)
        else:
            logger.info("Selection-only run: model not loaded, no generation")
        replayer = ConversationReplayer(config.experiment.qa_timing)
        runner = ConversationRunner(
            policy=policy,
            prompt_builder=prompt_builder,
            model=model,
            replayer=replayer,
            budget=config.budget,
        )
        evaluator = EVALUATOR_REGISTRY.create(config.experiment.evaluator)
        logger.info(
            "Engine ready: dataset=%s (%d conversations), policy=%s, budget=%d",
            dataset.name, len(dataset), policy.name, config.budget,
        )
        return cls(config, dataset, policy, model, runner, evaluator)

    def run(self) -> BenchmarkResult:
        """Run the policy over every conversation and collect results.

        Returns:
            A :class:`BenchmarkResult` with raw predictions and system metrics.
            Scoring (EM/F1) is applied separately by the evaluation stage.
        """
        uses_gpu = self._model is not None
        self._seed_everything(self._config.seed)
        if uses_gpu:
            self._reset_peak_gpu_memory()

        start = time.perf_counter()
        results: list[ConversationResult] = []
        for sample in self._iter_dataset():
            results.append(self._runner.run(sample))
        total_wall_s = time.perf_counter() - start

        num_predictions = sum(len(cr.predictions) for cr in results)
        system = SystemMetrics(
            total_wall_s=total_wall_s,
            num_conversations=len(results),
            num_predictions=num_predictions,
            peak_gpu_mem_mb=self._peak_gpu_memory_mb() if uses_gpu else None,
            cpu_mem_mb=self._cpu_memory_mb(),
        )
        logger.info(
            "Run complete: %d conversations, %d predictions in %.1fs",
            system.num_conversations, system.num_predictions, total_wall_s,
        )
        return BenchmarkResult(self._config, tuple(results), system)

    def evaluate(self, result: BenchmarkResult) -> EvaluationReport:
        """Score a run's raw predictions with the configured evaluator.

        Kept separate from :meth:`run` so predictions can be re-scored (e.g. with
        a different evaluator) without re-running the model.
        """
        report = self._evaluator.evaluate(result.predictions)
        logger.info(
            "Scored %d predictions: EM=%.3f F1=%.3f evidence_recall=%s",
            report.overall.n, report.overall.exact_match, report.overall.f1,
            "n/a" if report.overall.evidence_recall is None
            else f"{report.overall.evidence_recall:.3f}",
        )
        return report

    @property
    def config(self) -> BenchmarkConfig:
        """The run configuration."""
        return self._config

    @property
    def model(self) -> ModelBackend | None:
        """The loaded model backend (shared across the run); ``None`` if selection-only."""
        return self._model

    # -- internals -------------------------------------------------------------

    def _iter_dataset(self) -> Iterator:
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover - tqdm is a declared dependency
            return iter(self._dataset)
        return tqdm(
            self._dataset,
            total=len(self._dataset),
            desc=f"{self._policy.name}@{self._config.budget}",
            unit="conv",
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    @staticmethod
    def _reset_peak_gpu_memory() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    @staticmethod
    def _peak_gpu_memory_mb() -> float | None:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return None

    @staticmethod
    def _cpu_memory_mb() -> float:
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:  # pragma: no cover - psutil is a declared dependency
            return 0.0
