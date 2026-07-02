"""Validated configuration schema.

Hydra composes and overrides the YAML config; these Pydantic models validate the
resolved result once at startup and provide typed, immutable access everywhere
downstream. Keeping validation here (and out of Hydra) means invalid experiments
fail fast with a precise error instead of surfacing as a confusing runtime crash
deep in a model forward pass.

The Hydra ``DictConfig`` -> ``dict`` bridge lives in the engine/CLI so that this
module stays free of any Hydra/OmegaConf dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Dtype = Literal["float32", "float16", "bfloat16"]
QATiming = Literal["end", "per_session"]


class ModelConfig(BaseModel):
    """A causal-LM specification. One of the *fixed factors* of an experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str
    hf_id: str
    backend: str = "hf_causal_lm"
    dtype: Dtype = "float32"
    device_map: str = "auto"
    max_new_tokens: int = Field(default=64, gt=0)
    do_sample: bool = False
    context_window: int = Field(gt=0)
    trust_remote_code: bool = False


class PolicyConfig(BaseModel):
    """A context-policy specification.

    ``extra="allow"`` so each policy may declare its own hyper-parameters in its
    YAML (e.g. a summarizer's target length) without a schema change here; the
    policy reads them from :attr:`model_extra`.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str


class DatasetConfig(BaseModel):
    """A dataset specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    path: str
    num_samples: int | None = Field(default=None, gt=0)
    qa_categories: list[int] | None = None
    include_image_captions: bool = True


class ExperimentConfig(BaseModel):
    """Run-level settings shared across components."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "default"
    output_dir: str = "experiments/results"
    qa_timing: QATiming = "end"
    system_prompt: str
    question_template: str = (
        "Conversation:\n{transcript}\n\nQuestion: {question}\nAnswer:"
    )
    log_level: str = "INFO"


class BenchmarkConfig(BaseModel):
    """The fully resolved, validated experiment configuration.

    This is the single object the engine consumes. Construct it from a plain
    dict (e.g. ``OmegaConf.to_container(cfg, resolve=True)``) via
    :meth:`model_validate`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model: ModelConfig
    policy: PolicyConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig
    budget: int = Field(gt=0, description="Token budget for the selected context.")
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _budget_fits_context_window(self) -> BenchmarkConfig:
        """Ensure the context budget plus generation headroom fits the model.

        The budget covers only the selected conversation context; the system
        prompt, the question, and ``max_new_tokens`` of generation must also fit
        inside the model's native context window. We require the budget to leave
        room for generation rather than let it overflow silently at run time.
        """
        required = self.budget + self.model.max_new_tokens
        if required >= self.model.context_window:
            raise ValueError(
                f"budget ({self.budget}) + max_new_tokens "
                f"({self.model.max_new_tokens}) = {required} must be strictly less "
                f"than the model context_window ({self.model.context_window}). "
                "Lower the budget or choose a model with a larger context window."
            )
        return self
