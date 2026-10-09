# ContextBench

A modular framework for **benchmarking context-management policies** for
conversational transformer models under fixed token budgets.

The transformer, tokenizer, prompt format, dataset, and evaluation are held
**fixed**. The *only* swappable component on the critical path is the context
policy — the strategy that decides which of a growing conversation history the
model actually sees. Any measured difference in accuracy, latency, or memory is
therefore attributable to the policy alone. ContextBench is built to make that
claim defensible to a reviewer.

Four policies ship: **sliding window**, a heuristic extractive **summary**,
query-time **retrieval**, and **ΔContext**, which tracks the rate of semantic
change turn by turn, drops checkpoints at topic shifts, and uses the resulting
topic segments when ranking what to keep. New policies slot in without touching
any other part of the codebase.

---

## Why

Real deployments run conversational models under a hard context budget. When the
history exceeds it, *something* must be dropped or compressed — and that choice
is usually an afterthought. ContextBench treats the context-selection policy as
the independent variable of a controlled experiment and measures the quality /
cost trade-offs each policy makes on long, multi-session dialogues.

## Architecture

```
                          BenchmarkEngine            (loads model once, drives the sweep)
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
   LoCoMoDataset            ContextPolicy              Evaluator
        └────────────────────────┼────────────────────────┘
                                 ▼
                         ConversationRunner          (owns history; policy-agnostic)
                                 ▼
                          PromptBuilder ─▶ Model ─▶ ExperimentLogger
```

- **`ContextPolicy`** — the one abstraction the whole project revolves around.
  `build_context(history, query, budget) -> ContextResult` is the only required
  method; an optional `observe(message)` hook lets stateful policies
  (summarization, retrieval, ΔContext) maintain incremental state.
- **`ConversationRunner`** — owns the canonical history, replays a conversation
  turn by turn, and answers each QA probe. It never imports a concrete policy.
- **`BenchmarkEngine`** — the composition root: builds every component from the
  registries, loads the model and tokenizer **once**, and runs the sweep.
- **One shared `TokenCounter`** is injected into every policy, so all policies
  measure their budget with the exact tokenizer the model uses — the framework's
  core fairness guarantee.

```
contextbench/
  core/        types, registry, config schema (Pydantic)
  chatbot/     tokenizer, model backend, prompt builder, conversation, formatting
  context/     ΔContext machinery: embed -> state -> change metric -> accumulate -> checkpoint
  policies/    base ContextPolicy + sliding_window, summary, retrieval, delta_context
  datasets/    base + LoCoMo loader
  evaluation/  normalization, exact_match, f1, abstention, evidence recall, evaluators
  engine/      benchmark_engine, conversation_runner
  runlog/      experiment logger
```

## Installation

Requires Python ≥ 3.10.

```bash
python -m pip install -e .            # runtime deps
python -m pip install -e ".[dev]"     # + pytest, ruff, mypy
```

A GPU is recommended for the 3B evaluation models; the default config uses
**TinyLlama-1.1B**, which runs on CPU for development.

## Dataset

ContextBench uses **LoCoMo** (very long-term, multi-session conversations with
categorized QA). The dataset is *not* bundled. Obtain `locomo10.json` and place
it at `data/locomo10.json` (or point `dataset.path` at it). Only the
`conversation` and `qa` fields are used; image turns are inlined as text.

### Evaluation setup (`experiment=locomo_v2`, the default)

The v1 setup (`experiment=default`, kept to reproduce runs made before
2026-10-09) mis-scored 38% of LoCoMo for every policy. `locomo_v2` fixes both
problems:

- **Adversarial questions** (category 5) name the wrong speaker on purpose; the
  correct response is that the conversation never says. Their
  `adversarial_answer` field is the trap answer, which v1 graded against. The
  `locomo_qa` evaluator scores them on abstention, as LoCoMo's own evaluation
  does, and reports them separately: the headline F1 is over answerable
  questions (categories 1-4) only.
- **Session dates** were never shown to the model, so temporal questions
  (category 2) were unanswerable. Each transcript line now starts with its
  session date, and those tokens count against the budget.

## Quick start

```bash
# Single-conversation smoke run (fast; TinyLlama on CPU)
python main.py --budget 512 --max-qa 5

# Full benchmark (all conversations), writes a run directory
python benchmark.py

# Override anything via Hydra
python benchmark.py budget=1024 model=qwen2.5-3b

# Sweep budgets (each is its own run directory)
python benchmark.py --multirun budget=512,1024,2048

# Selection only: build every context and score evidence recall, no model.
# A full LoCoMo pass takes minutes instead of hours.
python benchmark.py experiment.generate=false policy=delta_context
```

For larger grids, `scripts/sweep.py` runs each (variant, budget) in its own
process, `scripts/summarize_runs.py` tabulates a results directory with a
dev/test split by conversation, and `scripts/compare_runs.py` runs a paired
bootstrap between two runs. For long budgets, `model.backend=hf_prefix_cached`
reuses the KV cache of the prompt prefix shared with the previous question and
prefills in chunks, so even a whole conversation fits on an 8 GB GPU.

Configuration is composed from Hydra groups under `configs/`
(`model/`, `policy/`, `dataset/`, `experiment/`) and validated by Pydantic at
startup — invalid experiments fail immediately with a precise error.

## Output

Each run writes a self-contained, hashed directory under
`experiments/results/`:

| File | Contents |
|------|----------|
| `config.json`    | the resolved, validated configuration |
| `metrics.json`   | system metrics + quality aggregates (overall & per category) |
| `per_sample.csv` | one row per QA probe (scores + system metrics), for pandas |
| `outputs.jsonl`  | raw predictions incl. policy diagnostics — re-scorable offline |
| `run.log`        | the captured log stream |

The directory name encodes policy, model, budget, and a config hash, so two runs
are comparable iff their hashes match.

## Metrics

- **Quality**: exact match and token-level F1 (SQuAD-style normalization),
  reported overall and per LoCoMo category (multi-hop, temporal, open-domain,
  single-hop, adversarial), plus the abstention rate on adversarial questions.
- **Evidence recall**: the fraction of each question's LoCoMo evidence turns
  that made it into the selected context. It scores the policy itself,
  separately from how well the model reads what it was given.
- **System**: prompt/response tokens, average context size, dropped messages,
  per-generation latency, wall time, peak GPU memory, CPU RSS.

## Extending: add a new policy

The framework is designed so a new policy is the *only* thing you write. Define a
class, implement `build_context` (and `observe` if it is stateful), and register
it:

```python
from contextbench.core.types import ContextResult, Message
from contextbench.policies.base import ContextPolicy, register_policy

@register_policy("my_policy")
class MyPolicy(ContextPolicy):
    def observe(self, message: Message) -> None:
        ...  # optional: update incremental state (index, summary, ...)

    def build_context(self, history, query, budget) -> ContextResult:
        selected = ...  # choose messages such that their tokens <= budget
        return ContextResult(
            messages=tuple(selected),
            token_count=self._token_counter.count_messages(selected),
        )

    def reset(self) -> None:
        ...  # clear per-conversation state
```

Then select it with `python benchmark.py policy=my_policy` (add a
`configs/policy/my_policy.yaml`). Datasets, models (backends), and evaluators are
extended the same way via their registries.

## Testing

```bash
pytest                          # fast, hermetic (no model download)
CONTEXTBENCH_SLOW=1 pytest      # + end-to-end integration with the real model
```

The core suite uses a fake token counter, so policy and pipeline logic are tested
without any HuggingFace dependency.

## Roadmap

- [x] Sliding-window policy (Phase 1)
- [x] Summarization policy (heuristic extractive)
- [x] Retrieval policy
- [x] Semantic-change (ΔContext) policy, with topic-segment ranking
- [x] Corrected LoCoMo evaluation (v2), evidence recall, selection-only runs
- [ ] Streaming (`per_session`) evaluation timing

## License

MIT.
