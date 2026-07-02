"""Two-tier orchestration.

``BenchmarkEngine`` is the sweep controller: it owns the resolved config, loads
the model and tokenizer exactly once, and drives runs across samples, budgets,
and policies, aggregating run-level metrics.

``ConversationRunner`` is the per-conversation orchestrator: it owns the
canonical history, replays turns, and at each evaluation checkpoint asks the
active policy for context. It is deliberately ignorant of *how* context is
selected.
"""
