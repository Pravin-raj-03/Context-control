"""ContextBench: a modular framework for benchmarking context-management
policies for conversational transformer models under fixed token budgets.

The transformer, tokenizer, prompt format, dataset, and evaluation are held
fixed; the only swappable component on the critical path is the
:class:`~contextbench.policies.base.ContextPolicy`.
"""

__version__ = "0.1.0"
