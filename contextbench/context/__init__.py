"""ΔContext infrastructure.

Treats a conversation as a trajectory of semantic states ``C_0 -> C_1 -> ... ->
C_t`` and measures *when* the context changes enough to justify a memory
operation. This package provides the reusable building blocks — semantic state
construction, change metrics, change accumulation, checkpointing, and the
context controller — that a future ``DeltaContextPolicy`` will compose. The
transformer, tokenizer, and prompt builder are never touched.
"""
