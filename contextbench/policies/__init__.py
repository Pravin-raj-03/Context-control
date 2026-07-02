"""Context-management policies. Every policy inherits from
:class:`~contextbench.policies.base.ContextPolicy` and is selected by name via
the component registry. Phase 1 ships only the sliding-window policy.
"""

from contextbench.policies.base import ContextPolicy, POLICY_REGISTRY
from contextbench.policies.sliding_window import SlidingWindowPolicy
from contextbench.policies.delta_context import DeltaContextPolicy

__all__ = ["ContextPolicy", "POLICY_REGISTRY", "SlidingWindowPolicy", "DeltaContextPolicy"]
