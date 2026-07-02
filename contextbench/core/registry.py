"""A tiny, reusable registry for swap-by-name components.

Each extensible subsystem (policies, models, datasets, evaluators) creates one
:class:`Registry` instance next to its abstract base class and exposes a
``register`` decorator. Configuration then selects an implementation purely by
name, so adding a new context policy is a matter of defining a class and
decorating it — no edits to any core dispatch code. This is what lets other
researchers extend ContextBench without touching the framework.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name-to-class registry with decorator registration and construction.

    Names are matched case-insensitively. Duplicate registration under the same
    name raises, so silent shadowing of one implementation by another cannot
    happen.
    """

    def __init__(self, kind: str) -> None:
        """Create an empty registry.

        Args:
            kind: Human-readable category label used in error messages
                (e.g. ``"policy"``).
        """
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a class decorator that registers the class under ``name``.

        Args:
            name: The lookup key. Case-insensitive.

        Raises:
            ValueError: If ``name`` is empty or already registered.
        """
        key = name.strip().lower()
        if not key:
            raise ValueError(f"Cannot register a {self._kind} under an empty name.")

        def decorator(cls: type[T]) -> type[T]:
            if key in self._entries:
                raise ValueError(
                    f"{self._kind} '{name}' is already registered to "
                    f"{self._entries[key].__name__}."
                )
            self._entries[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``.

        Raises:
            KeyError: If no implementation is registered under ``name``.
        """
        key = name.strip().lower()
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(
                f"Unknown {self._kind} '{name}'. Available: {self.available()}."
            ) from None

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        """Instantiate the class registered under ``name`` with the given args."""
        return self.get(name)(*args, **kwargs)

    def available(self) -> list[str]:
        """Return the sorted list of registered names."""
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, entries={self.available()})"
