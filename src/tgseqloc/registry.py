"""Named registry of the implementations that configuration selects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


Factory = Callable[..., Any]
T = TypeVar("T", bound=Factory)


class Registry:
    """Registry of factories grouped by component kind.

    Names are case-insensitive and normalized by replacing hyphens with
    underscores. Duplicate registrations fail unless ``replace=True``.
    """

    KINDS = frozenset(
        {
            "source",
            "encoder",
            "fusion",
            "graph_encoder",
            "dataset",
            "miner",
            "retriever",
            "metric",
        }
    )

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Factory]] = {
            kind: {} for kind in self.KINDS
        }

    def register(
        self,
        kind: str,
        name: str | None = None,
        *,
        replace: bool = False,
    ) -> Callable[[T], T]:
        """Return a decorator registering a class or factory."""

        normalized_kind = self._kind(kind)

        def decorator(factory: T) -> T:
            key = self._name(name or factory.__name__)
            entries = self._entries[normalized_kind]
            if key in entries and not replace:
                raise KeyError(
                    f"{normalized_kind} {key!r} is already registered; "
                    "pass replace=True to replace it"
                )
            entries[key] = factory
            return factory

        return decorator

    def add(
        self,
        kind: str,
        name: str,
        factory: T,
        *,
        replace: bool = False,
    ) -> T:
        """Explicitly register and return ``factory``."""

        return self.register(kind, name, replace=replace)(factory)

    def get(self, kind: str, name: str) -> Factory:
        """Resolve a factory, including available names in lookup errors."""

        normalized_kind = self._kind(kind)
        key = self._name(name)
        try:
            return self._entries[normalized_kind][key]
        except KeyError as exc:
            available = self.available(normalized_kind)
            choices = ", ".join(available) if available else "<none>"
            raise KeyError(
                f"unknown {normalized_kind} {name!r}; available: {choices}"
            ) from exc

    def create(self, kind: str, name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Resolve and invoke a registered factory."""

        return self.get(kind, name)(*args, **kwargs)

    def available(self, kind: str) -> tuple[str, ...]:
        """Return sorted registered names for ``kind``."""

        return tuple(sorted(self._entries[self._kind(kind)]))

    def contains(self, kind: str, name: str) -> bool:
        """Return whether ``name`` is registered for ``kind``."""

        normalized_kind = self._kind(kind)
        return self._name(name) in self._entries[normalized_kind]

    def _kind(self, kind: str) -> str:
        normalized = self._name(kind)
        if normalized not in self.KINDS:
            raise KeyError(
                f"unknown registry kind {kind!r}; available: "
                f"{', '.join(sorted(self.KINDS))}"
            )
        return normalized

    @staticmethod
    def _name(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_")
        if not normalized:
            raise ValueError("registry names cannot be empty")
        return normalized


registry = Registry()


def register(
    kind: str, name: str | None = None, *, replace: bool = False
) -> Callable[[T], T]:
    """Register a component in the process-wide default registry."""

    return registry.register(kind, name, replace=replace)
