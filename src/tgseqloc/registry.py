"""Extensible component contracts and named implementation registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class Source(Protocol):
    """Parses one precomputed sidecar into a canonical per-frame structure.

    A source reads what another tool already produced; a model that runs
    inference itself is an :class:`OCRModel` or :class:`Segmenter` instead.
    """

    def __call__(self, path: Any, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class OCRModel(Protocol):
    """Recognizes text on frames, returning the canonical frame format.

    ``load`` is separate from construction so that configuration can be
    validated, and missing weights reported, before anything reaches the GPU.
    """

    def load(self) -> None: ...

    def predict(self, image_paths: Sequence[Any]) -> Sequence[Any]: ...

    @property
    def cache_identity(self) -> Mapping[str, Any]: ...


@runtime_checkable
class Segmenter(Protocol):
    """Marks pixels belonging to ephemeral objects."""

    def load(self) -> None: ...

    def predict(self, image_paths: Sequence[Any]) -> Sequence[Any]: ...

    @property
    def cache_identity(self) -> Mapping[str, Any]: ...


@runtime_checkable
class TextDynamics(Protocol):
    """Scores how much of each detection sits on a dynamic object.

    The score stays continuous; turning it into a keep/drop decision is the
    caller's business, so the threshold remains a configuration choice.
    """

    def score(self, frame_text: Any, frame_masks: Any) -> Sequence[float]: ...


@runtime_checkable
class Filter(Protocol):
    """Accepts or rejects a sample."""

    def __call__(self, sample: Mapping[str, Any]) -> bool: ...


@runtime_checkable
class Encoder(Protocol):
    """Maps a batch of modality values to embeddings."""

    def encode(self, batch: Sequence[Any]) -> Any: ...


@runtime_checkable
class Fusion(Protocol):
    """Combines aligned modality embeddings."""

    def fuse(self, embeddings: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class GraphEncoder(Protocol):
    """Encodes node features and graph connectivity."""

    def encode_graph(self, node_features: Any, edge_index: Any) -> Any: ...


@runtime_checkable
class Dataset(Protocol):
    """Indexable training or evaluation sample collection."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


@runtime_checkable
class Miner(Protocol):
    """Selects informative tuples from labels and embeddings."""

    def mine(self, embeddings: Any, labels: Any) -> Any: ...


@runtime_checkable
class Retriever(Protocol):
    """Builds and queries a nearest-neighbour index."""

    def fit(self, embeddings: Any, ids: Sequence[Any]) -> None: ...

    def search(self, queries: Any, top_k: int) -> tuple[Any, Any]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved candidates using richer evidence."""

    def rerank(
        self, query: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
    ) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class Metric(Protocol):
    """Computes a named scalar evaluation result."""

    def compute(self, predictions: Any, targets: Any) -> float: ...


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
            "ocr",
            "segmenter",
            "text_dynamics",
            "filter",
            "encoder",
            "fusion",
            "graph_encoder",
            "dataset",
            "miner",
            "retriever",
            "reranker",
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
