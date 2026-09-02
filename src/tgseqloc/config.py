"""Strict configuration matching the implemented V4RL TGSeqLoc pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when configuration is unknown, ill-typed, or inconsistent."""


DATASET_ADAPTERS = ("v4rl",)
OCR_SOURCES = ("precomputed_paddleocr",)
SCENE_GRAPH_SOURCES = ("external_json",)
TEXT_FILTERS = ("confidence",)
FUSION_METHODS = ("text_nodes",)
CONNECTION_STRATEGIES = (
    "overlap_nearest",
    "overlap",
    "nearest",
    "fully_connected",
)
CACHE_POLICIES = ("reuse_if_compatible", "require_existing", "rebuild")
GRAPH_ENCODERS = ("gat",)
RETRIEVERS = ("faiss_cosine",)
RERANKERS: tuple[str, ...] = ()


@dataclass(slots=True)
class DatasetConfig:
    """V4RL paths, sequence roles, and temporal split controls."""

    adapter: str = "v4rl"
    root: Path = Path("data/V4RL")
    sequences: list[str] = field(default_factory=lambda: ["seq1", "seq2"])
    ocr_root_template: str = "data/ocr/{sequence}/chunk_{chunk}"
    scene_graph_root_template: str = "data/scene_graphs/{sequence}"
    gt_path: Path = Path("data/V4RL/gt/gt_shop_street_1.txt")
    prepared_root: Path = Path("data/prepared")
    reference_sequence: str = "seq1"
    query_sequence: str = "seq2"
    test_ratio: float = 0.2
    validation_ratio: float = 0.1
    interpolate_gt: bool = False
    chunk_size: int = 200


@dataclass(slots=True)
class SourceConfig:
    """Implemented external data providers."""

    ocr: str = "precomputed_paddleocr"
    scene_graph: str = "external_json"


# Backward-compatible public name from the initial package foundation.
SourcesConfig = SourceConfig


@dataclass(slots=True)
class PreprocessConfig:
    """Text filtering, encoding, graph fusion, and batching settings."""

    text_filter: str = "confidence"
    text_encoder_backend: str = "multilingual_e5"
    text_encoder: str = "intfloat/multilingual-e5-small"
    revision: str | None = None
    fusion: str = "text_nodes"
    connection_strategy: str = "overlap_nearest"
    connection_k: int = 1
    ocr_confidence_threshold: float = 0.0
    encoder_batch_size: int = 128
    frame_batch_size: int = 256
    gt_tolerance_ns: int = 50_000_000


@dataclass(slots=True)
class CacheConfig:
    """Prepared-artifact reuse behavior."""

    policy: str = "reuse_if_compatible"


@dataclass(slots=True)
class ModelConfig:
    """Arguments implemented by :class:`GATGraphEncoder`.

    Vocabulary sizes are inferred from the preparation manifest rather than
    configured here.
    """

    graph_encoder: str = "gat"
    in_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 2
    proj_dim: int = 256
    node_emb_dim: int = 64
    edge_emb_dim: int = 64
    edge_cont_dim: int = 10
    text_emb_dim: int = 384
    dropout: float = 0.1
    heads: int = 4
    use_text_nodes: bool = True
    use_edge_geometry: bool = True


@dataclass(slots=True)
class TrainingConfig:
    """Triplet training, hard-negative mining, and early stopping.

    ``patience=0`` is valid and stops at the first non-improving epoch.
    """

    epochs: int = 30
    patience: int = 7
    batch_size: int = 16
    infer_batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    margin: float = 0.3
    negatives_per_query: int = 2
    hard_search_depth: int = 256
    max_grad_norm: float = 5.0
    miner: str = "hard_negative"
    recall_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    early_stopping_metric: str = "R@5"
    resume_from: Path | None = None


@dataclass(slots=True)
class RetrievalConfig:
    """FAISS cosine retrieval and optional reranking."""

    retriever: str = "faiss_cosine"
    metric: str = "recall_at_k"
    recall_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    reranker: str | None = None
    top_k: int = 20


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42
    num_workers: int = 4


@dataclass(slots=True)
class OutputConfig:
    root: Path = Path("outputs")
    experiment_name: str = "v4rl-gat"


@dataclass(slots=True)
class AppConfig:
    """Complete nested TGSeqLoc configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> AppConfig:
        """Validate implemented names, ranges, and cross-field constraints."""

        _choice("dataset.adapter", self.dataset.adapter, DATASET_ADAPTERS)
        _choice("sources.ocr", self.sources.ocr, OCR_SOURCES)
        _choice(
            "sources.scene_graph", self.sources.scene_graph, SCENE_GRAPH_SOURCES
        )
        _choice(
            "preprocess.text_filter", self.preprocess.text_filter, TEXT_FILTERS
        )
        _choice(
            "preprocess.text_encoder_backend",
            self.preprocess.text_encoder_backend,
            ("multilingual_e5",),
        )
        _choice("preprocess.fusion", self.preprocess.fusion, FUSION_METHODS)
        _choice(
            "preprocess.connection_strategy",
            self.preprocess.connection_strategy,
            CONNECTION_STRATEGIES,
        )
        _choice("cache.policy", self.cache.policy, CACHE_POLICIES)
        _choice("model.graph_encoder", self.model.graph_encoder, GRAPH_ENCODERS)
        _choice(
            "retrieval.retriever", self.retrieval.retriever, RETRIEVERS
        )
        _choice("training.miner", self.training.miner, ("hard_negative",))
        _choice("retrieval.metric", self.retrieval.metric, ("recall_at_k",))
        if self.retrieval.reranker is not None:
            _choice("retrieval.reranker", self.retrieval.reranker, RERANKERS)

        if not self.dataset.sequences:
            raise ConfigError("dataset.sequences must not be empty")
        if any(not sequence.strip() for sequence in self.dataset.sequences):
            raise ConfigError("dataset.sequences cannot contain empty names")
        if len(set(self.dataset.sequences)) != len(self.dataset.sequences):
            raise ConfigError("dataset.sequences must contain unique names")
        for field_name, sequence in {
            "dataset.reference_sequence": self.dataset.reference_sequence,
            "dataset.query_sequence": self.dataset.query_sequence,
        }.items():
            if sequence not in self.dataset.sequences:
                raise ConfigError(
                    f"{field_name} must be one of dataset.sequences; available: "
                    f"{', '.join(self.dataset.sequences)}"
                )
        if self.dataset.reference_sequence == self.dataset.query_sequence:
            raise ConfigError(
                "dataset.reference_sequence and dataset.query_sequence must differ"
            )
        float_values = {
            "dataset.test_ratio": self.dataset.test_ratio,
            "dataset.validation_ratio": self.dataset.validation_ratio,
            "preprocess.ocr_confidence_threshold": (
                self.preprocess.ocr_confidence_threshold
            ),
            "model.dropout": self.model.dropout,
            "training.learning_rate": self.training.learning_rate,
            "training.weight_decay": self.training.weight_decay,
            "training.margin": self.training.margin,
            "training.max_grad_norm": self.training.max_grad_norm,
        }
        for name, value in float_values.items():
            _require_finite(name, value)
        if not 0.0 < self.dataset.test_ratio < 1.0:
            raise ConfigError("dataset.test_ratio must be in (0, 1)")
        if not 0.0 <= self.dataset.validation_ratio < 1.0:
            raise ConfigError("dataset.validation_ratio must be in [0, 1)")
        if self.dataset.test_ratio + self.dataset.validation_ratio >= 1.0:
            raise ConfigError(
                "dataset.test_ratio + dataset.validation_ratio must be less than 1"
            )

        positive_ints = {
            "dataset.chunk_size": self.dataset.chunk_size,
            "preprocess.connection_k": self.preprocess.connection_k,
            "preprocess.encoder_batch_size": self.preprocess.encoder_batch_size,
            "preprocess.frame_batch_size": self.preprocess.frame_batch_size,
            "preprocess.gt_tolerance_ns": self.preprocess.gt_tolerance_ns,
            "model.in_dim": self.model.in_dim,
            "model.hidden_dim": self.model.hidden_dim,
            "model.n_layers": self.model.n_layers,
            "model.proj_dim": self.model.proj_dim,
            "model.node_emb_dim": self.model.node_emb_dim,
            "model.edge_emb_dim": self.model.edge_emb_dim,
            "model.edge_cont_dim": self.model.edge_cont_dim,
            "model.text_emb_dim": self.model.text_emb_dim,
            "model.heads": self.model.heads,
            "training.epochs": self.training.epochs,
            "training.batch_size": self.training.batch_size,
            "training.infer_batch_size": self.training.infer_batch_size,
            "training.negatives_per_query": self.training.negatives_per_query,
            "training.hard_search_depth": self.training.hard_search_depth,
            "retrieval.top_k": self.retrieval.top_k,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ConfigError(f"{name} must be greater than zero")
        if self.training.patience < 0:
            raise ConfigError("training.patience must be non-negative")
        if self.runtime.num_workers < 0:
            raise ConfigError("runtime.num_workers must be non-negative")
        _validate_device(self.runtime.device)
        if not 0.0 <= self.preprocess.ocr_confidence_threshold <= 1.0:
            raise ConfigError(
                "preprocess.ocr_confidence_threshold must be in [0, 1]"
            )
        if not 0.0 <= self.model.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        for name, value in {
            "training.learning_rate": self.training.learning_rate,
            "training.margin": self.training.margin,
            "training.max_grad_norm": self.training.max_grad_norm,
        }.items():
            if value <= 0:
                raise ConfigError(f"{name} must be greater than zero")
        if self.training.weight_decay < 0:
            raise ConfigError("training.weight_decay must be non-negative")

        _validate_recalls("training.recall_values", self.training.recall_values)
        _validate_recalls("retrieval.recall_values", self.retrieval.recall_values)
        available_metrics = tuple(f"R@{k}" for k in self.training.recall_values)
        _choice(
            "training.early_stopping_metric",
            self.training.early_stopping_metric,
            available_metrics,
        )
        for name, value in {
            "preprocess.text_encoder": self.preprocess.text_encoder,
            "output.experiment_name": self.output.experiment_name,
        }.items():
            if not value.strip():
                raise ConfigError(f"{name} cannot be empty")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-friendly nested mapping."""

        return _plain(asdict(self))


T = TypeVar("T")


def load_config(path: str | Path) -> AppConfig:
    """Load a strict :class:`AppConfig` from YAML."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load configuration {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return _dataclass_from_mapping(AppConfig, raw, "").validate()


def dump_config(config: AppConfig, path: str | Path) -> None:
    """Validate and write stable, human-readable YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.validate().to_dict(), sort_keys=False),
        encoding="utf-8",
    )


def _choice(name: str, value: str, available: tuple[str, ...]) -> None:
    if value not in available:
        choices = ", ".join(available) if available else "<none>"
        raise ConfigError(f"unsupported {name} {value!r}; available: {choices}")


def _require_finite(name: str, value: float) -> None:
    try:
        finite = math.isfinite(value)
    except TypeError as exc:
        raise ConfigError(f"{name} must be a finite number") from exc
    if not finite:
        raise ConfigError(f"{name} must be finite")


def _validate_device(value: str) -> None:
    if value == "auto":
        return
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            "runtime.device must be 'auto' or a device accepted by torch.device"
        )
    import torch

    try:
        torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(
            f"invalid runtime.device {value!r}; use 'auto' or a value accepted "
            "by torch.device (for example 'cpu', 'cuda', or 'cuda:0')"
        ) from exc


def _validate_recalls(name: str, values: list[int]) -> None:
    if not values:
        raise ConfigError(f"{name} must not be empty")
    if any(value <= 0 for value in values):
        raise ConfigError(f"{name} values must be greater than zero")
    if values != sorted(set(values)):
        raise ConfigError(f"{name} must be sorted and contain unique values")


def _dataclass_from_mapping(cls: type[T], raw: Mapping[str, Any], prefix: str) -> T:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{prefix or 'configuration'} must be a mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        location = prefix or "configuration"
        available = ", ".join(sorted(known))
        raise ConfigError(
            f"unknown key(s) in {location}: {', '.join(unknown)}; "
            f"available: {available}"
        )
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for item in fields(cls):
        if item.name in raw:
            item_path = f"{prefix}.{item.name}" if prefix else item.name
            values[item.name] = _convert(raw[item.name], hints[item.name], item_path)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {prefix or 'configuration'}: {exc}") from exc


def _convert(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Any:
        return value
    if annotation is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a path string")
        return Path(value)
    if is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        return _dataclass_from_mapping(annotation, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        return [
            _convert(item, args[0], f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        candidates = [candidate for candidate in args if candidate is not type(None)]
        for candidate in candidates:
            try:
                return _convert(value, candidate, path)
            except ConfigError:
                pass
        raise ConfigError(f"{path} has an invalid value")
    if annotation is bool:
        if type(value) is not bool:
            raise ConfigError(f"{path} must be a boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise ConfigError(f"{path} must be an integer")
        return value
    if annotation is float:
        if type(value) not in (int, float):
            raise ConfigError(f"{path} must be a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
        return value
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value
