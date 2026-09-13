"""Datasets, encoding helpers, and the TGSeqLoc triplet trainer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from tgseqloc.evaluation import evaluate_retrieval
from tgseqloc.models import EdgeAttrNormalizer, GATGraphEncoder

from .mining import mine_hard_negatives


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    elif not isinstance(value, (Mapping, str, bytes, Path)) and hasattr(
        value, "__dict__"
    ):
        value = vars(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _get(config: Mapping[str, Any], *names: str, default=None):
    for name in names:
        value: Any = config
        found = True
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found:
            return value
    return default


def _load_json(value: str | Path | Mapping, base: Path | None = None) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if base is not None and not path.is_absolute():
        path = base / path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location="cpu")


def load_graph(
    path: str | Path,
    edge_normalizer: EdgeAttrNormalizer | None = None,
    text_emb_dim: int | None = None,
    *,
    edge_attr_dim: int | None = None,
) -> Data:
    """Load one prepared PyG graph without relying on preprocessing modules."""

    graph = _torch_load(path)
    if isinstance(graph, Mapping):
        graph = Data(**dict(graph))
    if not isinstance(graph, Data):
        raise TypeError(f"{path} does not contain a torch_geometric Data object")
    graph = graph.clone()
    if getattr(graph, "x", None) is None:
        raise ValueError(f"{path} has no node features")
    graph.x = graph.x.float()
    if getattr(graph, "edge_index", None) is None:
        graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    graph.edge_index = graph.edge_index.long().reshape(2, -1)
    edge_count = int(graph.edge_index.shape[1])
    if edge_attr_dim is None and edge_normalizer is not None:
        edge_attr_dim = edge_normalizer.feature_dim
    edge_attr = getattr(graph, "edge_attr", None)
    if edge_attr is None:
        graph.edge_attr = torch.empty((edge_count, edge_attr_dim or 0))
    else:
        graph.edge_attr = edge_attr.float()
        if graph.edge_attr.ndim == 1:
            graph.edge_attr = graph.edge_attr.reshape(edge_count, -1)
        if graph.edge_attr.shape[0] != edge_count:
            raise ValueError(f"{path} has inconsistent edge_index and edge_attr")
    if edge_attr_dim is not None and graph.edge_attr.shape[1] != edge_attr_dim:
        raise ValueError(
            f"{path} has {graph.edge_attr.shape[1]} edge features; "
            f"expected {edge_attr_dim}"
        )
    if edge_normalizer is not None:
        graph.edge_attr = edge_normalizer.transform(graph.edge_attr)
    for name in ("node_class", "edge_label"):
        value = getattr(graph, name, None)
        if value is not None:
            setattr(graph, name, value.long().reshape(-1))
    if text_emb_dim is not None and getattr(graph, "text_emb", None) is not None:
        expected = (graph.num_nodes, int(text_emb_dim))
        if tuple(graph.text_emb.shape) != expected:
            raise ValueError(f"{path} text_emb must have shape {expected}")
        graph.text_emb = graph.text_emb.float()
    return graph


class GraphPathDataset(Dataset):
    """Lazy dataset over serialized graph paths."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        edge_normalizer: EdgeAttrNormalizer | None = None,
        text_emb_dim: int | None = None,
        edge_attr_dim: int | None = None,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.edge_normalizer = edge_normalizer
        self.text_emb_dim = text_emb_dim
        self.edge_attr_dim = edge_attr_dim

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Data:
        return load_graph(
            self.paths[index],
            self.edge_normalizer,
            self.text_emb_dim,
            edge_attr_dim=self.edge_attr_dim,
        )


class TripletGraphDataset(Dataset):
    """Yield query, positive, and a fixed number of negative graphs."""

    def __init__(
        self,
        database_paths: Sequence[str | Path],
        query_paths: Sequence[str | Path],
        query_indices: Sequence[int],
        positives: Mapping[int, Sequence[int]] | Mapping[str, Sequence[int]],
        negatives_per_query: int,
        edge_normalizer: EdgeAttrNormalizer | None = None,
        text_emb_dim: int | None = None,
        seed: int = 0,
        edge_attr_dim: int | None = None,
    ) -> None:
        if negatives_per_query < 1:
            raise ValueError("negatives_per_query must be positive")
        self.database_paths = [Path(path) for path in database_paths]
        self.query_paths = [Path(path) for path in query_paths]
        self.positives = {
            int(index): [int(item) for item in values]
            for index, values in positives.items()
        }
        self.negatives_per_query = int(negatives_per_query)
        self.edge_normalizer = edge_normalizer
        self.text_emb_dim = text_emb_dim
        self.edge_attr_dim = edge_attr_dim
        self.seed = int(seed)
        self.epoch = 0
        self.hard_negatives: dict[int, list[int]] = {}
        self.query_indices = []
        for query_index in map(int, query_indices):
            positive = set(self.positives.get(query_index, ()))
            if positive and any(
                index not in positive for index in range(len(self.database_paths))
            ):
                self.query_indices.append(query_index)

    def __len__(self) -> int:
        return len(self.query_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_hard_negatives(self, values: Mapping[int, Sequence[int]]) -> None:
        self.hard_negatives = {
            int(index): [int(item) for item in items]
            for index, items in values.items()
        }

    def _load(self, path: Path) -> Data:
        return load_graph(
            path,
            self.edge_normalizer,
            self.text_emb_dim,
            edge_attr_dim=self.edge_attr_dim,
        )

    def __getitem__(self, local_index: int) -> list[Data]:
        query_index = self.query_indices[local_index]
        rng = random.Random(self.seed + self.epoch * 100_003 + query_index)
        positive_index = rng.choice(self.positives[query_index])
        negatives = [
            item
            for item in self.hard_negatives.get(query_index, ())
            if item not in self.positives[query_index]
        ]
        if not negatives:
            negatives = [
                index
                for index in range(len(self.database_paths))
                if index not in self.positives[query_index]
            ]
            rng.shuffle(negatives)
        if len(negatives) < self.negatives_per_query:
            base = negatives.copy()
            negatives.extend(
                base[offset % len(base)]
                for offset in range(self.negatives_per_query - len(negatives))
            )
        negatives = negatives[: self.negatives_per_query]
        return [
            self._load(self.query_paths[query_index]),
            self._load(self.database_paths[positive_index]),
            *(self._load(self.database_paths[index]) for index in negatives),
        ]


def collate_graphs(items) -> Batch:
    """Collate either graphs or per-query graph lists into one PyG batch."""

    flat = [
        graph
        for item in items
        for graph in (item if isinstance(item, (list, tuple)) else (item,))
    ]
    return Batch.from_data_list(flat)


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed every generator a run draws from, and return one for DataLoader.

    Called before the model is built, not after: weight initialization draws
    from the global generator, so seeding inside train() left the starting
    point random.

    Seeding alone is not enough on a GPU. Graph aggregation sums messages with
    atomic adds whose order varies between runs, so the same model on the same
    input returns different numbers. ``deterministic`` pins that down; the
    cuBLAS workspace variable has to be set before the handle is created,
    which is why it is set here rather than left to the caller.
    """

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_graph_loader(
    paths: Sequence[str | Path],
    edge_normalizer: EdgeAttrNormalizer | None = None,
    text_emb_dim: int | None = None,
    *,
    edge_attr_dim: int | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = False,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = GraphPathDataset(
        paths, edge_normalizer, text_emb_dim, edge_attr_dim=edge_attr_dim
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Batch.from_data_list,
        pin_memory=torch.device(device).type == "cuda",
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def encode_paths(
    model: nn.Module,
    paths: Sequence[str | Path],
    edge_normalizer: EdgeAttrNormalizer | None = None,
    text_emb_dim: int | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    num_workers: int = 0,
    *,
    edge_attr_dim: int | None = None,
    as_numpy: bool = True,
):
    device = torch.device(device)
    loader = make_graph_loader(
        paths,
        edge_normalizer,
        text_emb_dim,
        edge_attr_dim=edge_attr_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    model.eval()
    outputs = [model(batch.to(device)).detach().cpu() for batch in loader]
    out_dim = int(getattr(model, "out_dim", 0))
    result = torch.cat(outputs) if outputs else torch.empty((0, out_dim))
    return result.numpy().astype(np.float32) if as_numpy else result


def fit_edge_normalizer(
    paths: Sequence[str | Path],
    edge_attr_dim: int,
    log_indices: Sequence[int] | None = None,
    *,
    exclude_text_edges: bool = False,
) -> EdgeAttrNormalizer:
    """Standardization statistics for edge attributes.

    ``exclude_text_edges`` fits them on scene relations alone. The additive text
    form never passes text edges to the encoder, and if their geometry still
    entered these statistics a frame without text would be normalized
    differently from the text-free pipeline.
    """

    normalizer = EdgeAttrNormalizer(log_indices, feature_dim=edge_attr_dim)
    for path in paths:
        graph = load_graph(path, edge_attr_dim=edge_attr_dim)
        attributes = graph.edge_attr
        text_edge = getattr(graph, "is_text_edge", None)
        if exclude_text_edges and attributes is not None and text_edge is not None:
            attributes = attributes[~text_edge.bool().reshape(-1)]
        normalizer.update(attributes)
    return normalizer.finalize()


class Trainer:
    """End-to-end graph descriptor trainer for prepared TGSeqLoc data."""

    def __init__(
        self,
        config: object,
        data_root: str | Path,
        manifest: Mapping | str | Path | None = None,
        split: Mapping | str | Path | None = None,
        *,
        model: GATGraphEncoder | None = None,
        device: str | torch.device | None = None,
        miner=mine_hard_negatives,
        retriever_factory=None,
        metric_factory=None,
    ) -> None:
        self.config = _plain(config)
        if not isinstance(self.config, Mapping):
            raise TypeError("config must resolve to a mapping or object with attributes")
        self.config = dict(self.config)
        self.data_root = Path(data_root)
        manifest_value = (
            self.data_root / "manifest.json" if manifest is None else manifest
        )
        self.manifest = _load_json(manifest_value, self.data_root)
        split_value = (
            _get(self.config, "split", "data.split") if split is None else split
        )
        if split_value is None:
            split_value = self.manifest.get("temporal_split_path")
        if split_value is None:
            for name in ("split.json", "splits.json"):
                if (self.data_root / name).exists():
                    split_value = self.data_root / name
                    break
        if split_value is None:
            raise ValueError("a prepared split mapping or split JSON path is required")
        self.split = _load_json(split_value, self.data_root)
        self._check_split_matches_config()
        self._prepare_split()
        self._validate_manifest_graph_records()
        requested_device = device or _get(
            self.config, "runtime.device", "device", default="cpu"
        )
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.model = model or self._build_model()
        self.model.to(self.device)
        self.miner = miner
        self.retriever_factory = retriever_factory
        self.metric_factory = metric_factory
        self.normalizer: EdgeAttrNormalizer | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.start_epoch = 1
        self.best_metric = float("-inf")
        self.run_dir = self._resolve_run_dir()

    def _check_split_matches_config(self) -> None:
        """Refuse a split written for another protocol.

        Preparation writes the split into the prepared root, and one root serves
        several protocols -- a single time cut and every fold of the block cross
        validation. Training that silently read a split left there by another
        protocol would report numbers for a test set nobody asked for.
        """

        folds_path = _get(self.config, "dataset.split_folds_path", default="")
        recorded = self.split.get("split_fold")
        if folds_path:
            expected = int(_get(self.config, "dataset.split_fold", default=-1))
            if recorded is None or int(recorded) != expected:
                raise ValueError(
                    f"the prepared split is for fold {recorded}, the configuration "
                    f"asks for fold {expected}; run prepare with this configuration"
                )
        elif recorded is not None:
            raise ValueError(
                f"the prepared split is fold {recorded} of a cross validation, the "
                "configuration asks for a single split; run prepare with this configuration"
            )

    def _paths(self, key: str, fallback_dir: str) -> list[Path]:
        values = self.split.get(key)
        if values is None:
            values = sorted((self.data_root / fallback_dir).glob("*.pt"))
        return [
            path if (path := Path(value)).is_absolute() else self.data_root / path
            for value in values
        ]

    def _prepare_split(self) -> None:
        self.database_paths = self._paths("database_paths", "seq1")
        self.query_paths = self._paths("query_paths", "seq2")
        raw_positives = self.split.get("positives", {})
        self.positives = {
            int(index): [int(item) for item in values]
            for index, values in raw_positives.items()
        }
        self.train_query_indices = [
            int(index) for index in self.split.get("train_query_indices", ())
        ]
        self.validation_query_indices = [
            int(index)
            for index in self.split.get(
                "validation_query_indices",
                self.split.get("val_query_indices", ()),
            )
        ]
        self.test_query_indices = [
            int(index) for index in self.split.get("test_query_indices", ())
        ]
        if not self.database_paths or not self.query_paths:
            raise ValueError("prepared split contains no database or query graphs")
        if not self.train_query_indices:
            raise ValueError("prepared split contains no training queries")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _manifest_records_by_path(self) -> dict[Path, Mapping[str, Any]]:
        raw_records = self.manifest.get("graph_records")
        if raw_records is None:
            return {}
        normalized: list[Mapping[str, Any]] = []
        if isinstance(raw_records, Mapping):
            for raw_path, value in raw_records.items():
                if isinstance(value, Mapping):
                    record = dict(value)
                    record.setdefault("path", raw_path)
                else:
                    record = {"path": raw_path, "sha256": value}
                normalized.append(record)
        elif isinstance(raw_records, Sequence) and not isinstance(
            raw_records, (str, bytes)
        ):
            if not all(isinstance(record, Mapping) for record in raw_records):
                raise ValueError("manifest graph_records entries must be mappings")
            normalized = [dict(record) for record in raw_records]
        else:
            raise ValueError("manifest graph_records must be a mapping or sequence")

        records: dict[Path, Mapping[str, Any]] = {}
        for record in normalized:
            raw_path = next(
                (
                    record[key]
                    for key in ("path", "graph_path", "relative_path")
                    if record.get(key) is not None
                ),
                None,
            )
            if raw_path is None:
                raise ValueError("manifest graph record is missing its path")
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.data_root / path
            canonical = path.resolve()
            if canonical in records:
                raise ValueError(f"duplicate manifest graph record for {path}")
            records[canonical] = record
        return records

    def _validate_manifest_graph_records(self) -> None:
        """Validate declared prepared graphs once before any model work."""

        if "graph_records" not in self.manifest:
            return
        records = self._manifest_records_by_path()
        expected_paths = {
            path.resolve() for path in self.database_paths + self.query_paths
        }
        missing = sorted(str(path) for path in expected_paths - records.keys())
        if missing:
            preview = ", ".join(missing[:3])
            suffix = " ..." if len(missing) > 3 else ""
            raise ValueError(
                f"split graph path is absent from manifest graph_records: "
                f"{preview}{suffix}"
            )
        for path in sorted(expected_paths, key=str):
            record = records[path]
            if not path.is_file():
                raise ValueError(f"manifest graph file does not exist: {path}")
            expected_sha = record.get("sha256", record.get("file_sha256"))
            if not isinstance(expected_sha, str) or not expected_sha:
                raise ValueError(
                    f"manifest graph record has no SHA256 for {path}"
                )
            actual_sha = self._file_sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(f"graph file SHA256 mismatch for {path}")
            expected_fingerprint = record.get(
                "expected_fingerprint",
                record.get(
                    "preprocess_fingerprint",
                    record.get("fingerprint", self.fingerprint),
                ),
            )
            graph = _torch_load(path)
            actual_fingerprint = (
                graph.get("preprocess_fingerprint")
                if isinstance(graph, Mapping)
                else getattr(graph, "preprocess_fingerprint", None)
            )
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    f"graph preprocessing fingerprint mismatch for {path}: "
                    f"expected {expected_fingerprint!r}, got {actual_fingerprint!r}"
                )

    def _manifest_int(self, *names: str, default=None) -> int:
        for name in names:
            if name in self.manifest:
                return int(self.manifest[name])
        if default is None:
            raise KeyError(f"manifest is missing one of: {', '.join(names)}")
        return int(default)

    @property
    def edge_attr_dim(self) -> int:
        return self._manifest_int("edge_attr_dim", "edge_feature_dim", default=10)

    @property
    def text_emb_dim(self) -> int:
        return self._manifest_int(
            "text_embedding_dim", "text_emb_dim", default=384
        )

    @property
    def fingerprint(self) -> str:
        explicit = self.manifest.get(
            "preprocess_fingerprint", self.manifest.get("fingerprint")
        )
        if explicit is not None:
            return str(explicit)
        payload = json.dumps(_plain(self.manifest), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_model(self) -> GATGraphEncoder:
        model_config = _get(self.config, "model", default={})
        model_config = dict(model_config) if isinstance(model_config, Mapping) else {}
        node_class_count = self.manifest.get(
            "num_obj_classes", self.manifest.get("num_node_classes")
        )
        edge_class_count = self.manifest.get("num_edge_classes")
        defaults = {
            "in_dim": self._manifest_int(
                "node_feature_dim", "node_attr_dim", default=4
            ),
            "hidden_dim": 256,
            "n_layers": 2,
            "proj_dim": 256,
            "num_node_classes": (
                max(1, int(node_class_count)) if node_class_count is not None else None
            ),
            "node_emb_dim": 64,
            "num_edge_classes": (
                max(1, int(edge_class_count)) if edge_class_count is not None else None
            ),
            "edge_emb_dim": 64,
            "edge_cont_dim": self.edge_attr_dim,
            "dropout": 0.1,
            "heads": 4,
            "use_text_nodes": bool(
                self.manifest.get("uses_text_nodes", model_config.get("use_text_nodes", True))
            ),
            "text_emb_dim": self.text_emb_dim,
            "use_edge_geometry": True,
        }
        aliases = {
            "layers": "n_layers",
            "graph_layers": "n_layers",
            "embedding_dim": "proj_dim",
        }
        for key, value in model_config.items():
            defaults[aliases.get(key, key)] = value
        allowed = set(GATGraphEncoder.__init__.__annotations__) - {"return"}
        return GATGraphEncoder(**{key: value for key, value in defaults.items() if key in allowed})

    def _training_value(self, name: str, default):
        runtime_names = {
            "seed": "runtime.seed",
            "num_workers": "runtime.num_workers",
            "deterministic": "runtime.deterministic",
        }
        names = [f"training.{name}", name]
        if name in runtime_names:
            names.insert(0, runtime_names[name])
        return _get(self.config, *names, default=default)

    def _resolve_run_dir(self) -> Path:
        configured = _get(
            self.config,
            "output_dir",
            "training.output_dir",
            "run_dir",
            default=None,
        )
        if configured is None:
            configured = (
                Path(_get(self.config, "output.root", default="outputs"))
                / str(
                    _get(
                        self.config,
                        "output.experiment_name",
                        default="v4rl-baseline",
                    )
                )
            )
        path = Path(configured).expanduser()
        # OutputConfig.root is project/cwd-relative, not prepared-data-relative.
        return path if path.is_absolute() else Path.cwd() / path

    def _normalizer_paths(self) -> list[Path]:
        return self.database_paths + [
            self.query_paths[index] for index in self.train_query_indices
        ]

    def _checkpoint(self, epoch: int, metric: float, metrics: Mapping) -> dict:
        assert self.optimizer is not None and self.normalizer is not None
        return {
            "format_version": 1,
            "model_init_args": dict(self.model.init_args),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": int(epoch),
            "best_metric": float(self.best_metric),
            "metric": float(metric),
            "metrics": dict(metrics),
            "resolved_config": _plain(self.config),
            "manifest": _plain(self.manifest),
            "manifest_fingerprint": self.fingerprint,
            "normalizer_state": self.normalizer.state_dict(),
        }

    @staticmethod
    def _save_checkpoint(path: Path, checkpoint: Mapping) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(checkpoint), temporary)
        temporary.replace(path)

    @staticmethod
    def _save_json(path: Path, value: Mapping) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_plain(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _validate_checkpoint_compatibility(self, checkpoint: Mapping) -> None:
        if checkpoint.get("manifest_fingerprint") != self.fingerprint:
            raise ValueError("checkpoint manifest fingerprint does not match prepared data")
        expected = dict(self.model.init_args)
        actual = checkpoint.get("model_init_args")
        if actual != expected:
            raise ValueError(
                f"checkpoint model is incompatible: expected {expected}, got {actual}"
            )

    def resume(self, checkpoint_path: str | Path) -> dict:
        checkpoint = _torch_load(checkpoint_path)
        self._validate_checkpoint_compatibility(checkpoint)
        required = {
            "model_state_dict",
            "normalizer_state",
            "optimizer_state_dict",
            "epoch",
        }
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                "training checkpoint is incomplete; missing " + ", ".join(missing)
            )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.normalizer = EdgeAttrNormalizer.from_state_dict(
            checkpoint["normalizer_state"]
        )
        if self.optimizer is None:
            self.optimizer = self._make_optimizer()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = float(checkpoint.get("best_metric", float("-inf")))
        return checkpoint

    def load_checkpoint_for_evaluation(
        self,
        checkpoint: str | Path = "best",
    ) -> dict:
        """Load model and normalizer state without constructing an optimizer.

        ``checkpoint`` may be ``"best"``, ``"last"``, or an explicit path.
        Compatibility checks are identical to training resume checks, while
        optimizer and epoch state are deliberately optional.
        """

        if str(checkpoint) in {"best", "last"}:
            checkpoint_path = self.run_dir / f"{checkpoint}_model.pth"
        else:
            checkpoint_path = Path(checkpoint)
        payload = _torch_load(checkpoint_path)
        self._validate_checkpoint_compatibility(payload)
        required = {"model_state_dict", "normalizer_state"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(
                "evaluation checkpoint is incomplete; missing " + ", ".join(missing)
            )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.normalizer = EdgeAttrNormalizer.from_state_dict(
            payload["normalizer_state"]
        )
        self.model.eval()
        return payload

    def _make_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self._training_value("learning_rate", 1e-4)),
            weight_decay=float(self._training_value("weight_decay", 1e-4)),
        )

    def _encode(self, paths: Sequence[Path]):
        assert self.normalizer is not None
        return encode_paths(
            self.model,
            paths,
            self.normalizer,
            self.text_emb_dim,
            self.device,
            int(self._training_value("infer_batch_size", 128)),
            int(self._training_value("num_workers", 0)),
            edge_attr_dim=self.edge_attr_dim,
        )

    def evaluate(
        self,
        query_indices: Sequence[int],
        recall_values: Sequence[int] | None = None,
        *,
        search_k: int | None = None,
    ) -> tuple[dict, np.ndarray]:
        database = self._encode(self.database_paths)
        queries = self._encode([self.query_paths[index] for index in query_indices])
        recalls = (
            list(recall_values)
            if recall_values is not None
            else self._training_value("recall_values", (1, 5, 10))
        )
        kwargs = {}
        if self.retriever_factory is not None:
            kwargs["retriever_factory"] = self.retriever_factory
        if self.metric_factory is not None:
            kwargs["metric_factory"] = self.metric_factory
        return evaluate_retrieval(
            database,
            queries,
            self.positives,
            query_indices,
            recalls,
            search_k=search_k,
            **kwargs,
        )

    def evaluate_test(
        self,
        checkpoint: str | Path | None = "best",
        *,
        write_results: bool = True,
    ) -> tuple[dict, np.ndarray]:
        """Evaluate the prepared test split and optionally persist artifacts."""

        if not self.test_query_indices:
            raise ValueError(
                "prepared split contains no test query indices; test evaluation "
                "cannot run"
            )
        loaded = (
            self.load_checkpoint_for_evaluation(checkpoint)
            if checkpoint is not None
            else None
        )
        if self.normalizer is None:
            raise RuntimeError(
                "load an evaluation checkpoint or fit a normalizer before evaluation"
            )
        test_recalls = _get(
            self.config,
            "retrieval.recall_values",
            default=self._training_value("recall_values", (1, 5, 10)),
        )
        if isinstance(test_recalls, int):
            test_recalls = (test_recalls,)
        else:
            test_recalls = tuple(int(value) for value in test_recalls)
        if not test_recalls:
            raise ValueError("retrieval recall_values cannot be empty")
        top_k = int(
            _get(self.config, "retrieval.top_k", default=max(test_recalls))
        )
        if top_k < 1:
            raise ValueError("retrieval top_k must be positive")
        search_k = max(top_k, max(test_recalls))
        metrics, nearest = self.evaluate(
            self.test_query_indices, test_recalls, search_k=search_k
        )
        final_metrics = {
            **metrics,
            "num_database_frames": len(self.database_paths),
            "num_query_frames": len(self.query_paths),
            "num_evaluated_queries": len(self.test_query_indices),
            "best_epoch": loaded.get("epoch") if loaded is not None else None,
            "test_start_index": self.split.get("test_start"),
            "manifest_fingerprint": self.fingerprint,
        }
        if write_results:
            self._save_json(self.run_dir / "final_metrics.json", final_metrics)
            self._save_json(
                self.run_dir / "test_retrievals.json",
                {
                    "query_indices": self.test_query_indices,
                    "nearest_database_indices": nearest.tolist(),
                },
            )
        return final_metrics, nearest

    def train(self, resume_from: str | Path | None = None) -> dict:
        seed = int(self._training_value("seed", 42))
        generator = seed_everything(
            seed, deterministic=bool(self._training_value("deterministic", True))
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.optimizer = self._make_optimizer()
        if resume_from is not None:
            self.resume(resume_from)
        else:
            self.normalizer = fit_edge_normalizer(
                self._normalizer_paths(),
                self.edge_attr_dim,
                _get(self.config, "normalization.log_indices", default=None),
                exclude_text_edges=getattr(self.model, "text_add", None) is not None,
            )

        negatives_per_query = int(
            self._training_value("negatives_per_query", 2)
        )
        dataset = TripletGraphDataset(
            self.database_paths,
            self.query_paths,
            self.train_query_indices,
            self.positives,
            negatives_per_query,
            self.normalizer,
            self.text_emb_dim,
            seed,
            self.edge_attr_dim,
        )
        if not dataset:
            raise ValueError("no train query has both positives and negative candidates")
        loader = DataLoader(
            dataset,
            batch_size=int(self._training_value("batch_size", 16)),
            shuffle=True,
            num_workers=int(self._training_value("num_workers", 0)),
            collate_fn=collate_graphs,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )
        criterion = nn.TripletMarginLoss(
            margin=float(self._training_value("margin", 0.3)), p=2
        )
        epochs = int(self._training_value("epochs", 30))
        patience = int(self._training_value("patience", 7))
        no_improvement = 0
        history = []
        for epoch in range(self.start_epoch, epochs + 1):
            database_embeddings = self._encode(self.database_paths)
            query_embeddings = self._encode(
                [self.query_paths[index] for index in dataset.query_indices]
            )
            hard = self.miner(
                database_embeddings,
                query_embeddings,
                dataset.query_indices,
                self.positives,
                negatives_per_query,
                int(self._training_value("hard_search_depth", 256)),
                seed + epoch,
            )
            dataset.set_epoch(epoch)
            dataset.set_hard_negatives(hard)
            self.model.train()
            losses = []
            for batch in loader:
                embeddings = self.model(batch.to(self.device))
                group_size = 2 + negatives_per_query
                embeddings = embeddings.reshape(-1, group_size, self.model.out_dim)
                query = embeddings[:, 0]
                positive = embeddings[:, 1]
                negatives = embeddings[:, 2:]
                loss = criterion(
                    query[:, None, :].expand_as(negatives).reshape(-1, self.model.out_dim),
                    positive[:, None, :]
                    .expand_as(negatives)
                    .reshape(-1, self.model.out_dim),
                    negatives.reshape(-1, self.model.out_dim),
                )
                assert self.optimizer is not None
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self._training_value("max_grad_norm", 5.0)),
                )
                self.optimizer.step()
                losses.append(float(loss.item()))
            mean_loss = float(np.mean(losses))
            if self.validation_query_indices:
                metrics, _ = self.evaluate(self.validation_query_indices)
                target_key = str(
                    self._training_value(
                        "early_stopping_metric",
                        "R@5" if "R@5" in metrics else next(iter(metrics)),
                    )
                )
                if target_key not in metrics:
                    raise KeyError(f"validation metric {target_key!r} is unavailable")
                metric = float(metrics[target_key])
            else:
                metrics = {}
                metric = -mean_loss
            epoch_metrics = {"epoch": epoch, "loss": mean_loss, **metrics}
            history.append(epoch_metrics)
            improved = metric > self.best_metric
            if improved:
                self.best_metric = metric
                no_improvement = 0
            else:
                no_improvement += 1
            checkpoint = self._checkpoint(epoch, metric, epoch_metrics)
            self._save_checkpoint(self.run_dir / "last_model.pth", checkpoint)
            if improved:
                self._save_checkpoint(self.run_dir / "best_model.pth", checkpoint)
            if not improved and no_improvement >= patience:
                break
        final_metrics, _ = self.evaluate_test("best", write_results=True)
        return {
            "best_metric": self.best_metric,
            "last_epoch": history[-1]["epoch"] if history else self.start_epoch - 1,
            "history": history,
            "final_metrics": final_metrics,
            "run_dir": str(self.run_dir),
            "best_checkpoint": str(self.run_dir / "best_model.pth"),
            "last_checkpoint": str(self.run_dir / "last_model.pth"),
        }
