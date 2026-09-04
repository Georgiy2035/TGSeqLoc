"""End-to-end modular V4RL preparation."""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from tgseqloc.data.formats import FrameText
from tgseqloc.data.schema import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    SCHEMA_VERSION,
    FrameRecord,
    graph_is_compatible,
)
from tgseqloc.data.v4rl import (
    DEFAULT_NOOP_TEXTS,
    build_gt_mapping,
    build_positive_intervals,
    build_temporal_split,
    build_vocabularies,
    discover_v4rl_records,
    parse_paddleocr,
    parse_scene_graph,
    save_json,
)
from tgseqloc.preparation.fusion import build_fused_graph
from tgseqloc.preparation.text import (
    DEFAULT_TEXT_ENCODER,
    FrozenTextEncoder,
    TextEncoder,
    validate_embeddings,
)


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _normalize_config(config: Any) -> Any:
    """Flatten an AppConfig-like object while retaining simple configs."""

    dataset = _get(config, "dataset")
    if dataset is None or isinstance(dataset, str):
        return config
    preprocess = _get(config, "preprocess")
    cache = _get(config, "cache")
    runtime = _get(config, "runtime")
    device = _get(runtime, "device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "dataset": _get(dataset, "adapter", "v4rl"),
        "dataset_root": _get(dataset, "root"),
        "sequences": _get(dataset, "sequences", ("seq1", "seq2")),
        "ocr_root_template": _get(dataset, "ocr_root_template"),
        "scene_graph_root_template": _get(dataset, "scene_graph_root_template"),
        "gt_path": _get(dataset, "gt_path"),
        "output_root": _get(dataset, "prepared_root"),
        "reference_sequence": _get(dataset, "reference_sequence", "seq1"),
        "query_sequence": _get(dataset, "query_sequence", "seq2"),
        "test_ratio": _get(dataset, "test_ratio", 0.2),
        "validation_ratio": _get(dataset, "validation_ratio", 0.1),
        "interpolate_gt": _get(dataset, "interpolate_gt", False),
        "chunk_size": _get(dataset, "chunk_size", 200),
        "source_ocr_backend": _get(_get(config, "sources"), "ocr", "precomputed_paddleocr"),
        "source_scene_graph_backend": _get(
            _get(config, "sources"), "scene_graph", "external_json"
        ),
        "text_encoder": _get(preprocess, "text_encoder", DEFAULT_TEXT_ENCODER),
        "text_encoder_backend": _get(
            preprocess, "text_encoder_backend", "multilingual_e5"
        ),
        "text_encoder_revision": _get(preprocess, "revision"),
        "text_filter_backend": _get(preprocess, "text_filter", "confidence"),
        "fusion_backend": _get(preprocess, "fusion", "text_nodes"),
        "connection_strategy": _get(preprocess, "connection_strategy", "overlap_nearest"),
        "connection_k": _get(preprocess, "connection_k", 1),
        "ocr_confidence_threshold": _get(preprocess, "ocr_confidence_threshold", 0.0),
        "encoder_batch_size": _get(preprocess, "encoder_batch_size", 128),
        "frame_batch_size": _get(preprocess, "frame_batch_size", 256),
        "gt_tolerance_ns": _get(preprocess, "gt_tolerance_ns", 50_000_000),
        "cache_policy": _get(config, "cache_policy", _get(cache, "policy")),
        "device": device,
    }


def build_preprocess_fingerprint(
    config: Any,
    class_to_idx: Mapping[str, int],
    edge_label_to_idx: Mapping[str, int],
    text_embedding_dim: int,
    source_identities: Mapping[str, Any] | None = None,
    backend_identities: Mapping[str, Any] | None = None,
) -> str:
    """Hash every setting that can alter a prepared graph."""

    config = _normalize_config(config)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": _get(config, "dataset", "v4rl"),
        "sequences": list(_get(config, "sequences", ("seq1", "seq2"))),
        "chunk_size": int(_get(config, "chunk_size", 200)),
        "text_encoder": _get(config, "text_encoder", DEFAULT_TEXT_ENCODER),
        "text_encoder_revision": _get(config, "text_encoder_revision"),
        "text_embedding_dim": int(text_embedding_dim),
        "connection_strategy": _get(config, "connection_strategy", "overlap_nearest"),
        "connection_k": int(_get(config, "connection_k", 1)),
        "ocr_confidence_threshold": float(_get(config, "ocr_confidence_threshold", 0.0)),
        "ocr_noop_texts": sorted(_get(config, "ocr_noop_texts", DEFAULT_NOOP_TEXTS)),
        "node_class_to_idx": dict(class_to_idx),
        "edge_label_to_idx": dict(edge_label_to_idx),
        "backend_identities": backend_identities,
        "source_identities": source_identities,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_file_identity(path: str | Path) -> dict[str, Any]:
    """Return a reproducible path/stat/content identity for one source file."""

    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _stable_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_stable_value(item) for item in value]
    return {
        "module": value.__class__.__module__,
        "qualname": value.__class__.__qualname__,
    }


def callable_identity(value: Any) -> dict[str, Any] | None:
    """Identify a selected backend callable and its explicit cache identity."""

    if value is None:
        return None
    identity = {
        "module": getattr(value, "__module__", value.__class__.__module__),
        "qualname": getattr(value, "__qualname__", value.__class__.__qualname__),
    }
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = None
    if source is not None:
        identity["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cache_identity = getattr(value, "cache_identity", None)
    if callable(cache_identity):
        try:
            cache_identity = cache_identity()
        except TypeError:
            cache_identity = {
                "module": cache_identity.__module__,
                "qualname": cache_identity.__qualname__,
            }
    if cache_identity is not None:
        identity["cache_identity"] = _stable_value(cache_identity)
    return identity


def _require_deterministic_callable(value: Any, name: str) -> None:
    """Reject stateful callables that cannot provide a stable cache identity."""

    is_plain = inspect.isfunction(value) or inspect.isclass(value)
    has_closure = bool(getattr(value, "__closure__", None))
    if (not is_plain or has_closure) and getattr(value, "cache_identity", None) is None:
        raise ValueError(
            f"Stateful {name} implementations must define deterministic cache_identity"
        )


def _explicit_cache_identity(value: Any) -> Any:
    identity = getattr(value, "cache_identity", None)
    if callable(identity):
        try:
            identity = identity()
        except TypeError:
            identity = callable_identity(identity)
    return _stable_value(identity)


def encoder_identity(
    encoder: TextEncoder, factory: Callable[..., TextEncoder]
) -> dict[str, Any]:
    """Return the actual encoder implementation/model revision identity."""

    model = getattr(encoder, "model", None)
    tokenizer = getattr(encoder, "tokenizer", None)
    resolved_commit = (
        getattr(encoder, "resolved_commit", None)
        or getattr(encoder, "commit_hash", None)
        or getattr(getattr(model, "config", None), "_commit_hash", None)
        or getattr(tokenizer, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    )
    return {
        "implementation": callable_identity(encoder.__class__),
        "factory": callable_identity(factory),
        "cache_identity": _explicit_cache_identity(encoder),
        "model_name": getattr(encoder, "model_name", None),
        "revision": getattr(encoder, "revision", None),
        "resolved_commit": resolved_commit,
        "embedding_dim": int(encoder.embedding_dim),
    }


def backend_identities(
    config: Any,
    encoder: TextEncoder,
    encoder_factory: Callable[..., TextEncoder],
    ocr_parser: Callable[..., Any],
    scene_graph_parser: Callable[..., Any],
    text_filter: Callable[..., Any] | None,
    fusion_builder: Callable[..., Any],
) -> dict[str, Any]:
    """Describe configured backend names and selected implementations."""

    return {
        "configured": {
            "ocr_source": _get(config, "source_ocr_backend", "precomputed_paddleocr"),
            "scene_graph_source": _get(
                config, "source_scene_graph_backend", "external_json"
            ),
            "text_filter": _get(config, "text_filter_backend", "confidence"),
            "fusion": _get(config, "fusion_backend", "text_nodes"),
            "encoder": _get(config, "text_encoder_backend", "multilingual_e5"),
            "encoder_model": _get(config, "text_encoder", DEFAULT_TEXT_ENCODER),
        },
        "implementations": {
            "ocr_parser": callable_identity(ocr_parser),
            "scene_graph_parser": callable_identity(scene_graph_parser),
            "text_filter": callable_identity(text_filter),
            "fusion_builder": callable_identity(fusion_builder),
            "encoder": encoder_identity(encoder, encoder_factory),
        },
    }


def fingerprint_payload(value: Any) -> str:
    encoded = json.dumps(
        _stable_value(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def frame_fingerprint(
    global_fingerprint: str, source_identity: Mapping[str, Any]
) -> str:
    """Combine graph-global identity with one frame's source identity."""

    return fingerprint_payload(
        {"global_preprocess_fingerprint": global_fingerprint, "source": source_identity}
    )


def build_source_identities(
    records: list[FrameRecord], gt_path: str | Path
) -> dict[str, Any]:
    """Describe all files whose contents affect prepared V4RL artifacts."""

    return {
        "algorithm": "sha256",
        "frames": [
            {
                "sequence": record.sequence,
                "index": record.index,
                "stem": record.stem,
                "image": source_file_identity(record.image_path),
                "ocr": source_file_identity(record.ocr_path),
                "scene_graph": source_file_identity(record.graph_path),
            }
            for record in records
        ],
        "ground_truth": source_file_identity(gt_path),
    }


def resolve_cache_policy(config: Any) -> str:
    """Resolve modern cache policy with policy/resume compatibility."""

    config = _normalize_config(config)
    policy = _get(config, "cache_policy")
    if policy is None:
        policy = _get(config, "policy")
    if policy is None:
        policy = "reuse_if_compatible" if bool(_get(config, "resume", True)) else "rebuild"
    policy = str(policy)
    allowed = {"reuse_if_compatible", "require_existing", "rebuild"}
    if policy not in allowed:
        raise ValueError(
            f"Invalid V4RL cache policy {policy!r}; expected one of {sorted(allowed)}"
        )
    return policy


def load_prepared_graph(path: str | Path) -> Any:
    """Load trusted local preparation output across PyTorch defaults."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def is_compatible_output(
    path: str | Path, fingerprint: str, text_embedding_dim: int
) -> bool:
    try:
        return graph_is_compatible(
            load_prepared_graph(path), fingerprint, text_embedding_dim
        )
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        pickle.UnpicklingError,
    ):
        return False


def _make_encoder(
    config: Any,
    encoder: TextEncoder | None,
    factory: Callable[..., TextEncoder],
) -> TextEncoder:
    if encoder is not None:
        return encoder
    return factory(
        model_name=_get(config, "text_encoder", DEFAULT_TEXT_ENCODER),
        revision=_get(config, "text_encoder_revision"),
        device=_get(config, "device", "cpu"),
        batch_size=int(_get(config, "encoder_batch_size", 128)),
    )


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _graph_value(graph: Any, key: str, default: Any = None) -> Any:
    if isinstance(graph, Mapping):
        return graph.get(key, default)
    return getattr(graph, key, default)


def process_v4rl(
    config: Any,
    *,
    encoder: TextEncoder | None = None,
    text_encoder_factory: Callable[..., TextEncoder] = FrozenTextEncoder,
    ocr_parser: Callable[..., FrameText] = parse_paddleocr,
    scene_graph_parser: Callable[..., tuple[list[dict[str, Any]], list[tuple[int, int, str]], int]] = parse_scene_graph,
    text_filter: Callable[[Mapping[str, Any], float], bool] | None = None,
    fusion_builder: Callable[..., Any] = build_fused_graph,
) -> dict[str, Any]:
    """Prepare all V4RL frames and return the written manifest.

    ``config`` may be a dataclass, argparse Namespace, or mapping. An injected
    encoder avoids Hugging Face downloads in tests and controlled pipelines.
    """

    config = _normalize_config(config)
    dataset = _get(config, "dataset", "v4rl")
    if dataset != "v4rl":
        raise ValueError(f"Unsupported dataset adapter: {dataset}")
    required = (
        "dataset_root", "ocr_root_template", "scene_graph_root_template",
        "gt_path", "output_root",
    )
    missing = [name for name in required if _get(config, name) is None]
    if missing:
        raise ValueError(f"Missing V4RL preparation settings: {', '.join(missing)}")

    sequences = tuple(_get(config, "sequences", ("seq1", "seq2")))
    records = discover_v4rl_records(
        _get(config, "dataset_root"),
        str(_get(config, "ocr_root_template")),
        str(_get(config, "scene_graph_root_template")),
        sequences,
        chunk_size=int(_get(config, "chunk_size", 200)),
    )
    class_to_idx, edge_label_to_idx = build_vocabularies(records)
    source_identities = build_source_identities(records, _get(config, "gt_path"))
    for backend, implementation in {
        "OCR source": ocr_parser,
        "scene-graph source": scene_graph_parser,
        "text filter": text_filter,
        "fusion": fusion_builder,
        "text encoder factory": text_encoder_factory,
    }.items():
        if implementation is not None:
            _require_deterministic_callable(implementation, backend)
    text_encoder = _make_encoder(config, encoder, text_encoder_factory)
    if getattr(text_encoder, "cache_identity", None) is None:
        raise ValueError(
            "Text encoders must define deterministic cache_identity"
        )
    embedding_dim = int(text_encoder.embedding_dim)
    selected_backends = backend_identities(
        config,
        text_encoder,
        text_encoder_factory,
        ocr_parser,
        scene_graph_parser,
        text_filter,
        fusion_builder,
    )
    global_fingerprint = build_preprocess_fingerprint(
        config,
        class_to_idx,
        edge_label_to_idx,
        embedding_dim,
        backend_identities=selected_backends,
    )
    identities_by_frame = {
        (item["sequence"], item["index"]): item
        for item in source_identities["frames"]
    }
    frame_fingerprints = {
        (record.sequence, record.index): frame_fingerprint(
            global_fingerprint,
            identities_by_frame[(record.sequence, record.index)],
        )
        for record in records
    }
    output_root = Path(_get(config, "output_root")) / dataset
    output_root.mkdir(parents=True, exist_ok=True)
    previous_records: dict[str, Mapping[str, Any]] = {}
    previous_manifest_path = output_root / "manifest.json"
    if previous_manifest_path.is_file():
        try:
            previous_manifest = json.loads(
                previous_manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(previous_manifest, Mapping):
                raise TypeError("manifest root must be an object")
            previous_records = {
                str(item["path"]): item
                for item in previous_manifest.get("graph_records", [])
                if isinstance(item, Mapping) and item.get("path")
            }
        except (OSError, json.JSONDecodeError, TypeError):
            previous_records = {}
    cache_policy = resolve_cache_policy(config)
    frame_batch_size = max(1, int(_get(config, "frame_batch_size", 256)))

    written = skipped = dropped_this_run = 0
    for chunk_start in range(0, len(records), frame_batch_size):
        pending = []
        flat_texts: list[str] = []
        for record in records[chunk_start:chunk_start + frame_batch_size]:
            output_path = output_root / record.sequence / f"{record.stem}.pt"
            expected_fingerprint = frame_fingerprints[(record.sequence, record.index)]
            exists = output_path.is_file()
            compatible = exists and is_compatible_output(
                output_path, expected_fingerprint, embedding_dim
            )
            previous = previous_records.get(
                (Path(record.sequence) / f"{record.stem}.pt").as_posix()
            )
            if compatible:
                if previous is None:
                    compatible = False
                else:
                    previous_sha = previous.get("sha256")
                    previous_fingerprint = previous.get("expected_fingerprint")
                    compatible = (
                        previous_fingerprint == expected_fingerprint
                        and isinstance(previous_sha, str)
                        and source_file_identity(output_path)["sha256"] == previous_sha
                    )
            if cache_policy == "require_existing" and not compatible:
                reason = "absent" if not exists else "incompatible or corrupt"
                raise RuntimeError(
                    "V4RL cache policy 'require_existing' rejected "
                    f"{reason} prepared graph: {output_path}"
                )
            if cache_policy != "rebuild" and compatible:
                skipped += 1
                continue
            object_nodes, scene_edges, dropped = scene_graph_parser(
                record.graph_path, class_to_idx
            )
            frame_text = ocr_parser(
                record.ocr_path,
                float(_get(config, "ocr_confidence_threshold", 0.0)),
                noop_texts=_get(config, "ocr_noop_texts", DEFAULT_NOOP_TEXTS),
                prediction_filter=text_filter,
            )
            texts = frame_text.texts
            text_start = len(flat_texts)
            flat_texts.extend(texts)
            pending.append(
                (
                    record, output_path, object_nodes, scene_edges, dropped,
                    frame_text.boxes, texts, text_start,
                )
            )
        encoded = validate_embeddings(
            text_encoder, flat_texts, text_encoder.encode(flat_texts)
        )
        for record, output_path, nodes, edges, dropped, boxes, texts, start in pending:
            metadata = {
                "sequence": record.sequence,
                "frame_index": record.index,
                "timestamp": record.timestamp,
                "frame_stem": record.stem,
                "image_path": str(record.image_path),
                "ocr_path": str(record.ocr_path),
                "scene_graph_path": str(record.graph_path),
                "text_encoder_name": getattr(
                    text_encoder, "model_name", _get(config, "text_encoder")
                ),
                "text_encoder_revision": getattr(text_encoder, "revision", None),
                "preprocess_fingerprint": frame_fingerprints[
                    (record.sequence, record.index)
                ],
                "global_preprocess_fingerprint": global_fingerprint,
                "source_identity": identities_by_frame[
                    (record.sequence, record.index)
                ],
                "backend_identities": selected_backends,
                "dropped_scene_edge_count": dropped,
                "graph_rotated": False,
                "schema_version": SCHEMA_VERSION,
            }
            graph = fusion_builder(
                nodes, edges, boxes, texts, encoded[start:start + len(texts)],
                edge_label_to_idx,
                connection_strategy=_get(config, "connection_strategy", "overlap_nearest"),
                connection_k=int(_get(config, "connection_k", 1)),
                metadata=metadata,
            )
            _atomic_torch_save(graph, output_path)
            written += 1
            dropped_this_run += dropped

    mapping = build_gt_mapping(
        _get(config, "gt_path"), records,
        int(_get(config, "gt_tolerance_ns", 50_000_000)),
        reference_sequence=str(_get(config, "reference_sequence", "seq1")),
        query_sequence=str(_get(config, "query_sequence", "seq2")),
    )
    mappings_root = output_root / "mappings"
    save_json(mapping, mappings_root / "gt_shop_street_1.json")
    query_sequence = mapping["query_sequence"]
    query_records = [record for record in records if record.sequence == query_sequence]
    positives = build_positive_intervals(
        mapping, query_records, interpolate=bool(_get(config, "interpolate_gt", False))
    )
    split = build_temporal_split(
        len(query_records), positives,
        test_ratio=float(_get(config, "test_ratio", 0.2)),
        validation_ratio=float(_get(config, "validation_ratio", 0.1)),
        require_nonempty=False,
    )
    reference_sequence = mapping["reference_sequence"]
    reference_records = [
        record for record in records if record.sequence == reference_sequence
    ]
    split["database_paths"] = [
        f"{record.sequence}/{record.stem}.pt" for record in reference_records
    ]
    split["query_paths"] = [
        f"{record.sequence}/{record.stem}.pt" for record in query_records
    ]
    split["reference_sequence"] = reference_sequence
    split["query_sequence"] = query_sequence
    save_json(split, mappings_root / "temporal_split.json")

    expected_graphs = []
    compatible_count = 0
    existing_expected_count = 0
    total_dropped = 0
    for record in records:
        relative_path = Path(record.sequence) / f"{record.stem}.pt"
        path = output_root / relative_path
        expected_fingerprint = frame_fingerprints[(record.sequence, record.index)]
        exists = path.is_file()
        compatible = exists and is_compatible_output(
            path, expected_fingerprint, embedding_dim
        )
        existing_expected_count += int(exists)
        compatible_count += int(compatible)
        graph_sha256 = source_file_identity(path)["sha256"] if exists else None
        expected_graphs.append(
            {
                "path": relative_path.as_posix(),
                "sequence": record.sequence,
                "frame_index": record.index,
                "frame_stem": record.stem,
                "sha256": graph_sha256,
                "expected_fingerprint": expected_fingerprint,
                "compatible": compatible,
            }
        )
        if not exists:
            continue
        try:
            graph = load_prepared_graph(path)
            total_dropped += int(_graph_value(graph, "dropped_scene_edge_count", 0))
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError):
            pass
    expected_path_set = {entry["path"] for entry in expected_graphs}
    stale_graph_paths = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*.pt")
        if path.relative_to(output_root).as_posix() not in expected_path_set
    )
    dataset_fingerprint = fingerprint_payload(
        {
            "global_preprocess_fingerprint": global_fingerprint,
            "expected_frame_fingerprints": {
                entry["path"]: entry["expected_fingerprint"]
                for entry in expected_graphs
            },
            "graph_sha256": {
                entry["path"]: entry["sha256"] for entry in expected_graphs
            },
            "ground_truth": source_identities["ground_truth"],
            "mapping": mapping,
            "temporal_split": split,
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "sequences": list(sequences),
        "frame_count": len(records),
        "written_count": written,
        "skipped_count": skipped,
        "expected_graph_count": len(expected_graphs),
        "output_file_count": existing_expected_count,
        "validated_output_count": compatible_count,
        "output_complete": compatible_count == len(records),
        "cache_policy": cache_policy,
        "preprocess_fingerprint": dataset_fingerprint,
        "fingerprint": dataset_fingerprint,
        "global_preprocess_fingerprint": global_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint_identity": dataset_fingerprint,
        "expected_graph_paths": [entry["path"] for entry in expected_graphs],
        "expected_graphs": expected_graphs,
        "graph_records": expected_graphs,
        "stale_graph_paths": stale_graph_paths,
        "stale_graph_count": len(stale_graph_paths),
        "source_identities": source_identities,
        "backend_identities": selected_backends,
        "text_encoder": getattr(text_encoder, "model_name", _get(config, "text_encoder")),
        "text_encoder_revision": getattr(text_encoder, "revision", None),
        "text_embedding_dim": embedding_dim,
        "connection_strategy": _get(config, "connection_strategy", "overlap_nearest"),
        "connection_k": int(_get(config, "connection_k", 1)),
        "ocr_confidence_threshold": float(_get(config, "ocr_confidence_threshold", 0.0)),
        "node_class_to_idx": class_to_idx,
        "edge_label_to_idx": edge_label_to_idx,
        "num_obj_classes": len(class_to_idx),
        "num_edge_classes": len(edge_label_to_idx),
        "uses_text_nodes": True,
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_attr_dim": EDGE_FEATURE_DIM,
        "graph_rotated": False,
        "dropped_scene_edge_count_this_run": dropped_this_run,
        "dropped_scene_edge_count": total_dropped,
        "mapping_path": "mappings/gt_shop_street_1.json",
        "temporal_split_path": "mappings/temporal_split.json",
        "output_layout": "{dataset}/{sequence}/{frame_stem}.pt",
    }
    save_json(manifest, output_root / "manifest.json")
    return manifest


def discover_v4rl_inputs(config: Any) -> int:
    """Check that every precomputed input exists and count the frames.

    Attached to the adapter below so that the orchestrator can verify inputs
    without knowing how V4RL is laid out on disk.
    """

    settings = _normalize_config(config)
    records = discover_v4rl_records(
        _get(settings, "dataset_root"),
        _get(settings, "ocr_root_template"),
        _get(settings, "scene_graph_root_template"),
        _get(settings, "sequences", ("seq1", "seq2")),
        chunk_size=int(_get(settings, "chunk_size", 200)),
    )
    gt_path = Path(_get(settings, "gt_path"))
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing V4RL ground truth: {gt_path}")
    return len(records)


process_v4rl.discover_inputs = discover_v4rl_inputs
process_dataset = process_v4rl
