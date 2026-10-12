"""Evaluating a checkpoint on data it was not trained on.

A checkpoint normally refuses any prepared data but its own: it records the
fingerprint of the data it was trained on. Evaluating a model on another
dataset needs a different contract -- not "the same data" but "data the model
can read": the same node-class and relation vocabularies, the same feature
dimensions and the same string encoder. Everything learned from the training
data -- weights and edge-attribute statistics -- comes from the checkpoint;
nothing is fitted on the evaluation data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from tgseqloc.evaluation import evaluate_retrieval
from tgseqloc.models import EdgeAttrNormalizer, GATGraphEncoder

from .trainer import (
    _torch_load,
    encode_paths,
    file_sha256,
    manifest_fingerprint,
    validate_graph_records,
)


def _encoder_identity(manifest: Mapping[str, Any]) -> Any:
    implementations = (manifest.get("backend_identities") or {}).get("implementations") or {}
    encoder = implementations.get("encoder") or {}
    return encoder.get("cache_identity")


def transfer_incompatibilities(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    model_init_args: Mapping[str, Any],
) -> list[str]:
    """Why a model trained on ``source`` data cannot read ``target`` data.

    Text is one-directional: a model with a text layer reads graphs without
    text -- such a frame is encoded as by a model without the layer, the "text
    removed" control -- but a model without one cannot read text it has no
    weights for.
    """

    problems = []
    for key in (
        "node_class_to_idx",
        "edge_label_to_idx",
        "node_feature_dim",
        "edge_attr_dim",
        "text_embedding_dim",
        "schema_version",
    ):
        if source.get(key) != target.get(key):
            problems.append(f"{key} differs")
    if _encoder_identity(source) != _encoder_identity(target):
        problems.append("the string encoder differs")
    if target.get("uses_text_nodes", True) and not model_init_args.get("use_text_nodes", True):
        problems.append("a model without text cannot read graphs with text")
    if target.get("uses_text_nodes", True) and (
        source.get("connection_strategy"), source.get("connection_k")
    ) != (target.get("connection_strategy"), target.get("connection_k")):
        problems.append("text is attached to objects differently")
    return problems


def evaluate_transfer(
    config: Mapping[str, Any],
    data_root: Path,
    manifest: Mapping[str, Any],
    split: Mapping[str, Any],
    checkpoint: str | Path,
    *,
    run_dir: Path,
    retriever_factory=None,
    metric_factory=None,
) -> dict[str, Any]:
    """Recall of a checkpoint on the test queries of other prepared data."""

    payload = _torch_load(checkpoint)
    missing = sorted(
        {"model_init_args", "model_state_dict", "normalizer_state", "manifest"} - payload.keys()
    )
    if missing:
        raise ValueError(f"checkpoint {checkpoint} lacks {', '.join(missing)}")
    init_args = dict(payload["model_init_args"])
    problems = transfer_incompatibilities(payload["manifest"], manifest, init_args)
    if problems:
        raise ValueError(f"checkpoint {checkpoint} cannot read {data_root}: " + "; ".join(problems))

    data_root = Path(data_root)
    database = [data_root / path for path in split["database_paths"]]
    query_paths = [data_root / path for path in split["query_paths"]]
    test = [int(index) for index in split.get("test_query_indices", ())]
    if not test:
        raise ValueError("prepared split contains no test query indices")
    queries = [query_paths[index] for index in test]
    validate_graph_records(manifest, data_root, database + queries)

    retrieval = config.get("retrieval", {})
    runtime = config.get("runtime", {})
    training = config.get("training", {})
    device = str(runtime.get("device", "cpu"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GATGraphEncoder(**init_args)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    normalizer = EdgeAttrNormalizer.from_state_dict(payload["normalizer_state"])
    encode = dict(
        text_emb_dim=int(manifest["text_embedding_dim"]),
        device=device,
        batch_size=int(training.get("infer_batch_size", 128)),
        num_workers=int(runtime.get("num_workers", 0)),
        edge_attr_dim=int(manifest["edge_attr_dim"]),
    )
    database_descriptors = encode_paths(model, database, normalizer, **encode)
    query_descriptors = encode_paths(model, queries, normalizer, **encode)

    recalls = tuple(int(value) for value in retrieval.get("recall_values", (1, 5, 10)))
    top_k = int(retrieval.get("top_k", max(recalls)))
    kwargs = {}
    if retriever_factory is not None:
        kwargs["retriever_factory"] = retriever_factory
    if metric_factory is not None:
        kwargs["metric_factory"] = metric_factory
    metrics, nearest = evaluate_retrieval(
        database_descriptors,
        query_descriptors,
        {int(k): v for k, v in split.get("positives", {}).items()},
        test,
        recalls,
        search_k=max(top_k, max(recalls)),
        **kwargs,
    )
    result = {
        **metrics,
        "num_database_frames": len(database),
        "num_query_frames": len(query_paths),
        "num_evaluated_queries": len(test),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(Path(checkpoint)),
        "source_manifest_fingerprint": payload.get("manifest_fingerprint"),
        "source_epoch": payload.get("epoch"),
        "manifest_fingerprint": manifest_fingerprint(manifest),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("final_metrics.json", result),
        ("test_retrievals.json", {"query_indices": test, "nearest_database_indices": nearest.tolist()}),
    ):
        temporary = run_dir / f"{name}.tmp"
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(run_dir / name)
    return result
