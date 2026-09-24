"""Canonical graph schema used by TGSeqLoc preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch_geometric.data import Data

from .formats import FrameRecord

SCHEMA_VERSION = 3
NODE_FEATURE_DIM = 4
EDGE_FEATURE_DIM = 10





def _tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(dtype=dtype, device="cpu")
    return torch.as_tensor(value, dtype=dtype)


def _vector(value: Any, length: int, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        return torch.zeros(length, dtype=dtype)
    result = _tensor(value, dtype).reshape(-1)
    fixed = torch.zeros(length, dtype=dtype)
    fixed[: min(length, result.numel())] = result[:length]
    return fixed


def sanitize_graph(
    graph: Data | Mapping[str, Any],
    *,
    text_embedding_dim: int | None = None,
) -> Data:
    """Return a CPU, finite, shape-consistent PyG graph.

    Invalid edges are removed. Mismatched edge fields are padded rather than
    silently changing the topology.
    """

    source: Any = graph
    get = source.get if isinstance(source, Mapping) else lambda key, default=None: getattr(source, key, default)

    x = _tensor(get("x", torch.empty((0, NODE_FEATURE_DIM))), torch.float32)
    if x.ndim != 2 or x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError(f"x must have shape [N, {NODE_FEATURE_DIM}], got {tuple(x.shape)}")
    x = torch.nan_to_num(x)
    node_count = int(x.shape[0])

    raw_text_emb = get("text_emb")
    if text_embedding_dim is None:
        text_embedding_dim = (
            int(raw_text_emb.shape[1])
            if torch.is_tensor(raw_text_emb) and raw_text_emb.ndim == 2
            else 0
        )
    text_emb = (
        _tensor(raw_text_emb, torch.float32)
        if raw_text_emb is not None
        else torch.zeros((node_count, text_embedding_dim), dtype=torch.float32)
    )
    if text_emb.shape != (node_count, text_embedding_dim):
        fixed = torch.zeros((node_count, text_embedding_dim), dtype=torch.float32)
        if text_emb.ndim == 2:
            rows = min(node_count, text_emb.shape[0])
            cols = min(text_embedding_dim, text_emb.shape[1])
            fixed[:rows, :cols] = text_emb[:rows, :cols]
        text_emb = fixed
    text_emb = torch.nan_to_num(text_emb)

    edge_index = _tensor(get("edge_index", torch.empty((2, 0))), torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    original_edges = int(edge_index.shape[1])
    valid = (
        (edge_index[0] >= 0)
        & (edge_index[0] < node_count)
        & (edge_index[1] >= 0)
        & (edge_index[1] < node_count)
    )
    edge_index = edge_index[:, valid]

    def edge_matrix(value: Any) -> torch.Tensor:
        result = (
            _tensor(value, torch.float32)
            if value is not None
            else torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
        )
        if result.ndim == 1 and result.numel() == EDGE_FEATURE_DIM:
            result = result.reshape(1, EDGE_FEATURE_DIM)
        fixed = torch.zeros((original_edges, EDGE_FEATURE_DIM), dtype=torch.float32)
        if result.ndim == 2:
            rows = min(original_edges, result.shape[0])
            cols = min(EDGE_FEATURE_DIM, result.shape[1])
            fixed[:rows, :cols] = result[:rows, :cols]
        return torch.nan_to_num(fixed[valid])

    data = Data(
        x=x,
        node_class=_vector(get("node_class"), node_count, torch.long),
        is_text=_vector(get("is_text"), node_count, torch.bool),
        text_emb=text_emb,
        edge_index=edge_index,
        edge_attr=edge_matrix(get("edge_attr")),
        edge_label=_vector(get("edge_label"), original_edges, torch.long)[valid],
        edge_u_class=_vector(get("edge_u_class"), original_edges, torch.long)[valid],
        edge_v_class=_vector(get("edge_v_class"), original_edges, torch.long)[valid],
        is_text_edge=_vector(get("is_text_edge"), original_edges, torch.bool)[valid],
    )
    keys = source.keys() if isinstance(source, Mapping) else source.keys()
    canonical = set(data.keys())
    for key in keys:
        if key not in canonical:
            setattr(data, key, get(key))
    return data


def graph_is_compatible(
    graph: Any, fingerprint: str, text_embedding_dim: int
) -> bool:
    """Check schema and preprocessing identity without mutating a graph."""

    try:
        get = (
            graph.get
            if isinstance(graph, Mapping)
            else lambda key, default=None: getattr(graph, key, default)
        )
        x = get("x")
        edge_index = get("edge_index")
        edge_attr = get("edge_attr")
        text_emb = get("text_emb")
        if not all(torch.is_tensor(value) for value in (x, edge_index, edge_attr, text_emb)):
            return False
        if x.ndim != 2 or x.shape[1] != NODE_FEATURE_DIM:
            return False
        node_count = int(x.shape[0])
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            return False
        edge_count = int(edge_index.shape[1])
        node_class = get("node_class")
        is_text = get("is_text")
        edge_label = get("edge_label")
        edge_u_class = get("edge_u_class")
        edge_v_class = get("edge_v_class")
        is_text_edge = get("is_text_edge")
        return (
            get("preprocess_fingerprint") == fingerprint
            and text_emb.shape == (node_count, text_embedding_dim)
            and edge_attr.shape == (edge_count, EDGE_FEATURE_DIM)
            and torch.is_tensor(node_class)
            and node_class.numel() == node_count
            and torch.is_tensor(is_text)
            and is_text.numel() == node_count
            and torch.is_tensor(edge_label)
            and edge_label.numel() == edge_count
            and torch.is_tensor(edge_u_class)
            and edge_u_class.numel() == edge_count
            and torch.is_tensor(edge_v_class)
            and edge_v_class.numel() == edge_count
            and torch.is_tensor(is_text_edge)
            and is_text_edge.numel() == edge_count
            and (edge_count == 0 or bool(
                ((edge_index >= 0) & (edge_index < node_count)).all()
            ))
            and bool(torch.isfinite(x).all())
            and bool(torch.isfinite(text_emb).all())
            and bool(torch.isfinite(edge_attr).all())
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return False
