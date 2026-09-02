"""Geometry and text-node fusion for scene graphs."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch_geometric.data import Data

from tgseqloc.data.schema import sanitize_graph

EPSILON = 1e-8
CONNECTION_STRATEGIES = frozenset(
    {"overlap_nearest", "overlap", "nearest", "fully_connected"}
)


def cxcywh_to_xyxy(cx: float, cy: float, width: float, height: float) -> list[float]:
    return [
        max(0.0, cx - width / 2), max(0.0, cy - height / 2),
        min(1.0, cx + width / 2), min(1.0, cy + height / 2),
    ]


def iou_xyxy(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / (first_area + second_area - intersection + EPSILON)


def compact_edge_features(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> list[float]:
    """Compute the canonical directed 10D relative geometry vector."""

    sx, sy = source["center"]
    tx, ty = target["center"]
    sw, sh = source["wh"]
    tw, th = target["wh"]
    dx, dy = float(tx - sx), float(ty - sy)
    distance = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    source_area, target_area = max(0.0, sw * sh), max(0.0, tw * th)
    ax1, ay1, ax2, ay2 = source["xyxy"]
    bx1, by1, bx2, by2 = target["xyxy"]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    return [
        distance, dx, dy, math.sin(angle), math.cos(angle),
        iou_xyxy(source["xyxy"], target["xyxy"]),
        intersection / (min(source_area, target_area) + EPSILON)
        if min(source_area, target_area) > 0 else 0.0,
        math.log1p(target_area / (source_area + EPSILON)) if source_area > 0 else 0.0,
        math.log1p(tw / (sw + EPSILON)) if sw > 0 else 0.0,
        math.log1p(th / (sh + EPSILON)) if sh > 0 else 0.0,
    ]


def text_node_from_box(box: Sequence[float]) -> dict[str, Any]:
    x1, y1, x2, y2 = map(float, box)
    center = [(x1 + x2) / 2, (y1 + y2) / 2]
    return {
        "class_idx": 0, "class_name": "text", "center": center,
        "wh": [x2 - x1, y2 - y1], "xyxy": [x1, y1, x2, y2],
    }


def select_text_connections(
    text_node: Mapping[str, Any],
    object_nodes: Sequence[Mapping[str, Any]],
    strategy: str = "overlap_nearest",
    k: int = 1,
) -> list[int]:
    if strategy not in CONNECTION_STRATEGIES:
        raise ValueError(f"Unknown text connection strategy: {strategy}")
    if not object_nodes:
        return []
    tx, ty = text_node["center"]
    scored = []
    for index, node in enumerate(object_nodes):
        x1, y1, x2, y2 = node["xyxy"]
        contains = x1 <= tx <= x2 and y1 <= ty <= y2
        overlap = iou_xyxy(text_node["xyxy"], node["xyxy"])
        nx, ny = node["center"]
        scored.append((index, contains, overlap, math.hypot(tx - nx, ty - ny)))
    if strategy == "fully_connected":
        return [value[0] for value in scored]
    if strategy == "overlap":
        return [index for index, contains, overlap, _ in scored if contains or overlap > 0]
    if strategy == "nearest":
        return [value[0] for value in sorted(scored, key=lambda value: (value[3], value[0]))[:max(1, k)]]
    return [max(scored, key=lambda value: (value[1], value[2], -value[3], -value[0]))[0]]


def build_fused_graph(
    object_nodes: Sequence[Mapping[str, Any]],
    scene_edges: Sequence[tuple[int, int, str]],
    text_boxes: Sequence[Sequence[float]],
    texts: Sequence[str],
    text_embeddings: torch.Tensor,
    edge_label_to_idx: Mapping[str, int],
    *,
    connection_strategy: str = "overlap_nearest",
    connection_k: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> Data:
    """Fuse text nodes and bidirectional text-object edges into a PyG graph."""

    if len(text_boxes) != len(texts):
        raise ValueError("text_boxes and texts must have the same length")
    embeddings = torch.as_tensor(text_embeddings, dtype=torch.float32, device="cpu")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("text_embeddings must have shape [len(texts), D]")
    text_nodes = [text_node_from_box(box) for box in text_boxes]
    all_nodes = list(object_nodes) + text_nodes
    object_count, node_count = len(object_nodes), len(all_nodes)
    embedding_dim = embeddings.shape[1]

    sources: list[int] = []
    targets: list[int] = []
    attributes: list[list[float]] = []
    labels: list[int] = []
    source_classes: list[int] = []
    target_classes: list[int] = []
    text_edges: list[bool] = []

    def append_edge(source: int, target: int, label: int, is_text_edge: bool) -> None:
        sources.append(source)
        targets.append(target)
        attributes.append(compact_edge_features(all_nodes[source], all_nodes[target]))
        labels.append(label)
        source_classes.append(int(all_nodes[source]["class_idx"]))
        target_classes.append(int(all_nodes[target]["class_idx"]))
        text_edges.append(is_text_edge)

    unknown_edge = int(edge_label_to_idx.get("unknown", 0))
    for source, target, label in scene_edges:
        append_edge(source, target, int(edge_label_to_idx.get(label, unknown_edge)), False)
    for offset, text_node in enumerate(text_nodes):
        text_index = object_count + offset
        for object_index in select_text_connections(
            text_node, object_nodes, connection_strategy, connection_k
        ):
            append_edge(text_index, object_index, unknown_edge, True)
            append_edge(object_index, text_index, unknown_edge, True)

    edge_count = len(sources)
    data = Data(
        x=torch.tensor(
            [[*node["center"], *node["wh"]] for node in all_nodes],
            dtype=torch.float32,
        ).reshape(node_count, 4),
        node_class=torch.tensor([node["class_idx"] for node in all_nodes], dtype=torch.long),
        is_text=torch.tensor([False] * object_count + [True] * len(text_nodes), dtype=torch.bool),
        text_emb=torch.cat(
            [torch.zeros((object_count, embedding_dim)), embeddings], dim=0
        ),
        edge_index=torch.tensor([sources, targets], dtype=torch.long)
        if edge_count else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(attributes, dtype=torch.float32)
        if edge_count else torch.empty((0, 10), dtype=torch.float32),
        edge_label=torch.tensor(labels, dtype=torch.long),
        edge_u_class=torch.tensor(source_classes, dtype=torch.long),
        edge_v_class=torch.tensor(target_classes, dtype=torch.long),
        is_text_edge=torch.tensor(text_edges, dtype=torch.bool),
        text_strings=list(texts),
    )
    for key, value in (metadata or {}).items():
        setattr(data, key, value)
    return sanitize_graph(data, text_embedding_dim=int(embedding_dim))
