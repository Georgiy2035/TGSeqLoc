"""Dataset schemas and source adapters."""

from .schema import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    SCHEMA_VERSION,
    FrameRecord,
    graph_is_compatible,
    sanitize_graph,
)
from .v4rl import (
    build_gt_mapping,
    build_positive_intervals,
    build_split,
    build_temporal_split,
    build_vocabularies,
    discover_v4rl_records,
    parse_paddleocr,
    parse_scene_graph,
)

__all__ = [
    "EDGE_FEATURE_DIM",
    "NODE_FEATURE_DIM",
    "SCHEMA_VERSION",
    "FrameRecord",
    "build_gt_mapping",
    "build_positive_intervals",
    "build_split",
    "build_temporal_split",
    "build_vocabularies",
    "discover_v4rl_records",
    "graph_is_compatible",
    "parse_paddleocr",
    "parse_scene_graph",
    "sanitize_graph",
]
