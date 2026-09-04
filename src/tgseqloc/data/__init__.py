"""Dataset schemas and source adapters.

Names resolve lazily so that importing a format does not drag in torch. The
canonical frame formats are pure data, and a stage that only produces them —
OCR, segmentation — must be runnable in an environment built for that model
alone. PaddleOCR and torch, for one, are happier in separate environments.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "FrameMasks": "formats",
    "FrameText": "formats",
    "TextDetection": "formats",
    "EDGE_FEATURE_DIM": "schema",
    "NODE_FEATURE_DIM": "schema",
    "SCHEMA_VERSION": "schema",
    "FrameRecord": "schema",
    "graph_is_compatible": "schema",
    "sanitize_graph": "schema",
    "build_gt_mapping": "v4rl",
    "build_positive_intervals": "v4rl",
    "build_split": "v4rl",
    "build_temporal_split": "v4rl",
    "build_vocabularies": "v4rl",
    "discover_v4rl_records": "v4rl",
    "parse_paddleocr": "v4rl",
    "parse_scene_graph": "v4rl",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return __all__
