"""Stable source-adapter API for dataset preparation."""

from tgseqloc.data.v4rl import (
    DEFAULT_NOOP_TEXTS,
    build_vocabularies,
    discover_v4rl_records,
    load_json,
    parse_paddleocr,
    parse_scene_graph,
    save_json,
)

__all__ = [
    "DEFAULT_NOOP_TEXTS",
    "build_vocabularies",
    "discover_v4rl_records",
    "load_json",
    "parse_paddleocr",
    "parse_scene_graph",
    "save_json",
]
