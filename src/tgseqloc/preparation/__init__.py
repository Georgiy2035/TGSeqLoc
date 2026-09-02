"""Public graph-preparation API."""

from .fusion import (
    CONNECTION_STRATEGIES,
    build_fused_graph,
    compact_edge_features,
    select_text_connections,
)
from .text import FrozenTextEncoder, TextEncoder
from .v4rl import (
    backend_identities,
    build_preprocess_fingerprint,
    build_source_identities,
    callable_identity,
    encoder_identity,
    frame_fingerprint,
    is_compatible_output,
    process_dataset,
    process_v4rl,
    resolve_cache_policy,
    source_file_identity,
)

__all__ = [
    "CONNECTION_STRATEGIES",
    "FrozenTextEncoder",
    "TextEncoder",
    "backend_identities",
    "build_fused_graph",
    "build_preprocess_fingerprint",
    "build_source_identities",
    "callable_identity",
    "compact_edge_features",
    "encoder_identity",
    "frame_fingerprint",
    "is_compatible_output",
    "process_dataset",
    "process_v4rl",
    "resolve_cache_policy",
    "select_text_connections",
    "source_file_identity",
]
