"""Registration of the component implementations shipped with TGSeqLoc.

Configuration selects implementations by name (``preprocess.text_encoder``,
``model.graph_encoder``, ``training.miner``, ...). The recognizers, segmenters
and graph generators used to annotate data are not registered here: they run in
their own environments through the annotation scripts.
"""

from __future__ import annotations

from typing import Callable, Iterable

from tgseqloc.registry import Registry, registry

Entry = tuple[str, str, object]

GROUPS = ("graph",)


def _graph_entries() -> tuple[Entry, ...]:
    from tgseqloc.data.v4rl import parse_scene_graph
    from tgseqloc.evaluation import RecallAtK, Retriever
    from tgseqloc.models import GATGraphEncoder
    from tgseqloc.preparation import build_fused_graph, process_v4rl
    from tgseqloc.preparation.robotcar import process_robotcar
    from tgseqloc.preparation.tgseqloc_street import process_tgseqloc_street
    from tgseqloc.preparation.text import (
        build_char_ngram_encoder,
        build_multilingual_e5,
    )
    from tgseqloc.training import mine_hard_negatives
    from tgseqloc.training.mining import random_negatives

    return (
        ("source", "external_json", parse_scene_graph),
        # Character n-grams hashed into 64 buckets (Sec. III-B): no weights, and
        # a misread character changes only the n-grams that contain it.
        ("encoder", "char_ngram", build_char_ngram_encoder),
        # Frozen sentence encoders, compared in the string-encoder ablation.
        ("encoder", "multilingual_e5", build_multilingual_e5),
        ("fusion", "text_nodes", build_fused_graph),
        ("graph_encoder", "gat", GATGraphEncoder),
        ("dataset", "v4rl", process_v4rl),
        ("dataset", "robotcar", process_robotcar),
        ("dataset", "tgseqloc_street", process_tgseqloc_street),
        ("miner", "hard_negative", mine_hard_negatives),
        ("miner", "random", random_negatives),
        ("retriever", "faiss_cosine", Retriever),
        ("metric", "recall_at_k", RecallAtK),
    )


_GROUPS: dict[str, Callable[[], tuple[Entry, ...]]] = {"graph": _graph_entries}


def register_builtin_components(
    target: Registry = registry, *, groups: Iterable[str] | None = None
) -> Registry:
    """Register the shipped implementations and return the registry."""

    selected = tuple(groups) if groups is not None else GROUPS
    unknown = sorted(set(selected) - set(_GROUPS))
    if unknown:
        raise KeyError(
            f"unknown component group(s): {', '.join(unknown)}; "
            f"available: {', '.join(GROUPS)}"
        )
    for group in selected:
        for kind, name, factory in _GROUPS[group]():
            if not target.contains(kind, name):
                target.add(kind, name, factory)
    return target


def component_map(target: Registry = registry) -> dict[str, list[str]]:
    """Return available implementation names grouped by extension point."""

    register_builtin_components(target)
    return {kind: list(target.available(kind)) for kind in sorted(target.KINDS)}
