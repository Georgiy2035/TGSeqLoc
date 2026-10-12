"""Registration of the component implementations shipped with the MVP.

Imports happen per group, inside the registering function, so that populating
the registry for one stage does not require the dependencies of every other.
That is what lets the OCR stage run in an environment built for PaddleOCR,
where torch is deliberately absent -- importing a graph encoder there would
fail, and the stage does not need one.
"""

from __future__ import annotations

from typing import Callable, Iterable

from tgseqloc.registry import Registry, registry

Entry = tuple[str, str, object]

#: Which groups a caller can ask for. ``inference`` covers the model stages and
#: needs no torch beyond what the chosen backend itself imports.
GROUPS = ("inference", "graph")


def confidence_filter(sample: dict, threshold: float = 0.0) -> bool:
    """Return whether an OCR prediction passes the configured confidence.

    A generative recognizer reports no confidence at all, so an absent score is
    not a zero score: treating it as zero would let any positive threshold
    silently empty that recognizer's arm of a comparison. Unreported confidence
    passes every threshold instead, matching :func:`~tgseqloc.data.v4rl.parse_paddleocr`.
    """

    reported = sample.get("confidence", sample.get("score"))
    if reported is None:
        return True
    return float(reported) >= threshold


def _inference_entries() -> tuple[Entry, ...]:
    from tgseqloc.inference.dynamics import build_mask_ioa
    from tgseqloc.inference.ocr import build_paddleocr_v5
    from tgseqloc.inference.segmentation import build_yolo_seg

    return (
        ("ocr", "paddleocr_v5", build_paddleocr_v5),
        ("segmenter", "yolo_seg", build_yolo_seg),
        ("text_dynamics", "mask_ioa", build_mask_ioa),
        ("filter", "confidence", confidence_filter),
    )


def _graph_entries() -> tuple[Entry, ...]:
    from tgseqloc.data.v4rl import parse_paddleocr, parse_scene_graph
    from tgseqloc.evaluation import RecallAtK, Retriever
    from tgseqloc.models import GATGraphEncoder
    from tgseqloc.preparation import build_fused_graph, process_v4rl
    from tgseqloc.preparation.pervomay import process_pervomay
    from tgseqloc.preparation.robotcar import process_robotcar
    from tgseqloc.preparation.spelling import build_symspell
    from tgseqloc.preparation.text import (
        build_char_ngram_encoder,
        build_multilingual_e5,
    )
    from tgseqloc.training import mine_hard_negatives
    from tgseqloc.training.mining import random_negatives

    return (
        ("source", "precomputed_paddleocr", parse_paddleocr),
        ("source", "external_json", parse_scene_graph),
        ("text_correction", "symspell", build_symspell),
        ("encoder", "multilingual_e5", build_multilingual_e5),
        # Lexical alternative: no weights, and character n-grams degrade
        # gracefully where a subword tokenizer breaks on OCR errors.
        ("encoder", "char_ngram", build_char_ngram_encoder),
        ("fusion", "text_nodes", build_fused_graph),
        ("graph_encoder", "gat", GATGraphEncoder),
        ("dataset", "v4rl", process_v4rl),
        ("dataset", "robotcar", process_robotcar),
        ("dataset", "pervomay", process_pervomay),
        ("miner", "hard_negative", mine_hard_negatives),
        ("miner", "random", random_negatives),
        ("retriever", "faiss_cosine", Retriever),
        ("metric", "recall_at_k", RecallAtK),
    )


_GROUPS: dict[str, Callable[[], tuple[Entry, ...]]] = {
    "inference": _inference_entries,
    "graph": _graph_entries,
}


def register_builtin_components(
    target: Registry = registry, *, groups: Iterable[str] | None = None
) -> Registry:
    """Register the shipped implementations and return the registry.

    ``groups`` limits which are imported. ``("inference",)`` registers the model
    stages alone, which is what an environment holding one model but not the
    graph stack can actually import.
    """

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
