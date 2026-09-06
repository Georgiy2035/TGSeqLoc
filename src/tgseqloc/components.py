"""Registration of the component implementations shipped with the MVP."""

from __future__ import annotations

from tgseqloc.data.v4rl import parse_paddleocr, parse_scene_graph
from tgseqloc.evaluation import RecallAtK, Retriever
from tgseqloc.inference.dynamics import build_mask_ioa
from tgseqloc.inference.ocr import build_paddleocr_v5
from tgseqloc.inference.segmentation import build_yolo_seg
from tgseqloc.models import GATGraphEncoder
from tgseqloc.preparation import build_fused_graph, process_v4rl
from tgseqloc.preparation.text import build_char_ngram_encoder, build_multilingual_e5
from tgseqloc.registry import Registry, registry
from tgseqloc.training import mine_hard_negatives


def confidence_filter(sample: dict, threshold: float = 0.0) -> bool:
    """Return whether an OCR prediction passes the configured confidence."""

    return float(sample.get("confidence", sample.get("score", 0.0))) >= threshold


def register_builtin_components(target: Registry = registry) -> Registry:
    """Register all runnable MVP implementations once and return the registry."""

    entries = (
        ("source", "precomputed_paddleocr", parse_paddleocr),
        ("source", "external_json", parse_scene_graph),
        # Same models as the precomputed source above, run here instead of read
        # from JSON another tool produced.
        ("ocr", "paddleocr_v5", build_paddleocr_v5),
        ("segmenter", "yolo_seg", build_yolo_seg),
        ("text_dynamics", "mask_ioa", build_mask_ioa),
        ("filter", "confidence", confidence_filter),
        ("encoder", "multilingual_e5", build_multilingual_e5),
        # Lexical alternative: no weights, and character n-grams degrade
        # gracefully where a subword tokenizer breaks on OCR errors.
        ("encoder", "char_ngram", build_char_ngram_encoder),
        ("fusion", "text_nodes", build_fused_graph),
        ("graph_encoder", "gat", GATGraphEncoder),
        ("dataset", "v4rl", process_v4rl),
        ("miner", "hard_negative", mine_hard_negatives),
        ("retriever", "faiss_cosine", Retriever),
        ("metric", "recall_at_k", RecallAtK),
    )
    for kind, name, factory in entries:
        if not target.contains(kind, name):
            target.add(kind, name, factory)
    return target


def component_map(target: Registry = registry) -> dict[str, list[str]]:
    """Return available implementation names grouped by extension point."""

    register_builtin_components(target)
    return {kind: list(target.available(kind)) for kind in sorted(target.KINDS)}

