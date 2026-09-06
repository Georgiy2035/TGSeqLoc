"""Readable views of what the pipeline produced.

Prepared graphs are tensors in .pt files and stage outputs are one JSON per
frame, so answering "what did this actually recognize" otherwise means writing
a throwaway script with torch imported. This module answers it directly: what
exists, what ended up in the graphs, and what was dropped on the way.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Iterable

from tgseqloc.stages import STAGES, stage_root


def _prepared_root(config: Any) -> Path:
    return Path(config.dataset.prepared_root) / config.dataset.adapter


def stage_overview(config: Any) -> dict[str, Any]:
    """Per-stage artifact counts, plus how much text each frame carries."""

    overview: dict[str, Any] = {}
    for stage in STAGES:
        root = stage_root(config.dataset.prepared_root, config.dataset.adapter, stage)
        files = [p for p in root.rglob("*.json") if p.name != "manifest.json"]
        entry: dict[str, Any] = {"artifacts": len(files), "root": str(root)}
        if files:
            entry.update(_stage_detail(stage, files))
        overview[stage] = entry
    return overview


def _stage_detail(stage: str, files: list[Path]) -> dict[str, Any]:
    failed = 0
    detections = 0
    frames_with = 0
    masks_with = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "error" in payload:
            failed += 1
            continue
        if stage == "ocr":
            count = len(payload.get("detections", ()))
            detections += count
            frames_with += int(count > 0)
        else:
            masks_with += int(payload.get("dynamic_rle") is not None)
    if stage == "ocr":
        usable = len(files) - failed
        return {
            "failed": failed,
            "frames_with_text": frames_with,
            "share_with_text": round(frames_with / usable, 4) if usable else 0.0,
            "detections": detections,
            "detections_per_frame": round(detections / usable, 3) if usable else 0.0,
        }
    usable = len(files) - failed
    return {
        "failed": failed,
        "frames_with_dynamic_objects": masks_with,
        "share_with_dynamic": round(masks_with / usable, 4) if usable else 0.0,
    }


def graph_overview(config: Any, *, top_texts: int = 15) -> dict[str, Any]:
    """What actually reached the graphs: nodes, edges and the strings kept."""

    import torch

    root = _prepared_root(config)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {"prepared": False, "hint": "run `tgseqloc prepare` first"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    graphs = sorted(root.rglob("*.pt"))
    texts: collections.Counter = collections.Counter()
    nodes = text_nodes = edges = text_edges = 0
    frames_with_text = 0
    for path in graphs:
        graph = torch.load(path, map_location="cpu", weights_only=False)
        nodes += int(graph.x.shape[0])
        count = int(graph.is_text.sum())
        text_nodes += count
        frames_with_text += int(count > 0)
        edges += int(graph.edge_index.shape[1])
        text_edges += int(graph.is_text_edge.sum())
        texts.update(getattr(graph, "text_strings", ()) or ())
    return {
        "prepared": bool(manifest.get("output_complete")),
        "graphs": len(graphs),
        "text_encoder": manifest.get("text_encoder"),
        "text_embedding_dim": manifest.get("text_embedding_dim"),
        "nodes": nodes,
        "text_nodes": text_nodes,
        "frames_with_text": frames_with_text,
        "share_with_text": round(frames_with_text / len(graphs), 4) if graphs else 0.0,
        "text_nodes_per_frame": round(text_nodes / len(graphs), 3) if graphs else 0.0,
        "edges": edges,
        "text_edges": text_edges,
        "top_texts": texts.most_common(top_texts),
    }


def dynamics_overview(config: Any) -> dict[str, Any]:
    """How much recognized text the dynamics stage removed, and what.

    Compares the OCR stage against the strings that survived into the graphs,
    which is the only place the difference is visible after the fact.
    """

    import torch

    if not config.text_dynamics.enabled:
        return {"enabled": False}
    ocr_dir = stage_root(config.dataset.prepared_root, config.dataset.adapter, "ocr")
    files = [p for p in ocr_dir.rglob("*.json") if p.name != "manifest.json"]
    if not files:
        return {"enabled": True, "hint": "run `tgseqloc stage ocr` first"}

    recognized: collections.Counter = collections.Counter()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "error" in payload:
            continue
        recognized.update(row["text"] for row in payload.get("detections", ()))

    kept: collections.Counter = collections.Counter()
    for path in sorted(_prepared_root(config).rglob("*.pt")):
        graph = torch.load(path, map_location="cpu", weights_only=False)
        kept.update(getattr(graph, "text_strings", ()) or ())

    dropped = recognized - kept
    total = sum(recognized.values())
    return {
        "enabled": True,
        "threshold": config.text_dynamics.params.get("threshold", 0.5),
        "recognized": total,
        "kept": sum(kept.values()),
        "dropped": sum(dropped.values()),
        "share_dropped": round(sum(dropped.values()) / total, 4) if total else 0.0,
        "top_dropped": dropped.most_common(15),
    }


def frame_detail(config: Any, sequence: str, frame_stem: str) -> dict[str, Any]:
    """Everything the pipeline holds about one frame, stage by stage."""

    import torch

    from tgseqloc.stages import read_stage

    detail: dict[str, Any] = {"sequence": sequence, "frame_stem": frame_stem}
    for stage in STAGES:
        root = stage_root(config.dataset.prepared_root, config.dataset.adapter, stage)
        try:
            value = read_stage(stage, root, sequence, frame_stem)
        except (FileNotFoundError, RuntimeError) as error:
            detail[stage] = {"unavailable": str(error)}
            continue
        if stage == "ocr":
            detail[stage] = {
                "image_size": list(value.image_size),
                "model": value.model_identity.get("backend"),
                "detections": [
                    {
                        "text": d.text,
                        "confidence": round(d.confidence, 4),
                        "box": [round(v, 4) for v in d.box],
                        "dynamic_score": d.dynamic_score,
                    }
                    for d in value.detections
                ],
            }
        else:
            detail[stage] = {
                "image_size": list(value.image_size),
                "model": value.model_identity.get("backend"),
                "has_dynamic_objects": not value.is_empty,
            }

    path = _prepared_root(config) / sequence / f"{frame_stem}.pt"
    if not path.is_file():
        detail["graph"] = {"unavailable": f"{path} is absent; run `tgseqloc prepare`"}
        return detail
    graph = torch.load(path, map_location="cpu", weights_only=False)
    detail["graph"] = {
        "nodes": int(graph.x.shape[0]),
        "object_nodes": int((~graph.is_text).sum()),
        "text_nodes": int(graph.is_text.sum()),
        "edges": int(graph.edge_index.shape[1]),
        "text_edges": int(graph.is_text_edge.sum()),
        "texts": list(getattr(graph, "text_strings", ()) or ()),
        "schema_version": int(getattr(graph, "schema_version", 0)),
        "fingerprint": str(getattr(graph, "preprocess_fingerprint", ""))[:16],
    }
    return detail


def inspect(config: Any, *, top_texts: int = 15) -> dict[str, Any]:
    """Full summary: stages, graphs, and what the dynamics stage removed."""

    return {
        "prepared_root": str(_prepared_root(config)),
        "stages": stage_overview(config),
        "graphs": graph_overview(config, top_texts=top_texts),
        "dynamics": dynamics_overview(config),
    }


def format_summary(report: dict[str, Any]) -> str:
    """Render the summary as lines rather than JSON, for reading."""

    lines = [f"каталог: {report['prepared_root']}", "", "СТАДИИ"]
    for stage, entry in report["stages"].items():
        if not entry["artifacts"]:
            lines.append(f"  {stage:14} артефактов нет")
            continue
        parts = [f"артефактов {entry['artifacts']}"]
        if entry.get("failed"):
            parts.append(f"с ошибкой {entry['failed']}")
        if stage == "ocr":
            parts.append(f"кадров с текстом {entry['share_with_text']:.1%}")
            parts.append(f"детекций/кадр {entry['detections_per_frame']}")
        else:
            parts.append(f"кадров с динамикой {entry['share_with_dynamic']:.1%}")
        lines.append(f"  {stage:14} " + ", ".join(parts))

    graphs = report["graphs"]
    lines += ["", "ГРАФЫ"]
    if not graphs.get("graphs"):
        lines.append(f"  {graphs.get('hint', 'нет подготовленных графов')}")
    else:
        lines += [
            f"  графов {graphs['graphs']}, завершено: {graphs['prepared']}",
            f"  узлов {graphs['nodes']} (текстовых {graphs['text_nodes']}, "
            f"{graphs['text_nodes_per_frame']} на кадр)",
            f"  кадров с текстом {graphs['share_with_text']:.1%}",
            f"  рёбер {graphs['edges']} (text-object {graphs['text_edges']})",
            f"  энкодер: {graphs['text_encoder']}, размерность {graphs['text_embedding_dim']}",
        ]
        if graphs["top_texts"]:
            shown = ", ".join(f"{t}({c})" for t, c in graphs["top_texts"])
            lines.append(f"  частые строки: {shown}")

    dynamics = report["dynamics"]
    lines += ["", "ОТСЕВ ДИНАМИКИ"]
    if not dynamics.get("enabled"):
        lines.append("  выключен")
    elif "hint" in dynamics:
        lines.append(f"  {dynamics['hint']}")
    else:
        lines.append(
            f"  порог {dynamics['threshold']}: распознано {dynamics['recognized']}, "
            f"оставлено {dynamics['kept']}, отсеяно {dynamics['dropped']} "
            f"({dynamics['share_dropped']:.1%})"
        )
        if dynamics["top_dropped"]:
            shown = ", ".join(f"{t}({c})" for t, c in dynamics["top_dropped"])
            lines.append(f"  что отсеяно: {shown}")
    return "\n".join(lines)
