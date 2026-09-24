"""Acceptance checks of an annotated corpus (Sec. III-F, Fig. 4).

A corpus is accepted when

* coverage -- every frame has both a text and a scene-graph annotation;
* no dynamic objects -- no graph node carries a dynamic class;
* no degenerate boxes -- every node and text box is normalized and non-empty;
* no dangling edges -- every edge connects two existing nodes;
* frame-to-file binding -- the frame identifier recorded inside each file
  matches the file name.

Structural checks do not establish that the annotation is semantically correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tgseqloc.annotation.graphs import MASK2FORMER_DYNAMIC_CLASSES
from tgseqloc.annotation.segmentation import DYNAMIC_CLASSES


@dataclass
class CheckReport:
    frames: int = 0
    missing_text: list[str] = field(default_factory=list)
    missing_graph: list[str] = field(default_factory=list)
    dynamic_nodes: list[str] = field(default_factory=list)
    degenerate_boxes: list[str] = field(default_factory=list)
    dangling_edges: list[str] = field(default_factory=list)
    binding_errors: list[str] = field(default_factory=list)
    empty_graphs: int = 0
    nodes: int = 0
    strings: int = 0

    @property
    def accepted(self) -> bool:
        return not (self.missing_text or self.missing_graph or self.dynamic_nodes
                    or self.degenerate_boxes or self.dangling_edges or self.binding_errors)

    def summary(self) -> dict:
        return {
            "accepted": self.accepted,
            "frames": self.frames,
            "missing_text": len(self.missing_text),
            "missing_graph": len(self.missing_graph),
            "dynamic_nodes": len(self.dynamic_nodes),
            "degenerate_boxes": len(self.degenerate_boxes),
            "dangling_edges": len(self.dangling_edges),
            "binding_errors": len(self.binding_errors),
            "empty_graphs_percent": round(100.0 * self.empty_graphs / max(1, self.frames), 2),
            "nodes_per_frame": round(self.nodes / max(1, self.frames), 2),
            "strings_per_frame": round(self.strings / max(1, self.frames), 2),
            "examples": {
                key: getattr(self, key)[:5]
                for key in ("missing_text", "missing_graph", "dynamic_nodes",
                            "degenerate_boxes", "dangling_edges", "binding_errors")
                if getattr(self, key)
            },
        }


def _valid_box(box) -> bool:
    return (
        isinstance(box, (list, tuple)) and len(box) == 4
        and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in box)
        and box[2] > box[0] and box[3] > box[1]
    )


def check_sequence(text_root: Path, graph_root: Path, report: CheckReport | None = None,
                   dynamic_classes: Iterable[str] | None = None) -> CheckReport:
    """Check one sequence's text and graph annotation folders."""

    report = report or CheckReport()
    dynamic = {c.casefold() for c in (dynamic_classes or (set(DYNAMIC_CLASSES) | MASK2FORMER_DYNAMIC_CLASSES))}
    texts = {p.stem: p for p in Path(text_root).glob("*.json")}
    graphs = {p.stem: p for p in Path(graph_root).glob("*.json")}
    report.missing_text += [f"{graph_root.name}/{s}" for s in sorted(set(graphs) - set(texts))]
    report.missing_graph += [f"{text_root.name}/{s}" for s in sorted(set(texts) - set(graphs))]
    for stem in sorted(set(texts) | set(graphs)):
        report.frames += 1
        if stem in graphs:
            graph = json.loads(graphs[stem].read_text(encoding="utf-8"))
            recorded = (graph.get("graph") or {}).get("scene_name", graph.get("frame_id"))
            if recorded is not None and str(recorded) != stem:
                report.binding_errors.append(f"graph {stem}: records frame {recorded}")
            ids = set()
            nodes = graph.get("nodes", [])
            report.nodes += len(nodes)
            report.empty_graphs += int(not nodes)
            for node in nodes:
                data = node.get("data", node)
                ids.add(node.get("id"))
                if str(data.get("class_name", "")).casefold() in dynamic:
                    report.dynamic_nodes.append(f"{stem}: {data.get('class_name')}")
                box = data.get("bbox_2d", {}).get("xyxy", data.get("bbox"))
                if not _valid_box(box):
                    report.degenerate_boxes.append(f"graph {stem}: node {node.get('id')}")
            for edge in graph.get("links", graph.get("edges", [])):
                if edge.get("source") not in ids or edge.get("target") not in ids:
                    report.dangling_edges.append(f"{stem}: {edge.get('source')}->{edge.get('target')}")
        if stem in texts:
            text = json.loads(texts[stem].read_text(encoding="utf-8"))
            recorded = text.get("frame_id")
            if recorded is not None and str(recorded) != stem:
                report.binding_errors.append(f"text {stem}: records frame {recorded}")
            detections = text.get("detections", [])
            report.strings += len(detections)
            for index, detection in enumerate(detections):
                if not _valid_box(detection.get("box")):
                    report.degenerate_boxes.append(f"text {stem}: detection {index}")
    return report
