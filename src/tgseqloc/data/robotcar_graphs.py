"""Scene graphs for Oxford RobotCar, converted from the generator's own format.

The generator emits one JSONL record per frame whose ``predict`` field holds a
scene graph as text -- an object table and a relation table in pixel
coordinates -- while the rest of the pipeline consumes node-link JSON with
boxes normalized to the unit square. This module is that translation, and
nothing else: it does not filter, merge or repair graphs beyond what is needed
to make a well-formed file.

Two properties of the generator's output shape the code. A small fraction of
records degenerate into repeated brace fragments instead of a table (0.9% of
the base split, none of the query split); those are counted and skipped rather
than half-parsed. And the record names a frame by its position inside a chunk
of resized images, so recovering the original timestamp -- which is how every
other stage identifies a frame -- requires the chunk's own ``frame_index.json``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Size the generator was prompted with; boxes come back in these pixels.
GENERATOR_IMAGE_SIZE = (640.0, 480.0)

_OBJECT_HEADER = re.compile(r"obj\[(\d+)\]\{id,name,x1,y1,x2,y2\}:")
_RELATION_HEADER = re.compile(r"rel\[(\d+)\]\{subj,pred,obj\}:")
_OBJECT_ROW = re.compile(
    r"^\s*(\d+)\s*,\s*([^,]+?)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*$"
)
_RELATION_ROW = re.compile(r"^\s*(\d+)\s*,\s*([^,]+?)\s*,\s*(\d+)\s*$")

#: Beyond this many braces the output is a repetition loop, not a table.
_DEGENERATE_BRACES = 50


@dataclass(slots=True)
class ConversionStats:
    """What the conversion did, so a silent partial run is impossible to miss."""

    records: int = 0
    written: int = 0
    degenerate: int = 0
    unparsed: int = 0
    empty: int = 0
    unknown_frame: int = 0
    dropped_boxes: int = 0
    dropped_relations: int = 0
    classes: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "written": self.written,
            "degenerate": self.degenerate,
            "unparsed": self.unparsed,
            "empty": self.empty,
            "unknown_frame": self.unknown_frame,
            "dropped_boxes": self.dropped_boxes,
            "dropped_relations": self.dropped_relations,
            "distinct_classes": len(self.classes),
        }


def parse_scene_graph_text(
    text: str, image_size: tuple[float, float] = GENERATOR_IMAGE_SIZE
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Parse one generated graph into normalized nodes and links.

    Returns ``None`` when the output is degenerate or carries no object table,
    which the caller counts separately -- an empty graph and an unusable one
    are different failures and should not be summed.
    """

    if not text or not text.strip():
        return None
    header = _OBJECT_HEADER.search(text)
    if header is None:
        return None
    if text.count("{") > _DEGENERATE_BRACES:
        return None

    width, height = image_size
    relation_header = _RELATION_HEADER.search(text)
    object_body = text[header.end(): relation_header.start() if relation_header else len(text)]

    nodes: list[dict[str, Any]] = []
    index_by_id: dict[int, int] = {}
    for line in object_body.splitlines():
        row = _OBJECT_ROW.match(line)
        if row is None:
            continue
        identifier = int(row.group(1))
        x1, y1, x2, y2 = (float(row.group(position)) for position in (3, 4, 5, 6))
        x1, x2 = sorted((x1 / width, x2 / width))
        y1, y2 = sorted((y1 / height, y2 / height))
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 <= x1 or y2 <= y1:
            # A zero-area box carries no location; keeping it would add a node
            # whose geometry features are meaningless.
            continue
        if identifier in index_by_id:
            continue
        index_by_id[identifier] = len(nodes)
        nodes.append(
            {
                "id": identifier,
                "data": {
                    "global_id": identifier,
                    "class_name": row.group(2).strip() or "unknown",
                    "bbox_2d": {
                        "xyxy": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    },
                },
            }
        )

    links: list[dict[str, Any]] = []
    if relation_header is not None:
        for line in text[relation_header.end():].splitlines():
            row = _RELATION_ROW.match(line)
            if row is None:
                continue
            source, target = int(row.group(1)), int(row.group(3))
            if source not in index_by_id or target not in index_by_id:
                continue
            links.append(
                {"source": source, "target": target, "label": row.group(2).strip() or "unknown"}
            )
    return nodes, links


def build_frame_index(chunk_roots: Iterable[Path]) -> dict[tuple[str, str], str]:
    """Map (chunk name, local index) to the frame's original timestamp stem.

    The generator identifies a frame by its position in a chunk of resized
    images; every other stage identifies it by the timestamp in the original
    file name. Only the chunk's own index knows both.
    """

    mapping: dict[tuple[str, str], str] = {}
    for root in chunk_roots:
        for index_path in sorted(Path(root).glob("*/frame_index.json")):
            chunk = index_path.parent.name
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            for local, original in payload.get("frames", {}).items():
                mapping[(chunk, str(local))] = Path(str(original)).stem
    return mapping


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def convert_split(
    jsonl_path: Path,
    frame_index: dict[tuple[str, str], str],
    output_root: Path,
    *,
    image_size: tuple[float, float] = GENERATOR_IMAGE_SIZE,
) -> ConversionStats:
    """Write one node-link JSON per frame, named by the frame's timestamp."""

    stats = ConversionStats()
    output_root.mkdir(parents=True, exist_ok=True)
    for record in _records(jsonl_path):
        stats.records += 1
        images = record.get("images") or []
        if not images:
            stats.unknown_frame += 1
            continue
        image_path = Path(str(images[0]))
        stem = frame_index.get((image_path.parent.name, image_path.stem))
        if stem is None:
            stats.unknown_frame += 1
            continue

        parsed = parse_scene_graph_text(record.get("predict", ""), image_size)
        if parsed is None:
            if record.get("predict", "").count("{") > _DEGENERATE_BRACES:
                stats.degenerate += 1
            else:
                stats.unparsed += 1
            continue
        nodes, links = parsed
        if not nodes:
            stats.empty += 1
            continue
        for node in nodes:
            stats.classes[node["data"]["class_name"]] += 1

        payload = {
            "directed": True,
            "multigraph": True,
            "graph": {"scene_name": stem, "source": str(jsonl_path)},
            "nodes": nodes,
            "links": links,
        }
        target = output_root / f"{stem}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        temporary.replace(target)
        stats.written += 1
    return stats
