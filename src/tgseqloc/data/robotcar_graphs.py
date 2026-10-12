"""Scene graphs for Oxford RobotCar, converted from the generator's own format.

The generator emits one JSONL record per frame whose ``predict`` field holds a
scene graph as text -- an object table and a relation table in pixel
coordinates -- while the rest of the pipeline consumes node-link JSON with
boxes normalized to the unit square. This module is that translation, and
nothing else: it does not filter, merge or repair graphs beyond what is needed
to make a well-formed file.

The object table's columns are read from its own header rather than assumed.
Two generations of the generator exist side by side: the first wrote
``{id,name,x1,y1,x2,y2}``, the second -- Gemini 3.1 Flash Lite through
OpenRouter -- puts ``color`` and ``material`` between the name and the box.
Locating columns by name reads both, and any column beyond the six the
pipeline needs is kept on the node as an attribute rather than discarded. A
graph in the first layout converts to exactly the file it did before.

Two properties of the generator's output shape the code. A small fraction of
records carry no table at all -- repeated brace fragments in the first
generation (300 of 34,838 base records), empty or failed completions in the
second (19 of 56,487); those are counted and skipped rather than half-parsed.
And the record names a frame by its position inside a chunk of resized images,
so recovering the original timestamp -- which is how every other stage
identifies a frame -- requires the chunk's own ``frame_index.json``.
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

#: Columns every object row must provide; any others become node attributes.
REQUIRED_COLUMNS = ("id", "name", "x1", "y1", "x2", "y2")

_OBJECT_HEADER = re.compile(r"obj\[(\d+)\]\{([^}]*)\}:")
_RELATION_HEADER = re.compile(r"rel\[(\d+)\]\{subj,pred,obj\}:")
_RELATION_ROW = re.compile(r"^\s*(\d+)\s*,\s*([^,]+?)\s*,\s*(\d+)\s*$")
# The same acceptance as the fixed-layout row pattern these replace, so a
# graph in the first layout keeps or drops exactly the rows it did before.
_INTEGER = re.compile(r"^\d+$")
_NUMBER = re.compile(r"^-?[\d.]+$")

#: Beyond this many braces the output is a repetition loop, not a table.
_DEGENERATE_BRACES = 50

_LAYOUT_PREFIX = "layout:"


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
    layouts: Counter = field(default_factory=Counter)

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
            "object_layouts": dict(self.layouts),
        }


def parse_scene_graph_text(
    text: str | None,
    image_size: tuple[float, float] = GENERATOR_IMAGE_SIZE,
    counters: Counter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Parse one generated graph into normalized nodes and links.

    Returns ``None`` when the output is degenerate, carries no object table, or
    its table lacks one of :data:`REQUIRED_COLUMNS`; the caller counts these
    separately, since an empty graph and an unusable one are different
    failures and should not be summed.

    ``counters``, when given, receives ``dropped_boxes`` for object rows that
    did not become nodes, ``dropped_relations`` for relation rows that did not
    become links, and one ``layout:<columns>`` entry naming the table's header.
    """

    if not text or not text.strip():
        return None
    header = _OBJECT_HEADER.search(text)
    if header is None:
        return None
    if text.count("{") > _DEGENERATE_BRACES:
        return None
    columns = [column.strip() for column in header.group(2).split(",")]
    if any(required not in columns for required in REQUIRED_COLUMNS):
        return None
    position = {name: columns.index(name) for name in REQUIRED_COLUMNS}
    extra = [(index, name) for index, name in enumerate(columns) if name not in REQUIRED_COLUMNS]
    tally = counters if counters is not None else Counter()
    tally[_LAYOUT_PREFIX + ",".join(columns)] += 1

    width, height = image_size
    relation_header = _RELATION_HEADER.search(text)
    object_body = text[header.end(): relation_header.start() if relation_header else len(text)]

    nodes: list[dict[str, Any]] = []
    index_by_id: dict[int, int] = {}
    for line in object_body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("<"):
            continue
        raw = line.split(",")
        if len(raw) != len(columns):
            tally["dropped_boxes"] += 1
            continue
        fields = [value.strip() for value in raw]
        coords = [fields[position[name]] for name in ("x1", "y1", "x2", "y2")]
        if (
            not _INTEGER.match(fields[position["id"]])
            or raw[position["name"]] == ""
            or not all(_NUMBER.match(value) for value in coords)
        ):
            tally["dropped_boxes"] += 1
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in coords)
        except ValueError:
            tally["dropped_boxes"] += 1
            continue
        identifier = int(fields[position["id"]])
        x1, x2 = sorted((x1 / width, x2 / width))
        y1, y2 = sorted((y1 / height, y2 / height))
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 <= x1 or y2 <= y1:
            # A zero-area box carries no location; keeping it would add a node
            # whose geometry features are meaningless.
            tally["dropped_boxes"] += 1
            continue
        if identifier in index_by_id:
            tally["dropped_boxes"] += 1
            continue
        index_by_id[identifier] = len(nodes)
        data: dict[str, Any] = {
            "global_id": identifier,
            "class_name": fields[position["name"]] or "unknown",
            "bbox_2d": {
                "xyxy": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            },
        }
        if extra:
            data["attributes"] = {name: fields[index] for index, name in extra}
        nodes.append({"id": identifier, "data": data})

    links: list[dict[str, Any]] = []
    if relation_header is not None:
        for line in text[relation_header.end():].splitlines():
            row = _RELATION_ROW.match(line)
            if row is None:
                if _INTEGER.match(line.split(",")[0].strip()):
                    tally["dropped_relations"] += 1
                continue
            source, target = int(row.group(1)), int(row.group(3))
            if source not in index_by_id or target not in index_by_id:
                tally["dropped_relations"] += 1
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

        # A failed completion stores null here; it is a missing graph, not a
        # reason to stop the whole conversion.
        predict = record.get("predict") or ""
        tally: Counter = Counter()
        parsed = parse_scene_graph_text(predict, image_size, tally)
        stats.dropped_boxes += tally["dropped_boxes"]
        stats.dropped_relations += tally["dropped_relations"]
        for key, value in tally.items():
            if key.startswith(_LAYOUT_PREFIX):
                stats.layouts[key[len(_LAYOUT_PREFIX):]] += value
        if parsed is None:
            if predict.count("{") > _DEGENERATE_BRACES:
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
