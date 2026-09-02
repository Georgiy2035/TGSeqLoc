"""V4RL source discovery, parsing, ground truth, and temporal splits."""

from __future__ import annotations

import bisect
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .schema import FrameRecord

FRAME_RE = re.compile(r"^(?P<index>\d{6})_(?P<timestamp>\d{19})\.(?:png|jpg|jpeg)$", re.I)
DEFAULT_NOOP_TEXTS = frozenset(
    {"", "none", "null", "n/a", "no text", "[no text]", "<no_text>", "<no text>"}
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(target)


def discover_v4rl_records(
    dataset_root: str | Path,
    ocr_root_template: str,
    graph_root_template: str,
    sequences: Sequence[str] = ("seq1", "seq2"),
    *,
    chunk_size: int = 200,
    require_inputs: bool = True,
) -> list[FrameRecord]:
    """Discover ordered frames and their chunked graph/OCR sidecars."""

    records: list[FrameRecord] = []
    for sequence in sequences:
        frame_root = Path(dataset_root) / f"{sequence}_frames"
        sequence_records: list[FrameRecord] = []
        for image_path in sorted(frame_root.iterdir() if frame_root.exists() else ()):
            match = FRAME_RE.match(image_path.name)
            if match is None:
                continue
            index = int(match["index"])
            timestamp = int(match["timestamp"])
            chunk, local = divmod(index, chunk_size)
            ocr_path = (
                Path(ocr_root_template.format(sequence=sequence, chunk=chunk))
                / image_path.stem
                / "paddleocr_v5.json"
            )
            graph_path = (
                Path(graph_root_template.format(sequence=sequence, chunk=chunk))
                / f"{sequence}_c{chunk:03d}"
                / f"{local:06d}.json"
            )
            if require_inputs and not ocr_path.is_file():
                raise FileNotFoundError(f"Missing OCR for {sequence} frame {index}: {ocr_path}")
            if require_inputs and not graph_path.is_file():
                raise FileNotFoundError(
                    f"Missing scene graph for {sequence} frame {index}: {graph_path}"
                )
            sequence_records.append(
                FrameRecord(
                    sequence, index, timestamp, image_path.stem,
                    image_path, ocr_path, graph_path,
                )
            )
        if not sequence_records:
            raise RuntimeError(f"No V4RL frames found in {frame_root}")
        indices = [record.index for record in sequence_records]
        if indices != list(range(len(indices))):
            raise ValueError(f"{sequence} frame indices are not contiguous from zero: {indices[:5]}")
        timestamps = [record.timestamp for record in sequence_records]
        if timestamps != sorted(timestamps):
            raise ValueError(f"{sequence} timestamps are not monotonic")
        records.extend(sequence_records)
    return records


def _prediction_rows(data: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    rows = data.get("predictions", data.get("results", data.get("ocr", [])))
    if isinstance(rows, Mapping):
        rows = rows.get("predictions", rows.get("results", []))
    return rows if isinstance(rows, list) else ()


def parse_paddleocr(
    path: str | Path,
    confidence_threshold: float = 0.0,
    *,
    noop_texts: Iterable[str] = DEFAULT_NOOP_TEXTS,
    prediction_filter: Callable[[Mapping[str, Any], float], bool] | None = None,
) -> tuple[list[list[float]], list[str]]:
    """Parse PaddleOCR JSON into normalized xyxy boxes and useful strings."""

    data = load_json(path)
    width = float(data.get("image_width", data.get("width", 0)))
    height = float(data.get("image_height", data.get("height", 0)))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid OCR image dimensions in {path}")
    noops = {str(value).strip().casefold() for value in noop_texts}
    boxes: list[list[float]] = []
    texts: list[str] = []
    for row in _prediction_rows(data):
        confidence = float(row.get("confidence", row.get("score", 0.0)))
        if prediction_filter is not None:
            if not prediction_filter(row, confidence_threshold):
                continue
        elif confidence < confidence_threshold:
            continue
        text = str(row.get("text", row.get("rec_text", ""))).strip()
        if text.casefold() in noops:
            continue
        raw_box = row.get("bbox", row.get("box"))
        if not isinstance(raw_box, (list, tuple)):
            continue
        if len(raw_box) == 4 and all(isinstance(value, (int, float)) for value in raw_box):
            x, y, box_width, box_height = map(float, raw_box)
            x1, y1, x2, y2 = x, y, x + box_width, y + box_height
        elif len(raw_box) >= 4:
            points = [point for point in raw_box if isinstance(point, (list, tuple)) and len(point) >= 2]
            if not points:
                continue
            xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        normalized = [
            min(1.0, max(0.0, x1 / width)),
            min(1.0, max(0.0, y1 / height)),
            min(1.0, max(0.0, x2 / width)),
            min(1.0, max(0.0, y2 / height)),
        ]
        if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
            boxes.append(normalized)
            texts.append(text)
    return boxes, texts


def parse_scene_graph(
    path: str | Path, class_to_idx: Mapping[str, int]
) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]], int]:
    """Parse an external node-link scene graph, dropping dangling edges."""

    graph = load_json(path)
    nodes: list[dict[str, Any]] = []
    id_to_index: dict[Any, int] = {}
    for raw in graph.get("nodes", []):
        payload = raw.get("data", raw)
        box = payload.get("bbox_2d", {}).get("xyxy", payload.get("bbox"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [min(1.0, max(0.0, float(value))) for value in box]
        if x2 <= x1 or y2 <= y1:
            continue
        name = str(payload.get("class_name", payload.get("label", "unknown"))).strip() or "unknown"
        identifier = raw.get("id", payload.get("id"))
        id_to_index[identifier] = len(nodes)
        nodes.append(
            {
                "id": identifier,
                "class_name": name,
                "class_idx": int(class_to_idx.get(name, class_to_idx.get("unknown", 0))),
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "wh": [x2 - x1, y2 - y1],
                "xyxy": [x1, y1, x2, y2],
            }
        )
    edges: list[tuple[int, int, str]] = []
    dropped = 0
    for raw in graph.get("links", graph.get("edges", [])):
        source, target = id_to_index.get(raw.get("source")), id_to_index.get(raw.get("target"))
        if source is None or target is None:
            dropped += 1
            continue
        label = str(raw.get("label", raw.get("type", "unknown"))).strip() or "unknown"
        edges.append((source, target, label))
    return nodes, edges, dropped


def build_vocabularies(
    records: Iterable[FrameRecord],
) -> tuple[dict[str, int], dict[str, int]]:
    classes, labels = {"unknown"}, {"unknown"}
    for record in records:
        graph = load_json(record.graph_path)
        for raw in graph.get("nodes", []):
            payload = raw.get("data", raw)
            classes.add(str(payload.get("class_name", payload.get("label", "unknown"))).strip() or "unknown")
        for raw in graph.get("links", graph.get("edges", [])):
            labels.add(str(raw.get("label", raw.get("type", "unknown"))).strip() or "unknown")
    class_to_idx = {"unknown": 0, **{name: index for index, name in enumerate(sorted(classes - {"unknown"}), 1)}}
    edge_to_idx = {"unknown": 0, **{name: index for index, name in enumerate(sorted(labels - {"unknown"}), 1)}}
    return class_to_idx, edge_to_idx


def _nearest(records: Sequence[FrameRecord], timestamp: int, tolerance_ns: int) -> tuple[FrameRecord, int, bool]:
    stamps = [record.timestamp for record in records]
    position = bisect.bisect_left(stamps, timestamp)
    candidates = records[max(0, position - 1): min(len(records), position + 1)]
    nearest = min(candidates, key=lambda record: abs(record.timestamp - timestamp))
    delta = abs(nearest.timestamp - timestamp)
    if delta > tolerance_ns:
        raise ValueError(f"GT timestamp {timestamp} is {delta} ns from nearest frame")
    return nearest, delta, timestamp < stamps[0] or timestamp > stamps[-1]


def build_gt_mapping(
    gt_path: str | Path,
    records: Sequence[FrameRecord],
    tolerance_ns: int = 50_000_000,
    *,
    reference_sequence: str = "seq1",
    query_sequence: str = "seq2",
) -> dict[str, Any]:
    by_sequence = {
        sequence: sorted((r for r in records if r.sequence == sequence), key=lambda r: r.timestamp)
        for sequence in (reference_sequence, query_sequence)
    }
    reference, query = by_sequence[reference_sequence], by_sequence[query_sequence]
    loops, singles = [], []
    for line_number, raw_line in enumerate(
        Path(gt_path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if any(not field.strip() for field in fields):
            raise ValueError(
                f"Invalid GT line {line_number}: empty CSV field: {raw_line}"
            )
        try:
            values = [int(field.strip()) for field in fields]
        except ValueError as exc:
            raise ValueError(
                f"Invalid GT line {line_number}: expected integer CSV fields: {raw_line}"
            ) from exc
        if len(values) == 1:
            singles.append(values[0])
            continue
        if len(values) != 3:
            raise ValueError(
                f"Invalid GT line {line_number}: expected 1 or 3 fields: {raw_line}"
            )
        query_ts, start_ts, end_ts = values
        query_record, query_delta, query_clamped = _nearest(query, query_ts, tolerance_ns)

        def reference_value(timestamp: int) -> tuple[FrameRecord | None, int | None]:
            if not reference or timestamp < reference[0].timestamp or timestamp > reference[-1].timestamp:
                return None, None
            record, delta, _ = _nearest(reference, timestamp, tolerance_ns)
            return record, delta

        start, start_delta = reference_value(start_ts)
        end, end_delta = reference_value(end_ts)
        loops.append(
            {
                "query_timestamp": query_ts, "query_index": query_record.index,
                "query_stem": query_record.stem, "query_timestamp_delta_ns": query_delta,
                "query_clamped": query_clamped,
                "reference_start_timestamp": start_ts,
                "reference_start_index": start.index if start else None,
                "reference_start_stem": start.stem if start else None,
                "reference_start_timestamp_delta_ns": start_delta,
                "reference_start_available": start is not None,
                "reference_end_timestamp": end_ts,
                "reference_end_index": end.index if end else None,
                "reference_end_stem": end.stem if end else None,
                "reference_end_timestamp_delta_ns": end_delta,
                "reference_end_available": end is not None,
            }
        )
    return {
        "reference_sequence": reference_sequence, "query_sequence": query_sequence,
        "timestamp_tolerance_ns": tolerance_ns,
        "single_keyframe_timestamps": singles, "loops": loops,
    }


def build_positive_intervals(
    mapping: Mapping[str, Any],
    query_records: Sequence[FrameRecord | Path] | None = None,
    *,
    query_paths: Sequence[Path] | None = None,
    interpolate: bool = False,
) -> dict[int, list[int]]:
    positives: dict[int, set[int]] = {}
    for loop in mapping.get("loops", []):
        if not loop.get("reference_start_available") or not loop.get("reference_end_available"):
            continue
        start, end = int(loop["reference_start_index"]), int(loop["reference_end_index"])
        start, end = min(start, end), max(start, end)
        positives.setdefault(int(loop["query_index"]), set()).update(range(start, end + 1))
    direct = {index: sorted(values) for index, values in positives.items()}
    if not interpolate:
        return direct
    if query_records is None:
        query_records = query_paths
    if not query_records:
        raise ValueError("query_records are required for interpolation")
    timestamps = [
        item.timestamp if isinstance(item, FrameRecord) else int(Path(item).stem.split("_", 1)[1])
        for item in query_records
    ]
    keyframes: dict[int, list[int] | None] = {
        int(value): None for value in mapping.get("single_keyframe_timestamps", [])
        if timestamps[0] <= int(value) <= timestamps[-1]
    }
    for loop in mapping.get("loops", []):
        keyframes[int(loop["query_timestamp"])] = direct.get(int(loop["query_index"]))
    if not keyframes:
        return {}
    ordered = sorted(keyframes)
    result: dict[int, list[int]] = {}
    for index, timestamp in enumerate(timestamps):
        position = bisect.bisect_left(ordered, timestamp)
        candidates = ordered[max(0, position - 1): min(len(ordered), position + 1)]
        interval = keyframes[min(candidates, key=lambda value: abs(value - timestamp))]
        if interval:
            result[index] = interval
    return result


def build_temporal_split(
    query_count: int,
    positives: Mapping[int, Sequence[int]],
    *,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    if not 0 < test_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("split ratios must satisfy 0 < test < 1 and 0 <= validation < 1")
    test_start = int(query_count * (1 - test_ratio))
    before = sorted(index for index in positives if index < test_start)
    validation_count = min(len(before), max(1, int(len(before) * validation_ratio))) if validation_ratio else 0
    train = before[:-validation_count] if validation_count else before
    validation = before[-validation_count:] if validation_count else []
    test = sorted(index for index in positives if index >= test_start)
    if require_nonempty and (not train or not test):
        raise RuntimeError("Temporal split produced an empty annotated train/test set")
    return {
        "positives": {int(k): list(v) for k, v in positives.items()},
        "train_query_indices": train, "validation_query_indices": validation,
        "test_query_indices": test, "test_start": test_start,
    }


def build_split(
    data_root: str | Path,
    mapping: Mapping[str, Any],
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    interpolate_gt: bool = False,
    *,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    """Build the path-aware temporal split consumed by training code."""

    root = Path(data_root)
    reference_sequence = str(mapping.get("reference_sequence", "seq1"))
    query_sequence = str(mapping.get("query_sequence", "seq2"))
    database_paths = sorted((root / reference_sequence).glob("*.pt"))
    query_paths = sorted((root / query_sequence).glob("*.pt"))
    if not database_paths or not query_paths:
        raise RuntimeError(f"Missing prepared graph files under {root}")
    positives = build_positive_intervals(
        mapping, query_paths=query_paths, interpolate=interpolate_gt
    )
    split = build_temporal_split(
        len(query_paths), positives, test_ratio=test_ratio,
        validation_ratio=validation_ratio, require_nonempty=require_nonempty,
    )
    return {
        "database_paths": database_paths,
        "query_paths": query_paths,
        **split,
    }
