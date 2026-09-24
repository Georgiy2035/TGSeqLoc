"""V4RL source discovery, parsing, ground truth, and temporal splits."""

from __future__ import annotations

import bisect
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .formats import FrameRecord

FRAME_RE = re.compile(r"^(?P<index>\d{6})_(?P<timestamp>\d{19})$")
"""Frame identifier of V4RL: ``{index:06d}_{timestamp_ns:019d}``."""


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
    text_root_template: str,
    graph_root_template: str,
    sequences: Sequence[str] = ("seq1", "seq2"),
    *,
    require_text: bool = True,
) -> list[FrameRecord]:
    """Frames of the given sequences, found through their annotations.

    Every frame has a scene-graph annotation ``{frame_id}.json`` under
    ``graph_root_template`` and a text annotation with the same name under
    ``text_root_template`` (both formatted with ``sequence``). Frame indices
    must run from zero without gaps and timestamps must not decrease.
    """

    records: list[FrameRecord] = []
    for sequence in sequences:
        graph_root = Path(graph_root_template.format(sequence=sequence))
        text_root = Path(text_root_template.format(sequence=sequence))
        sequence_records: list[FrameRecord] = []
        for graph_path in sorted(graph_root.glob("*.json") if graph_root.exists() else ()):
            match = FRAME_RE.match(graph_path.stem)
            if match is None:
                continue
            text_path = text_root / graph_path.name
            if require_text and not text_path.is_file():
                raise FileNotFoundError(f"Missing text annotation for {sequence}: {text_path}")
            sequence_records.append(
                FrameRecord(
                    sequence, int(match["index"]), int(match["timestamp"]), graph_path.stem,
                    None, text_path, graph_path,
                )
            )
        if not sequence_records:
            raise RuntimeError(f"No V4RL scene-graph annotations found in {graph_root}")
        indices = [record.index for record in sequence_records]
        if indices != list(range(len(indices))):
            raise ValueError(f"{sequence} frame indices are not contiguous from zero: {indices[:5]}")
        timestamps = [record.timestamp for record in sequence_records]
        if timestamps != sorted(timestamps):
            raise ValueError(f"{sequence} timestamps are not monotonic")
        records.extend(sequence_records)
    return records


def parse_scene_graph(
    path: str | Path,
    class_to_idx: Mapping[str, int],
    dynamic_classes: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]], int]:
    """Parse an external node-link scene graph, dropping dangling edges.

    ``dynamic_classes`` removes ephemeral objects by name. The generator labels
    every object, so a class is a stronger statement about ephemerality than a
    segmentation mask -- and it is what the generator's own static mode uses.
    Edges touching a removed node go with it.
    """

    graph = load_json(path)
    ephemeral = {str(name).strip().casefold() for name in dynamic_classes}
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
        if name.casefold() in ephemeral:
            continue
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
    dynamic_classes: Iterable[str] = (),
) -> tuple[dict[str, int], dict[str, int]]:
    ephemeral = {str(name).strip().casefold() for name in dynamic_classes}
    classes, labels = {"unknown"}, {"unknown"}
    for record in records:
        graph = load_json(record.graph_path)
        for raw in graph.get("nodes", []):
            payload = raw.get("data", raw)
            name = str(payload.get("class_name", payload.get("label", "unknown"))).strip() or "unknown"
            if name.casefold() in ephemeral:
                continue
            classes.add(name)
        for raw in graph.get("links", graph.get("edges", [])):
            labels.add(str(raw.get("label", raw.get("type", "unknown"))).strip() or "unknown")
    class_to_idx = {"unknown": 0, **{name: index for index, name in enumerate(sorted(classes - {"unknown"}), 1)}}
    edge_to_idx = {"unknown": 0, **{name: index for index, name in enumerate(sorted(labels - {"unknown"}), 1)}}
    return class_to_idx, edge_to_idx


def load_vocabularies(manifest_path: str | Path) -> tuple[dict[str, int], dict[str, int]]:
    """Node-class and relation vocabularies of a prepared dataset or a checkpoint.

    ``manifest_path`` is either a preparation ``manifest.json`` or a model
    checkpoint (``.pth``), which carries the manifest of its training data. A
    class absent from that vocabulary is read as ``unknown``.
    """

    if Path(manifest_path).suffix in (".pth", ".pt"):
        import torch

        payload = torch.load(manifest_path, map_location="cpu", weights_only=False)
        manifest = payload["manifest"]
    else:
        manifest = load_json(manifest_path)
    vocabularies = []
    for key in ("node_class_to_idx", "edge_label_to_idx"):
        vocabulary = manifest.get(key)
        if not isinstance(vocabulary, dict) or vocabulary.get("unknown") != 0:
            raise ValueError(f"{manifest_path} has no {key} with 'unknown' at index 0")
        indices = sorted(int(index) for index in vocabulary.values())
        if indices != list(range(len(indices))):
            raise ValueError(f"{manifest_path}: {key} indices are not contiguous from 0")
        vocabularies.append({str(name): int(index) for name, index in vocabulary.items()})
    return vocabularies[0], vocabularies[1]


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


def separate_by_ground_truth(split: Mapping[str, Any]) -> dict[str, Any]:
    """Make the parts of a V4RL split share no ground-truth database frame.

    Validation is the tail of the training segment and the test follows it, so
    queries on either side of a cut look at the same stretch of the reference
    sequence, and a place revisited later in the query sequence can land in the
    test while its first visit trained the model. V4RL gives no metric
    positions, but its ground truth is a set of reference frames per query:
    a validation query is dropped when one of its positives is a positive of a
    training query, and a test query when one of its positives belongs to a
    training or a kept validation query. The dropped queries are recorded.
    """

    positives = {int(k): {int(v) for v in values} for k, values in split["positives"].items()}
    train = [int(i) for i in split["train_query_indices"]]
    seen = set().union(*(positives.get(i, set()) for i in train)) if train else set()
    validation, dropped_validation = [], []
    for index in split["validation_query_indices"]:
        (dropped_validation if positives.get(int(index), set()) & seen else validation).append(int(index))
    seen_before_test = seen.union(*(positives.get(i, set()) for i in validation)) if validation else seen
    test, dropped_test = [], []
    for index in split["test_query_indices"]:
        (dropped_test if positives.get(int(index), set()) & seen_before_test else test).append(int(index))
    if not test:
        raise RuntimeError("separating the split by ground truth left no test query")
    return {
        **dict(split),
        "validation_query_indices": validation,
        "test_query_indices": test,
        "excluded_validation_indices": dropped_validation,
        "excluded_test_indices": dropped_test,
        "split_guard": "ground_truth_overlap",
    }


def build_block_split(
    query_count: int,
    positives: Mapping[int, Sequence[int]],
    *,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    validation_centres: Sequence[float],
) -> dict[str, Any]:
    """A V4RL split whose validation is several blocks spread along the route.

    Recall along the V4RL route changes by tens of points from one hundred
    queries to the next: a model without text found 1 % of one stretch and
    91 % of its neighbour. A validation set that is one contiguous stretch
    therefore measures where it happens to lie, and the epoch it selects is
    close to random. Blocks centred at fixed fractions of the route average
    over places; their positions are a rule, not a choice made by looking at
    results.

    The test is the tail of the query sequence, as in the temporal split, and
    is kept whole. Ground truth is a set of reference frames per query, and the
    parts must not share one: validation queries sharing a frame with the test
    are dropped, then training queries sharing a frame with the test or with
    the kept validation. Training pays for the separation, so that the sets
    that are measured stay the size the protocol names.
    """

    if not validation_centres:
        raise ValueError("validation_centres must name at least one block")
    if not 0 < test_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("split ratios must satisfy 0 < test < 1 and 0 < validation < 1")
    test_start = int(query_count * (1 - test_ratio))
    before = sorted(index for index in positives if index < test_start)
    test = sorted(index for index in positives if index >= test_start)
    if not before or not test:
        raise RuntimeError("the block split needs annotated queries before and after the test start")
    width = max(1, int(len(before) * validation_ratio / len(validation_centres)))
    blocks: list[list[int]] = []
    for centre in validation_centres:
        start = min(max(0, int(len(before) * float(centre)) - width // 2), len(before) - width)
        block = before[start:start + width]
        if blocks and block[0] <= blocks[-1][-1]:
            raise ValueError("validation blocks overlap; use fewer blocks or a smaller validation_ratio")
        blocks.append(block)

    frames = {int(k): {int(v) for v in values} for k, values in positives.items()}
    test_frames = set().union(*(frames[i] for i in test))
    candidates = [index for block in blocks for index in block]
    validation = [i for i in candidates if not frames[i] & test_frames]
    dropped_validation = [i for i in candidates if frames[i] & test_frames]
    held_frames = test_frames.union(*(frames[i] for i in validation)) if validation else test_frames
    in_validation = set(candidates)
    train, dropped_train = [], []
    for index in before:
        if index in in_validation:
            continue
        (dropped_train if frames[index] & held_frames else train).append(index)
    if not train or not validation:
        raise RuntimeError("the block split left no training or no validation query")
    return {
        "positives": {int(k): list(v) for k, v in positives.items()},
        "train_query_indices": train,
        "validation_query_indices": validation,
        "test_query_indices": test,
        "test_start": test_start,
        "validation_blocks": [[block[0], block[-1]] for block in blocks],
        "validation_centres": [float(c) for c in validation_centres],
        "excluded_train_indices": dropped_train,
        "excluded_validation_indices": dropped_validation,
        "excluded_test_indices": [],
        "split_guard": "ground_truth_blocks",
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
