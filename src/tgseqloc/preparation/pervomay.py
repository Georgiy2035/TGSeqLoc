"""Preparation for Pervomayskaya street.

Reuses the frame-by-frame graph building of :mod:`tgseqloc.preparation.v4rl`
through its two seams: frames are those with recognized text, a scene graph and
a pose, and ground truth is a distance in metres between poses. The split is
for evaluation only -- the dataset tests models trained on other data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tgseqloc.data.formats import FrameRecord
from tgseqloc.data.pervomay import (
    build_evaluation_split,
    discover_pervomay_records,
    load_poses,
    pose_positions,
)
from tgseqloc.data.robotcar import DEFAULT_RADIUS_M, build_radius_positives, load_frame_list
from tgseqloc.data.v4rl import DEFAULT_OCR_FILE_NAME, save_json
from tgseqloc.data.identity import source_file_identity
from tgseqloc.preparation.robotcar import _traversals
from tgseqloc.preparation.v4rl import _get, _normalize_config, process_v4rl


def _poses_path(settings: Any, role: str, traversal: str) -> Path:
    template = str(_get(settings, "poses_path_template", "") or "")
    if not template:
        raise ValueError("Pervomay preparation requires dataset.poses_path_template")
    return Path(template.format(traversal=traversal, sequence=role))


def discover_pervomay_inputs(config: Any, *, require_inputs: bool = True) -> list[FrameRecord]:
    """Frames carrying recognized text, a scene graph and a pose."""

    settings = _normalize_config(config)
    frame_list_path = _get(settings, "frame_list_path")
    return discover_pervomay_records(
        str(_get(settings, "ocr_root_template")),
        str(_get(settings, "scene_graph_root_template")),
        str(_get(settings, "poses_path_template")),
        _traversals(settings),
        image_path_template=str(_get(settings, "image_path_template", "") or ""),
        ocr_file_name=str(_get(settings, "ocr_file_name", DEFAULT_OCR_FILE_NAME)),
        require_inputs=require_inputs,
        frame_list=load_frame_list(frame_list_path) if frame_list_path else None,
    )


def write_pervomay_split(
    config: Any, records: list[FrameRecord], mappings_root: Path
) -> tuple[dict[str, Any], Any]:
    """Metric ground truth between the two walks and an evaluation-only split."""

    settings = _normalize_config(config)
    traversals = _traversals(settings)
    reference = str(_get(settings, "reference_sequence", "base"))
    query = str(_get(settings, "query_sequence", "query"))
    radius = float(_get(settings, "gt_radius_m", DEFAULT_RADIUS_M))
    pose_files = {role: _poses_path(settings, role, name) for role, name in traversals.items()}
    positions = {
        role: pose_positions(
            [record for record in records if record.sequence == role],
            load_poses(pose_files[role]),
        )
        for role in traversals
    }

    positives = build_radius_positives(positions[query], positions[reference], radius)
    split = build_evaluation_split(positions[query], positives)
    by_sequence = {
        role: [record for record in records if record.sequence == role] for role in traversals
    }
    split["database_paths"] = [
        f"{record.sequence}/{record.stem}.pt" for record in by_sequence[reference]
    ]
    split["query_paths"] = [
        f"{record.sequence}/{record.stem}.pt" for record in by_sequence[query]
    ]
    split["reference_sequence"] = reference
    split["query_sequence"] = query
    split["query_positions"] = {
        str(index): [float(value) for value in point] for index, point in positions[query].items()
    }
    split["database_positions"] = {
        str(index): [float(value) for value in point]
        for index, point in positions[reference].items()
    }
    split["queries_without_positives"] = sum(
        1 for index in positions[query] if not positives.get(index)
    )
    save_json(split, mappings_root / "temporal_split.json")

    digest = hashlib.sha256(
        json.dumps({str(k): sorted(v) for k, v in positives.items()}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    descriptor = {
        "kind": "metric_radius",
        "radius_m": radius,
        "dimensions": 3,
        "reference_sequence": reference,
        "query_sequence": query,
        "traversals": traversals,
        # Both walks' poses define the answer, not only the one named gt_path.
        "poses": {role: source_file_identity(path) for role, path in pose_files.items()},
        "positives_sha256": digest,
        "annotated_queries": len(positives),
    }
    save_json(descriptor, mappings_root / "gt_radius.json")
    return split, descriptor


def process_pervomay(config: Any, **kwargs: Any) -> dict[str, Any]:
    """Prepare Pervomayskaya frames and return the written manifest."""

    return process_v4rl(
        config,
        record_discovery=discover_pervomay_inputs,
        split_writer=write_pervomay_split,
        **kwargs,
    )


def discover_pervomay_frame_count(config: Any) -> int:
    """Number of usable frames, for the doctor and the CLI."""

    return len(discover_pervomay_inputs(config))


def discover_pervomay_frames(config: Any) -> list[tuple[str, str, Path]]:
    """Frames as (sequence, stem, image path) for the model stages."""

    return [
        (record.sequence, record.stem, record.image_path)
        for record in discover_pervomay_inputs(config, require_inputs=False)
    ]


process_pervomay.discover_inputs = discover_pervomay_frame_count
process_pervomay.discover_frames = discover_pervomay_frames
