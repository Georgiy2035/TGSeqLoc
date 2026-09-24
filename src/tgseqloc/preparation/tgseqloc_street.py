"""Preparation for TGSeqLoc-Street.

Reuses the frame-by-frame graph building of :mod:`tgseqloc.preparation.v4rl`
through its two seams: frames are those with text and scene-graph annotations
and a pose, and ground truth is a distance in metres between poses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tgseqloc.data.formats import FrameRecord
from tgseqloc.data.tgseqloc_street import (
    build_cross_validation_split,
    build_evaluation_split,
    discover_street_records,
    load_poses,
    pose_positions,
)
from tgseqloc.data.robotcar import DEFAULT_RADIUS_M, build_radius_positives, load_frame_list
from tgseqloc.data.v4rl import save_json
from tgseqloc.data.identity import source_file_identity
from tgseqloc.preparation.robotcar import _traversals
from tgseqloc.preparation.v4rl import _get, _normalize_config, process_v4rl


def _poses_path(settings: Any, role: str, traversal: str) -> Path:
    template = str(_get(settings, "poses_path_template", "") or "")
    if not template:
        raise ValueError("TGSeqLoc-Street preparation requires dataset.poses_path_template")
    return Path(template.format(traversal=traversal, sequence=role))


def discover_street_inputs(config: Any, *, require_inputs: bool = True) -> list[FrameRecord]:
    """Frames that have text and scene-graph annotations and a pose."""

    settings = _normalize_config(config)
    frame_list_path = _get(settings, "frame_list_path")
    return discover_street_records(
        str(_get(settings, "text_root_template")),
        str(_get(settings, "scene_graph_root_template")),
        str(_get(settings, "poses_path_template")),
        _traversals(settings),
        require_inputs=require_inputs,
        frame_list=load_frame_list(frame_list_path) if frame_list_path else None,
    )


def write_street_split(
    config: Any, records: list[FrameRecord], mappings_root: Path
) -> tuple[dict[str, Any], Any]:
    """Metric ground truth between the two traversals and the split.

    The split is evaluation-only (every query with a correct answer is a test
    query) unless ``split_fold_count`` asks for K-fold cross-validation.
    """

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

    dimensions = int(_get(settings, "gt_dimensions", 0) or 0)
    if dimensions:
        positions = {
            role: {index: tuple(point[:dimensions]) for index, point in mapping.items()}
            for role, mapping in positions.items()
        }
    positives = build_radius_positives(positions[query], positions[reference], radius)
    fold_count = int(_get(settings, "split_fold_count", 0) or 0)
    if fold_count:
        split = build_cross_validation_split(
            positions[query], positives,
            folds=fold_count, test_fold=int(_get(settings, "split_fold", 0) or 0),
        )
    else:
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
    # Training reads the dimensionality from the split, so that mining uses
    # the same metric as the ground truth.
    split["gt_dimensions"] = dimensions
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
    if fold_count:
        # Graphs do not depend on the fold, so one preparation writes all K splits.
        for index in range(fold_count):
            other = build_cross_validation_split(
                positions[query], positives, folds=fold_count, test_fold=index
            )
            for key in ("database_paths", "query_paths", "reference_sequence",
                        "query_sequence", "query_positions", "database_positions",
                        "queries_without_positives", "gt_dimensions"):
                if key in split:
                    other[key] = split[key]
            save_json(other, mappings_root / f"split_fold{index}.json")
    save_json(split, mappings_root / "temporal_split.json")

    digest = hashlib.sha256(
        json.dumps({str(k): sorted(v) for k, v in positives.items()}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    descriptor = {
        "kind": "metric_radius",
        "radius_m": radius,
        "dimensions": dimensions or 3,
        "reference_sequence": reference,
        "query_sequence": query,
        "traversals": traversals,
        # Both traversals' poses define the answer.
        "poses": {role: source_file_identity(path) for role, path in pose_files.items()},
        "positives_sha256": digest,
        "annotated_queries": len(positives),
    }
    save_json(descriptor, mappings_root / "gt_radius.json")
    return split, descriptor


def process_tgseqloc_street(config: Any, **kwargs: Any) -> dict[str, Any]:
    """Prepare TGSeqLoc-Street frames and return the written manifest."""

    return process_v4rl(
        config,
        record_discovery=discover_street_inputs,
        split_writer=write_street_split,
        **kwargs,
    )


def discover_street_frame_count(config: Any) -> int:
    """Number of usable frames."""

    return len(discover_street_inputs(config))


process_tgseqloc_street.discover_inputs = discover_street_frame_count
