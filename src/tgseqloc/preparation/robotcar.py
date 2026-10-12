"""Preparation for Oxford RobotCar.

Reuses the frame-by-frame graph building of :mod:`tgseqloc.preparation.v4rl`
through its two seams and supplies only what RobotCar does differently:
frames are the intersection of what the recognizer and the graph generator
covered, and ground truth is a distance in metres taken from the INS log
rather than a correspondence file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tgseqloc.data.formats import FrameRecord
from tgseqloc.data.robotcar import (
    DEFAULT_RADIUS_M,
    build_geographic_split,
    load_fold_split,
    build_radius_positives,
    discover_camera_records,
    discover_robotcar_records,
    extend_fold_assignment,
    frame_camera,
    load_frame_list,
    load_ins_track,
    positions_for,
)
from tgseqloc.data.v4rl import DEFAULT_OCR_FILE_NAME, save_json
from tgseqloc.preparation.v4rl import _get, _normalize_config, process_v4rl


def _traversals(config: Any) -> dict[str, str]:
    """Which recording plays the database and which plays the queries."""

    traversals = _get(config, "traversals")
    if not traversals:
        raise ValueError("RobotCar preparation requires dataset.traversals")
    return {str(role): str(name) for role, name in dict(traversals).items()}


def _frame_list(config: Any) -> frozenset[str] | None:
    """The configured allowlist of frames, or None to take every frame found."""

    path = _get(config, "frame_list_path")
    return load_frame_list(path) if path else None


def _camera_sources(settings: Any) -> dict[str, dict[str, Any]] | None:
    """Per-camera inputs, or None for the single front camera.

    A camera inherits the top-level templates and frame list and overrides
    what differs for it; ``{camera}`` in a template is its name.
    """

    cameras = _get(settings, "cameras")
    if not cameras:
        return None
    defaults = {
        key: _get(settings, key)
        for key in ("ocr_root_template", "scene_graph_root_template", "image_path_template", "frame_list_path", "ocr_file_name")
    }
    return {
        str(name): {**defaults, **dict(source or {})}
        for name, source in dict(cameras).items()
    }


def discover_robotcar_inputs(config: Any) -> list[FrameRecord]:
    """Frames carrying both a scene graph and recognized text."""

    settings = _normalize_config(config)
    sources = _camera_sources(settings)
    if sources is not None:
        return discover_camera_records(
            sources,
            _traversals(settings),
            primary=str(_get(settings, "primary_camera", "stereo_centre")),
            ocr_file_name=str(_get(settings, "ocr_file_name", DEFAULT_OCR_FILE_NAME)),
        )
    return discover_robotcar_records(
        str(_get(settings, "ocr_root_template")),
        str(_get(settings, "scene_graph_root_template")),
        _traversals(settings),
        image_path_template=str(_get(settings, "image_path_template", "") or ""),
        ocr_file_name=str(_get(settings, "ocr_file_name", DEFAULT_OCR_FILE_NAME)),
        frame_list=_frame_list(settings),
    )


def write_robotcar_split(
    config: Any, records: list[FrameRecord], mappings_root: Path
) -> tuple[dict[str, Any], Any]:
    """Metric ground truth and a split whose parts do not share ground."""

    settings = _normalize_config(config)
    traversals = _traversals(settings)
    reference = str(_get(settings, "reference_sequence", "base"))
    query = str(_get(settings, "query_sequence", "query"))
    radius = float(_get(settings, "gt_radius_m", DEFAULT_RADIUS_M))
    ins_template = str(_get(settings, "ins_path_template"))

    tracks = {
        role: load_ins_track(ins_template.format(traversal=name, sequence=role))
        for role, name in traversals.items()
    }
    by_sequence = {
        role: [record for record in records if record.sequence == role]
        for role in traversals
    }
    positions = {
        role: positions_for(by_sequence[role], tracks[role]) for role in traversals
    }
    unplaced = {
        role: len(by_sequence[role]) - len(positions[role]) for role in traversals
    }

    positives = build_radius_positives(positions[query], positions[reference], radius)
    folds_path = str(_get(settings, "split_folds_path", "") or "")
    if folds_path:
        index_by_stem = {record.stem: record.index for record in by_sequence[query]}
        assignment = json.loads(Path(folds_path).read_text(encoding="utf-8"))
        if _camera_sources(settings) is not None:
            primary = str(_get(settings, "primary_camera", "stereo_centre"))
            others = [
                (record.stem, record.timestamp)
                for record in by_sequence[query]
                if frame_camera(record.stem, primary) != primary
            ]
            # RobotCar timestamps are microseconds.
            gap = int(float(_get(settings, "camera_fold_max_gap_s", 0.3)) * 1_000_000)
            assignment = extend_fold_assignment(assignment, others, gap)
        split = load_fold_split(
            assignment,
            int(_get(settings, "split_fold", -1)),
            index_by_stem,
            positives,
        )
    else:
        split = build_geographic_split(
            positions[query],
            positives,
            test_ratio=float(_get(settings, "test_ratio", 0.2)),
            validation_ratio=float(_get(settings, "validation_ratio", 0.1)),
            radius=radius,
        )
    split["database_paths"] = [
        f"{record.sequence}/{record.stem}.pt" for record in by_sequence[reference]
    ]
    split["query_paths"] = [
        f"{record.sequence}/{record.stem}.pt" for record in by_sequence[query]
    ]
    split["reference_sequence"] = reference
    split["query_sequence"] = query
    # Metric positions travel with the split, so that training can keep frames
    # that are almost positives out of the negatives without the INS logs.
    split["query_positions"] = {
        str(index): [float(p[0]), float(p[1])] for index, p in positions[query].items()
    }
    split["database_positions"] = {
        str(index): [float(p[0]), float(p[1])] for index, p in positions[reference].items()
    }
    split["frames_without_position"] = unplaced
    split["queries_without_positives"] = sum(
        1 for index in positions[query] if not positives.get(index)
    )
    split_name = (
        f"split_fold{int(_get(settings, 'split_fold', -1))}.json"
        if folds_path
        else "temporal_split.json"
    )
    save_json(split, mappings_root / split_name)

    digest = hashlib.sha256(
        json.dumps(
            {str(k): sorted(v) for k, v in positives.items()}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    descriptor = {
        "kind": "metric_radius",
        "radius_m": radius,
        "reference_sequence": reference,
        "query_sequence": query,
        "traversals": traversals,
        "positives_sha256": digest,
        "annotated_queries": len(positives),
    }
    save_json(descriptor, mappings_root / "gt_radius.json")
    return split, descriptor


def process_robotcar(config: Any, **kwargs: Any) -> dict[str, Any]:
    """Prepare RobotCar frames and return the written manifest."""

    return process_v4rl(
        config,
        record_discovery=discover_robotcar_inputs,
        split_writer=write_robotcar_split,
        **kwargs,
    )


def discover_robotcar_frame_count(config: Any) -> int:
    """Number of usable frames, for the doctor and the CLI."""

    return len(discover_robotcar_inputs(config))


def discover_robotcar_frames(config: Any) -> list[tuple[str, str, Path]]:
    """Frames as (sequence, stem, image path) for the model stages."""

    return [
        (record.sequence, record.stem, record.image_path)
        for record in discover_robotcar_inputs(config)
    ]


process_robotcar.discover_inputs = discover_robotcar_frame_count
process_robotcar.discover_frames = discover_robotcar_frames
