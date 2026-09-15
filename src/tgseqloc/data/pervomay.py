"""Pervomayskaya street as a place-recognition task.

Two walks along the same street, a day apart, with a camera looking along the
direction of walking. Each walk comes with ``poses.csv``: a camera pose per
image, both walks expressed in one coordinate frame. Ground truth is therefore
metric, as on RobotCar, but taken in three dimensions: the common frame is not
levelled -- height changes by some 300 m along a flat street -- so a distance in
its first two coordinates is not a distance on the ground.

The dataset is used only to evaluate models trained elsewhere, so the split
has no training part: every query that has a correct answer is a test query.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Collection, Mapping

from tgseqloc.data.formats import FrameRecord


def parse_timestamp_ns(value: str) -> int:
    """Seconds with a fractional part, as written in ``poses.csv``, in nanoseconds.

    Parsed as text: a float holds only about 16 significant digits, and a Unix
    time in seconds with nine decimals needs 19.
    """

    text = str(value).strip()
    seconds, _, fraction = text.partition(".")
    if not seconds.lstrip("-").isdigit() or (fraction and not fraction.isdigit()):
        raise ValueError(f"not a timestamp in seconds: {value!r}")
    return int(seconds) * 1_000_000_000 + int((fraction + "000000000")[:9])


def load_poses(path: str | Path) -> dict[str, tuple[int, tuple[float, float, float]]]:
    """Timestamp and position per image stem, from one walk's ``poses.csv``."""

    poses: dict[str, tuple[int, tuple[float, float, float]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = Path(str(row["image"])).stem
            if stem in poses:
                raise ValueError(f"image {stem} appears twice in {path}")
            poses[stem] = (
                parse_timestamp_ns(row["timestamp"]),
                (float(row["tx"]), float(row["ty"]), float(row["tz"])),
            )
    if not poses:
        raise ValueError(f"no poses in {path}")
    return poses


def discover_pervomay_records(
    ocr_root_template: str,
    graph_root_template: str,
    poses_path_template: str,
    traversals: Mapping[str, str],
    *,
    image_path_template: str = "",
    ocr_file_name: str = "paddleocr_v5.json",
    require_inputs: bool = True,
    frame_list: Collection[str] | None = None,
) -> list[FrameRecord]:
    """Frames that carry recognized text, a scene graph and a pose, in walk order.

    Recognition and graphs cover every tenth frame of each walk. A frame without
    a pose is an error rather than a frame left out: the poses file lists every
    image of the walk, so a missing row means the inputs do not belong together.
    """

    records: list[FrameRecord] = []
    for sequence, traversal in traversals.items():
        def fill(template: str) -> Path:
            return Path(template.format(traversal=traversal, sequence=sequence))

        ocr_root, graph_root = fill(ocr_root_template), fill(graph_root_template)
        if not ocr_root.is_dir() or not graph_root.is_dir():
            if require_inputs:
                raise FileNotFoundError(
                    f"Missing Pervomay inputs for {sequence}: {ocr_root} / {graph_root}"
                )
            continue
        poses = load_poses(fill(poses_path_template))
        image_root = fill(image_path_template) if image_path_template else None
        graphs = {path.stem for path in graph_root.glob("*.json")}
        stems = sorted(
            stem for stem in (path.name for path in ocr_root.iterdir())
            if stem.isdigit() and stem in graphs
            and (frame_list is None or stem in frame_list)
        )
        unposed = [stem for stem in stems if stem not in poses]
        if unposed:
            raise ValueError(
                f"{len(unposed)} frames of {traversal} have no pose, e.g. {unposed[0]}"
            )
        for index, stem in enumerate(stems):
            records.append(
                FrameRecord(
                    sequence=sequence,
                    index=index,
                    timestamp=poses[stem][0],
                    stem=stem,
                    image_path=(
                        image_root / f"{stem}.jpg" if image_root else ocr_root / stem
                    ),
                    ocr_path=ocr_root / stem / ocr_file_name,
                    graph_path=graph_root / f"{stem}.json",
                )
            )
    return records


def pose_positions(
    records: list[FrameRecord], poses: Mapping[str, tuple[int, tuple[float, float, float]]]
) -> dict[int, tuple[float, float, float]]:
    """Position per record index."""

    return {record.index: poses[record.stem][1] for record in records}


def build_evaluation_split(
    query_positions: Mapping[int, Any], positives: Mapping[int, list[int]]
) -> dict[str, Any]:
    """Every query with a correct answer is a test query; nothing is trained on."""

    return {
        "positives": {int(k): [int(v) for v in values] for k, values in positives.items()},
        "train_query_indices": [],
        "validation_query_indices": [],
        "test_query_indices": sorted(index for index in query_positions if positives.get(index)),
        "protocol": "evaluation_only",
    }
