"""Oxford RobotCar as a place-recognition task.

Two differences from V4RL shape this module. Ground truth here is metric: the
INS log gives a UTM position for every moment, so a query is answered
correctly when the retrieved frame was recorded near where the query was, and
"near" is a distance in metres rather than a row in a correspondence file. And
the route is revisited -- 23% of base frames come within 25 m of a frame more
than 200 positions away -- so ordering frames by time does not by itself
separate places, and the split has to check distance explicitly.
"""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from tgseqloc.data.formats import FrameRecord

#: Standard tolerance for RobotCar in the place-recognition literature.
DEFAULT_RADIUS_M = 25.0

#: Frames closer together than this along the trajectory are neighbours, not a
#: revisit; used only when reporting how often the route returns to a place.
_NEIGHBOUR_SPAN = 200


@dataclass(frozen=True, slots=True)
class Track:
    """Timestamped positions from one traversal's INS log."""

    timestamps: tuple[int, ...]
    positions: tuple[tuple[float, float], ...]

    def position_at(self, timestamp: int) -> tuple[float, float] | None:
        """Interpolate the position at ``timestamp``, or None if uncovered.

        A frame outside the INS interval has no position that can be trusted,
        and extrapolating one would invent ground truth.
        """

        if not self.timestamps:
            return None
        if timestamp < self.timestamps[0] or timestamp > self.timestamps[-1]:
            return None
        index = bisect.bisect_left(self.timestamps, timestamp)
        if index == 0:
            return self.positions[0]
        before, after = self.timestamps[index - 1], self.timestamps[index]
        first, second = self.positions[index - 1], self.positions[index]
        if after == before:
            return second
        weight = (timestamp - before) / (after - before)
        return (
            first[0] + weight * (second[0] - first[0]),
            first[1] + weight * (second[1] - first[1]),
        )


def load_ins_track(path: str | Path) -> Track:
    """Read northing/easting against time from a RobotCar INS log."""

    timestamps: list[int] = []
    positions: list[tuple[float, float]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = int(row["timestamp"])
                northing = float(row["northing"])
                easting = float(row["easting"])
            except (KeyError, TypeError, ValueError):
                continue
            timestamps.append(timestamp)
            positions.append((northing, easting))
    if not timestamps:
        raise ValueError(f"No usable INS rows in {path}")
    order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
    return Track(
        tuple(timestamps[i] for i in order), tuple(positions[i] for i in order)
    )


def load_frame_list(path: str | Path) -> frozenset[str]:
    """Frame stems to keep, one per line; ``role/stem`` is accepted as well.

    An allowlist rather than a stride: which frames an experiment uses has to
    be written down to be reproduced, and a stride over a directory depends on
    what the directory happens to hold -- which changed the moment recognition
    was extended from every eighth frame to all of them. An empty list is an
    error, not a request for zero frames.
    """

    stems = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        stems.add(value.rsplit("/", 1)[-1])
    if not stems:
        raise ValueError(f"frame list {path} names no frames")
    return frozenset(stems)


def discover_robotcar_records(
    ocr_root_template: str,
    graph_root_template: str,
    traversals: Mapping[str, str],
    *,
    image_path_template: str = "",
    ocr_file_name: str = "paddleocr_v5.json",
    require_inputs: bool = True,
    frame_list: Collection[str] | None = None,
) -> list[FrameRecord]:
    """Frames that carry both a scene graph and recognized text, in time order.

    The recognizer and the graph generator were run over different samplings of
    the same traversals, so the usable set is their intersection; taking either
    alone would leave frames whose graph or whose text is missing.
    """

    records: list[FrameRecord] = []
    for sequence, traversal in traversals.items():
        ocr_root = Path(ocr_root_template.format(traversal=traversal, sequence=sequence))
        graph_root = Path(graph_root_template.format(traversal=traversal, sequence=sequence))
        image_root = (
            Path(image_path_template.format(traversal=traversal, sequence=sequence))
            if image_path_template
            else None
        )
        if not ocr_root.is_dir() or not graph_root.is_dir():
            if require_inputs:
                raise FileNotFoundError(
                    f"Missing RobotCar inputs for {sequence}: {ocr_root} / {graph_root}"
                )
            continue
        graphs = {path.stem for path in graph_root.glob("*.json")}
        stems = sorted(
            stem for stem in (p.name for p in ocr_root.iterdir())
            if stem.isdigit() and stem in graphs
            and (frame_list is None or stem in frame_list)
        )
        for index, stem in enumerate(stems):
            records.append(
                FrameRecord(
                    sequence=sequence,
                    index=index,
                    timestamp=int(stem),
                    stem=stem,
                    image_path=(
                        image_root / f"{stem}.png" if image_root else ocr_root / stem
                    ),
                    ocr_path=ocr_root / stem / ocr_file_name,
                    graph_path=graph_root / f"{stem}.json",
                )
            )
    return records


def positions_for(records: Sequence[FrameRecord], track: Track) -> dict[int, tuple[float, float]]:
    """Interpolated position per record index, skipping frames the INS misses."""

    placed: dict[int, tuple[float, float]] = {}
    for record in records:
        position = track.position_at(record.timestamp)
        if position is not None:
            placed[record.index] = position
    return placed


def build_radius_positives(
    query_positions: Mapping[int, tuple[float, float]],
    database_positions: Mapping[int, tuple[float, float]],
    radius: float = DEFAULT_RADIUS_M,
) -> dict[int, list[int]]:
    """Database frames recorded within ``radius`` metres of each query.

    Metric ground truth replaces V4RL's correspondence file, which is what lets
    a query have several right answers instead of exactly one -- and what makes
    "no correct match exists" a statement about coverage rather than about a
    missing line in a file.
    """

    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    database = sorted(database_positions.items(), key=lambda item: item[1][0])
    northings = [position[0] for _, position in database]
    positives: dict[int, list[int]] = {}
    for query_index, (north, east) in query_positions.items():
        low = bisect.bisect_left(northings, north - radius)
        high = bisect.bisect_right(northings, north + radius)
        near = [
            index
            for index, position in database[low:high]
            if abs(position[1] - east) <= radius
            and math.dist(position, (north, east)) <= radius
        ]
        if near:
            positives[query_index] = sorted(near)
    return positives


def load_fold_split(
    assignment: Mapping[str, Any],
    fold: int,
    index_by_stem: Mapping[str, int],
    positives: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    """Train, validation and test taken from a written fold assignment.

    Blocks of route, which fold each belongs to and the guard bands between
    them are decided once, offline, from a single reference recognizer, and
    written to a file. Every arm then reads that same file, so the split cannot
    drift with the text an arm happens to find -- which is what makes a paired
    comparison between arms mean anything. A cut computed per arm would give
    the recognizer that reads more text a different test set.
    """

    folds = assignment.get("assignment") or {}
    part_stems = folds.get(str(int(fold)))
    if part_stems is None:
        raise ValueError(f"fold {fold} is not in the assignment ({sorted(folds)})")
    parts: dict[str, list[int]] = {}
    unknown = 0
    for part in ("train", "validation", "test"):
        indices = []
        for stem in part_stems.get(part, ()):
            index = index_by_stem.get(str(stem))
            if index is None:
                unknown += 1
                continue
            if positives.get(index):
                indices.append(index)
        parts[part] = sorted(indices)
    if not parts["train"] or not parts["test"]:
        raise RuntimeError(f"fold {fold} left an empty train or test set")
    return {
        "positives": {int(k): [int(v) for v in values] for k, values in positives.items()},
        "train_query_indices": parts["train"],
        "validation_query_indices": parts["validation"],
        "test_query_indices": parts["test"],
        "split_fold": int(fold),
        "split_folds": int(assignment.get("folds", 0)),
        "split_block_length_m": float(assignment.get("block_length_m", 0.0)),
        "split_radius_m": float(assignment.get("radius_m", 0.0)),
        "split_density_reference": str(assignment.get("density_reference", "")),
        "frames_not_in_this_run": unknown,
    }


def build_geographic_split(
    query_positions: Mapping[int, tuple[float, float]],
    positives: Mapping[int, Sequence[int]],
    *,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    radius: float = DEFAULT_RADIUS_M,
) -> dict[str, Any]:
    """Split queries into segments that do not share ground.

    Cutting the query sequence by time almost separates places, because the
    vehicle drives a route -- but only almost: it revisits, and a handful of
    late frames land back on early ground. Those are dropped rather than left
    in, so that no held-out query can be answered from a place the model was
    trained on.

    Validation is a segment of its own, held to the same rule as the test set.
    It used to be the tail of the training block, which left it up against
    training ground -- 9 metres away on the RobotCar route -- and on a stretch
    of the route whose text density was three times the test set's. The epoch
    is chosen by validation recall, so a validation set that overlaps training
    and misrepresents the very signal under ablation chooses the wrong epoch.
    """

    ordered = sorted(query_positions)
    if not ordered:
        raise ValueError("no positioned query frames")
    if not 0 < test_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("split ratios must satisfy 0 < test < 1 and 0 <= validation < 1")

    if test_ratio + validation_ratio >= 1:
        raise ValueError("test and validation ratios must leave room for training")

    total = len(ordered)
    test_cut = int(total * (1 - test_ratio))
    validation_cut = int(total * (1 - test_ratio - validation_ratio))
    train_block = ordered[:validation_cut]
    validation_block = ordered[validation_cut:test_cut]
    test_block = ordered[test_cut:]

    def separate(block, reference):
        """Keep only frames further than ``radius`` from every reference point."""

        kept, leaked = [], []
        for index in block:
            position = query_positions[index]
            if any(math.dist(position, point) <= radius for point in reference):
                leaked.append(index)
            else:
                kept.append(index)
        return kept, leaked

    train_points = [query_positions[index] for index in train_block]
    validation_kept, validation_leaked = separate(validation_block, train_points)
    held_out_points = train_points + [query_positions[index] for index in validation_kept]
    test_kept, test_leaked = separate(test_block, held_out_points)

    scored = lambda block: [index for index in block if positives.get(index)]
    return {
        "positives": {int(k): [int(v) for v in values] for k, values in positives.items()},
        "train_query_indices": scored(train_block),
        "validation_query_indices": scored(validation_kept),
        "test_query_indices": scored(test_kept),
        "test_start": ordered[test_cut] if test_cut < total else total,
        "validation_start": ordered[validation_cut] if validation_cut < total else total,
        "excluded_test_indices": test_leaked,
        "excluded_validation_indices": validation_leaked,
        "split_radius_m": float(radius),
    }


def revisit_fraction(
    positions: Sequence[tuple[float, float]], radius: float = DEFAULT_RADIUS_M
) -> float:
    """Share of frames that come back within ``radius`` of a distant frame.

    Reported rather than acted on: it is the reason the split checks distance
    instead of trusting time order.
    """

    total = len(positions)
    if total <= _NEIGHBOUR_SPAN:
        return 0.0
    revisits = 0
    for i in range(total):
        for j in range(i + _NEIGHBOUR_SPAN, total):
            if math.dist(positions[i], positions[j]) <= radius:
                revisits += 1
                break
    return revisits / total
