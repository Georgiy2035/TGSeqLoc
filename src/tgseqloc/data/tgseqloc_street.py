"""TGSeqLoc-Street: two traversals of one street recorded by a wheeled robot.

The traversals were recorded on consecutive days with a forward-facing camera.
Each comes with ``poses.csv``, one camera pose per image, both traversals in one
coordinate frame, so ground truth is a metric radius around the query pose.

The dataset is used to evaluate models trained on other data: in the
evaluation split every query that has a correct answer is a test query.
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
    """Timestamp and position per image stem, from one traversal's ``poses.csv``."""

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


def discover_street_records(
    text_root_template: str,
    graph_root_template: str,
    poses_path_template: str,
    traversals: Mapping[str, str],
    *,
    require_inputs: bool = True,
    frame_list: Collection[str] | None = None,
) -> list[FrameRecord]:
    """Frames that have text and scene-graph annotations and a pose, in order.

    Annotations cover every tenth frame of each traversal. A frame without a
    pose is an error rather than a frame left out: ``poses.csv`` lists every
    image of the traversal.
    """

    records: list[FrameRecord] = []
    for sequence, traversal in traversals.items():
        def fill(template: str) -> Path:
            return Path(template.format(traversal=traversal, sequence=sequence))

        text_root, graph_root = fill(text_root_template), fill(graph_root_template)
        if not text_root.is_dir() or not graph_root.is_dir():
            if require_inputs:
                raise FileNotFoundError(
                    f"Missing TGSeqLoc-Street annotations for {sequence}: {text_root} / {graph_root}"
                )
            continue
        poses = load_poses(fill(poses_path_template))
        graphs = {path.stem for path in graph_root.glob("*.json")}
        stems = sorted(
            stem for stem in (path.stem for path in text_root.glob("*.json"))
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
                    image_path=None,
                    text_path=text_root / f"{stem}.json",
                    graph_path=graph_root / f"{stem}.json",
                )
            )
    return records


def pose_positions(
    records: list[FrameRecord], poses: Mapping[str, tuple[int, tuple[float, float, float]]]
) -> dict[int, tuple[float, float, float]]:
    """Position per record index."""

    return {record.index: poses[record.stem][1] for record in records}


def build_cross_validation_split(
    query_positions: Mapping[int, Any],
    positives: Mapping[int, list[int]],
    *,
    folds: int,
    test_fold: int,
) -> dict[str, Any]:
    """One fold of a K-fold split over the query traversal, in order.

    Fold ``test_fold`` is the test set, the next one round is validation and
    the rest are training. Bounds are cut over all queries, not only the
    annotated ones, so the folds stay the same stretches of street whatever
    the ground-truth radius; queries with no positive then drop out of every
    part, because a query with no right answer trains nothing and scores
    nothing.
    """

    if folds < 3:
        raise ValueError(f"need at least three folds, got {folds}")
    if not 0 <= test_fold < folds:
        raise ValueError(f"test_fold must be in [0, {folds}), got {test_fold}")

    ordered = sorted(int(index) for index in query_positions)
    total = len(ordered)
    edges = [round(total * i / folds) for i in range(folds + 1)]
    members = [ordered[edges[i]:edges[i + 1]] for i in range(folds)]
    annotated = {int(k) for k, v in positives.items() if v}

    validation_fold = (test_fold + 1) % folds
    train_folds = [i for i in range(folds) if i not in (test_fold, validation_fold)]

    def keep(fold_ids: list[int]) -> list[int]:
        return sorted(i for fold in fold_ids for i in members[fold] if i in annotated)

    train = keep(train_folds)
    validation = keep([validation_fold])
    test = keep([test_fold])
    if not train or not test:
        raise RuntimeError(
            f"fold {test_fold} of {folds} left an empty annotated train or test set"
        )
    return {
        "positives": {int(k): [int(v) for v in values] for k, values in positives.items()},
        "train_query_indices": train,
        "validation_query_indices": validation,
        "test_query_indices": test,
        "protocol": "cross_validation",
        "split_fold": int(test_fold),
        "split_fold_count": int(folds),
        "fold_bounds": [[edges[i], edges[i + 1]] for i in range(folds)],
        "fold_assignment": {
            "train": train_folds,
            "validation": [validation_fold],
            "test": [test_fold],
        },
    }


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
