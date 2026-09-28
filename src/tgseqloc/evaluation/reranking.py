"""Two-stage retrieval and sequence aggregation (Sec. III-E).

The compact text-graph descriptor selects the ``K`` nearest database frames of
a query (Eq. 5); only those candidates are compared with the query by an image
descriptor, EDTformer in the paper, and ordered by cosine similarity (Eq. 6).
The image descriptor is never searched against the whole database.

Sequence pool fusion (Eq. 7) carries candidates along the query sequence: the
pool of a frame is the union of its own candidates with the pool of the
preceding frame, ordered by the current frame's image descriptor. Here the
pool spans the last ``frames`` query frames, so ``frames = 1`` is plain
reranking. Queries are processed in sequence order and only earlier frames are
used, so the fusion is causal.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _normalized(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("descriptors must be a 2-D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def rerank_sequence(
    candidates,
    query_descriptors,
    database_descriptors,
    *,
    frames: int = 1,
    depth: int | None = None,
) -> np.ndarray:
    """Candidates of consecutive queries reordered by image-descriptor similarity.

    ``candidates`` is ``[Q, K]`` database indices from the first stage, one row
    per query in sequence order; ``query_descriptors`` is ``[Q, D]`` aligned
    with those rows and ``database_descriptors`` is ``[N, D]``. Returns the
    first ``depth`` (default ``K``) entries of every query's ordered pool.
    """

    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.ndim != 2:
        raise ValueError("candidates must be a [queries, K] matrix")
    queries = _normalized(query_descriptors)
    database = _normalized(database_descriptors)
    if len(queries) != len(candidates):
        raise ValueError("query descriptors must align with the candidate rows")
    if frames < 1:
        raise ValueError("frames must be at least 1")
    depth = candidates.shape[1] if depth is None else int(depth)
    ordered = np.full((len(candidates), depth), -1, dtype=np.int64)
    for row in range(len(candidates)):
        window = candidates[max(0, row - frames + 1): row + 1]
        # Union of the window's candidates, each database frame once.
        pool = np.unique(window[window >= 0])
        if not len(pool):
            continue
        scores = database[pool] @ queries[row]
        # Stable order: equal scores keep the lower database index first.
        best = pool[np.argsort(-scores, kind="stable")][:depth]
        ordered[row, : len(best)] = best
    return ordered


def align_descriptors(
    paths: Sequence[str],
    descriptors: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Rows of per-sequence image descriptors in the order of prepared ``paths``.

    A prepared path is ``{sequence}/{frame_id}.pt``; ``descriptors`` maps a
    sequence to ``{frame_id: vector}``.
    """

    rows = []
    for path in paths:
        sequence, name = str(path).replace("\\", "/").rsplit("/", 1)
        frame_id = name.rsplit(".", 1)[0]
        try:
            rows.append(descriptors[sequence.rsplit("/", 1)[-1]][frame_id])
        except KeyError as error:
            raise KeyError(f"no image descriptor for frame {sequence}/{frame_id}") from error
    return np.stack(rows).astype(np.float32)


def load_image_descriptors(folder, sequences: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    """Read ``{folder}/{sequence}.npz`` files written by extract_image_descriptors.py."""

    from pathlib import Path

    out = {}
    for sequence in sequences:
        with np.load(Path(folder) / f"{sequence}.npz") as data:
            out[sequence] = dict(zip((str(i) for i in data["frame_ids"]), data["descriptors"]))
    return out
