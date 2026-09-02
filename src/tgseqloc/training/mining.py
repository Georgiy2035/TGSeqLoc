"""Hard-negative mining for descriptor learning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import faiss
import numpy as np
import torch


def _float32_matrix(values, name: str) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    return np.ascontiguousarray(result)


def mine_hard_negatives(
    database_embeddings,
    query_embeddings,
    query_indices: Sequence[int],
    positives: Mapping[int, Sequence[int]] | Mapping[str, Sequence[int]],
    negatives_per_query: int,
    search_depth: int = 256,
    seed: int = 0,
) -> dict[int, list[int]]:
    """Mine nearest non-positive database items using cosine similarity.

    If fewer unique candidates exist than requested, valid candidates are
    repeated deterministically.  If every database item is positive, the
    corresponding result is an empty list so callers can skip that query.
    Inputs are copied and never normalized in-place.
    """

    if negatives_per_query < 0:
        raise ValueError("negatives_per_query cannot be negative")
    database = _float32_matrix(database_embeddings, "database_embeddings").copy()
    queries = _float32_matrix(query_embeddings, "query_embeddings").copy()
    query_indices = [int(index) for index in query_indices]
    if database.shape[1] != queries.shape[1]:
        raise ValueError("database and query embedding dimensions differ")
    if len(query_indices) != len(queries):
        raise ValueError("query_indices must align with query_embeddings")
    if negatives_per_query == 0:
        return {index: [] for index in query_indices}
    if len(database) == 0:
        return {index: [] for index in query_indices}

    faiss.normalize_L2(database)
    faiss.normalize_L2(queries)
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    depth = min(max(int(search_depth), negatives_per_query, 1), len(database))
    _, nearest = index.search(queries, depth)
    rng = np.random.default_rng(seed)
    mined: dict[int, list[int]] = {}
    all_indices = np.arange(len(database), dtype=np.int64)
    for row, query_index in enumerate(query_indices):
        positive_values = positives.get(
            query_index, positives.get(str(query_index), ())  # type: ignore[arg-type]
        )
        positive_set = {int(value) for value in positive_values}
        selected = [
            int(candidate)
            for candidate in nearest[row]
            if candidate >= 0 and int(candidate) not in positive_set
        ]
        selected = list(dict.fromkeys(selected))[:negatives_per_query]
        if len(selected) < negatives_per_query:
            remaining = [
                int(candidate)
                for candidate in all_indices
                if int(candidate) not in positive_set
                and int(candidate) not in selected
            ]
            rng.shuffle(remaining)
            selected.extend(remaining[: negatives_per_query - len(selected)])
        if selected and len(selected) < negatives_per_query:
            base = selected.copy()
            selected.extend(
                base[offset % len(base)]
                for offset in range(negatives_per_query - len(selected))
            )
        mined[query_index] = selected
    return mined
