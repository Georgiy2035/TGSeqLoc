"""FAISS cosine retrieval and recall metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import faiss
import numpy as np
import torch


def _matrix(values, name: str) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    return np.ascontiguousarray(result)


class Retriever:
    """Exact cosine-similarity retrieval backed by ``faiss.IndexFlatIP``."""

    def __init__(self, embeddings=None, dimension: int | None = None) -> None:
        self.dimension = int(dimension) if dimension is not None else None
        self.index: faiss.Index | None = None
        self.ids: np.ndarray | None = None
        if embeddings is not None:
            self.fit(embeddings)

    def fit(self, embeddings, ids: Sequence | None = None) -> "Retriever":
        database = _matrix(embeddings, "embeddings").copy()
        if self.dimension is not None and database.shape[1] != self.dimension:
            raise ValueError(
                f"expected descriptor dimension {self.dimension}, got {database.shape[1]}"
            )
        self.dimension = int(database.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        if ids is not None and len(ids) != len(database):
            raise ValueError("ids must align with embeddings")
        self.ids = None if ids is None else np.asarray(ids)
        if len(database):
            faiss.normalize_L2(database)
            self.index.add(database)
        return self

    add = fit

    def search(self, queries, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.dimension is None:
            raise RuntimeError("fit must be called before search")
        if k < 1:
            raise ValueError("k must be positive")
        query = _matrix(queries, "queries").copy()
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"expected descriptor dimension {self.dimension}, got {query.shape[1]}"
            )
        if not len(query):
            return (
                np.empty((0, k), dtype=np.float32),
                np.empty((0, k), dtype=np.int64),
            )
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, int(k))
        if self.ids is not None:
            mapped = np.empty(indices.shape, dtype=object)
            valid = indices >= 0
            mapped[~valid] = None
            mapped[valid] = self.ids[indices[valid]]
            indices = mapped
        return scores, indices

    def retrieve(self, queries, k: int) -> np.ndarray:
        return self.search(queries, k)[1]


class RecallAtK:
    """Compute query-level retrieval recall for one or more cutoffs."""

    def __init__(
        self, k_values: int | Sequence[int] = (1, 5, 10), percentage: bool = True
    ) -> None:
        if isinstance(k_values, int):
            k_values = (k_values,)
        self.k_values = tuple(sorted({int(k) for k in k_values}))
        if not self.k_values or self.k_values[0] < 1:
            raise ValueError("all recall cutoffs must be positive")
        self.percentage = bool(percentage)

    def compute(
        self,
        nearest,
        positives: Mapping[int, Sequence[int]] | Mapping[str, Sequence[int]],
        query_indices: Sequence[int] | None = None,
    ) -> dict[str, float]:
        nearest = np.asarray(nearest)
        if nearest.ndim != 2:
            raise ValueError("nearest must be a 2-D index matrix")
        if query_indices is None:
            query_indices = list(range(len(nearest)))
        query_indices = [int(index) for index in query_indices]
        if len(query_indices) != len(nearest):
            raise ValueError("query_indices must align with nearest")
        scale = 100.0 if self.percentage else 1.0
        if not query_indices:
            return {f"R@{k}": 0.0 for k in self.k_values}
        recalls = {}
        for k in self.k_values:
            hits = 0
            for row, query_index in enumerate(query_indices):
                values = positives.get(
                    query_index, positives.get(str(query_index), ())  # type: ignore[arg-type]
                )
                positive_set = {int(value) for value in values}
                candidates = nearest[row, : min(k, nearest.shape[1])]
                hits += any(
                    int(candidate) >= 0 and int(candidate) in positive_set
                    for candidate in candidates
                )
            recalls[f"R@{k}"] = scale * hits / len(query_indices)
        return recalls

    __call__ = compute


def evaluate_retrieval(
    database_embeddings,
    query_embeddings,
    positives,
    query_indices: Sequence[int] | None = None,
    recall_values: Sequence[int] = (1, 5, 10),
    *,
    search_k: int | None = None,
    retriever_factory=Retriever,
    metric_factory=RecallAtK,
) -> tuple[dict[str, float], np.ndarray]:
    """Retrieve descriptors and return recalls plus requested-depth neighbors."""

    metric = metric_factory(recall_values)
    retriever = retriever_factory(database_embeddings)
    depth = max(metric.k_values)
    if search_k is not None:
        if int(search_k) < 1:
            raise ValueError("search_k must be positive")
        depth = max(depth, int(search_k))
    _, nearest = retriever.search(query_embeddings, depth)
    return metric(nearest, positives, query_indices), nearest
