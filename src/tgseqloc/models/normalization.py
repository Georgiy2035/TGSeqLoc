"""Streaming normalization for continuous graph edge attributes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor


class EdgeAttrNormalizer:
    """Fit and apply per-column edge-attribute standardization.

    Statistics are accumulated on CPU in float64 using the parallel variance
    algorithm.  Empty tensors are intentionally ignored, which makes fitting
    datasets containing graphs without edges safe.
    """

    def __init__(
        self,
        log_indices: Iterable[int] | None = None,
        eps: float = 1e-6,
        feature_dim: int | None = None,
    ) -> None:
        self.log_indices = tuple(int(i) for i in (log_indices or ()))
        self.eps = float(eps)
        self.feature_dim = feature_dim
        self.count = 0
        self.mean: Tensor | None = None
        self.M2: Tensor | None = None
        self.std: Tensor | None = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def _validate(self, x: Tensor) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.ndim != 2:
            raise ValueError(f"edge attributes must be 2-D, got {tuple(x.shape)}")
        if self.feature_dim is None:
            self.feature_dim = int(x.shape[1])
        elif x.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} edge features, got {x.shape[1]}"
            )
        if self.log_indices and max(self.log_indices) >= x.shape[1]:
            raise ValueError("log_indices contains an out-of-range column")
        return x

    def _preprocess(self, x: Tensor) -> Tensor:
        x = self._validate(x)
        x = x.clone() if x.is_floating_point() else x.float()
        if self.log_indices:
            columns = list(self.log_indices)
            if torch.any(x[:, columns] <= -1):
                raise ValueError("log1p edge attributes must be greater than -1")
            x[:, columns] = torch.log1p(x[:, columns])
        return x

    def update(self, x: Tensor | None) -> "EdgeAttrNormalizer":
        if x is None:
            return self
        x = self._validate(x)
        if x.numel() == 0:
            return self
        values = self._preprocess(x).detach().to(device="cpu", dtype=torch.float64)
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        if self.mean is None:
            self.mean = torch.zeros(values.shape[1], dtype=torch.float64)
            self.M2 = torch.zeros(values.shape[1], dtype=torch.float64)
        assert self.M2 is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        self.M2 = (
            self.M2
            + batch_m2
            + delta.square() * self.count * batch_count / total
        )
        self.count = total
        self.std = None
        return self

    def finalize(self) -> "EdgeAttrNormalizer":
        if self.mean is None:
            if self.feature_dim is None:
                raise RuntimeError(
                    "cannot finalize without observations or a known feature_dim"
                )
            self.mean = torch.zeros(self.feature_dim, dtype=torch.float32)
            self.std = torch.ones(self.feature_dim, dtype=torch.float32)
            self.M2 = torch.zeros(self.feature_dim, dtype=torch.float64)
            return self
        if self.count < 2:
            self.std = torch.ones_like(self.mean, dtype=torch.float32)
        else:
            assert self.M2 is not None
            self.std = torch.sqrt((self.M2 / (self.count - 1)).clamp_min(0)).float()
        self.std = torch.where(self.std.isfinite(), self.std, torch.ones_like(self.std))
        self.mean = self.mean.float()
        return self

    def transform(self, x: Tensor | None) -> Tensor | None:
        if x is None:
            return None
        x = self._validate(x)
        if x.numel() == 0:
            return x
        if not self.fitted:
            raise RuntimeError("normalizer must be finalized before transform")
        values = self._preprocess(x)
        assert self.mean is not None and self.std is not None
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return (values - mean) / (std + self.eps)

    def __call__(self, x: Tensor | None) -> Tensor | None:
        return self.transform(x)

    def state_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "M2": self.M2,
            "std": self.std,
            "log_indices": list(self.log_indices),
            "eps": self.eps,
            "feature_dim": self.feature_dim,
        }

    def load_state_dict(self, state: Mapping) -> "EdgeAttrNormalizer":
        self.count = int(state.get("count", 0))
        self.mean = state.get("mean")
        self.M2 = state.get("M2")
        self.std = state.get("std")
        self.log_indices = tuple(int(i) for i in state.get("log_indices") or ())
        self.eps = float(state.get("eps", self.eps))
        self.feature_dim = state.get("feature_dim")
        if self.feature_dim is None and self.mean is not None:
            self.feature_dim = int(self.mean.numel())
        return self

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "EdgeAttrNormalizer":
        return cls().load_state_dict(state)
