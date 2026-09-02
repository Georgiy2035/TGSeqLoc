"""Frozen text encoders for offline graph preparation."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as functional

DEFAULT_TEXT_ENCODER = "intfloat/multilingual-e5-small"


@runtime_checkable
class TextEncoder(Protocol):
    embedding_dim: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class FrozenTextEncoder:
    """Mean-pooled Hugging Face encoder excluded from graph training."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_ENCODER,
        revision: str | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        *,
        prefix: str = "passage: ",
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FrozenTextEncoder requires the optional `transformers` package"
            ) from exc
        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.prefix = prefix
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.embedding_dim = int(self.model.config.hidden_size)

    @property
    def cache_identity(self) -> dict[str, str | int | None]:
        """Return the resolved model identity used by preparation caching."""

        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "resolved_commit": getattr(self.model.config, "_commit_hash", None),
            "embedding_dim": self.embedding_dim,
            "prefix": self.prefix,
        }

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        prefixed = [f"{self.prefix}{text}" for text in texts]
        batches = []
        for start in range(0, len(prefixed), self.batch_size):
            tokens = self.tokenizer(
                prefixed[start:start + self.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            batches.append(functional.normalize(pooled, p=2, dim=1).cpu().float())
        return torch.cat(batches)


def validate_embeddings(
    encoder: TextEncoder, texts: Sequence[str], embeddings: torch.Tensor
) -> torch.Tensor:
    """Validate an injected encoder's output and normalize its storage type."""

    value = torch.as_tensor(embeddings, dtype=torch.float32, device="cpu")
    expected = (len(texts), int(encoder.embedding_dim))
    if value.shape != expected:
        raise ValueError(f"Text encoder returned {tuple(value.shape)}, expected {expected}")
    if not torch.isfinite(value).all():
        raise ValueError("Text encoder returned non-finite embeddings")
    return value
