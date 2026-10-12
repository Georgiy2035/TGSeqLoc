"""Frozen text encoders for offline graph preparation."""

from __future__ import annotations

import zlib
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as functional

DEFAULT_TEXT_ENCODER = "intfloat/multilingual-e5-small"


@runtime_checkable
class TextEncoder(Protocol):
    embedding_dim: int

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class FrozenTextEncoder:
    """Frozen Hugging Face encoder, excluded from graph training.

    ``pooling`` follows the model rather than the pipeline: E5 is trained with
    mean pooling and BGE-M3 and LaBSE with the CLS token, and using one recipe
    for all of them would compare pooling choices instead of representations.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_ENCODER,
        revision: str | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        *,
        prefix: str = "passage: ",
        pooling: str = "mean",
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
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self.prefix = prefix
        self.pooling = pooling
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
            "pooling": self.pooling,
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
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            batches.append(functional.normalize(pooled, p=2, dim=1).cpu().float())
        return torch.cat(batches)


class CharNGramEncoder:
    """Hashed character n-grams, built for text a recognizer got partly wrong.

    A subword language model tokenizes ``PECLARD`` and ``PECLABD`` into
    unrelated pieces, so one wrong character can move the whole embedding. With
    character n-grams the two strings still share most of their features, and
    similarity falls roughly in proportion to how much of the string survived.
    At the 30-50% CER these datasets produce, that difference decides whether
    the text channel carries signal at all.

    Nothing is downloaded and nothing is trained: the mapping from n-gram to
    bucket is a checksum, which also makes it reproducible across machines.
    Being purely lexical, it has no notion of meaning -- two synonyms are as
    far apart as two unrelated words -- which is the trade for the robustness.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        min_n: int = 3,
        max_n: int = 5,
        *,
        lowercase: bool = True,
        device: str | torch.device = "cpu",
        batch_size: int = 128,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if not 1 <= min_n <= max_n:
            raise ValueError(f"require 1 <= min_n <= max_n, got {min_n} and {max_n}")
        self.embedding_dim = int(embedding_dim)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.lowercase = bool(lowercase)
        # Kept for interface parity with the Hugging Face encoder; hashing is
        # cheap enough that neither changes the result.
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": "char_ngram",
            "embedding_dim": self.embedding_dim,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "lowercase": self.lowercase,
            "hash": "crc32",
        }

    def _features(self, text: str) -> list[int]:
        """Bucket indices of every n-gram, with word boundaries marked.

        The ``<`` and ``>`` markers let a prefix or suffix be a feature of its
        own, which is what keeps a truncated reading close to the full word.
        """

        value = text.strip()
        if self.lowercase:
            value = value.lower()
        if not value:
            return []
        padded = f"<{value}>"
        buckets: list[int] = []
        for size in range(self.min_n, self.max_n + 1):
            if size > len(padded):
                break
            for start in range(len(padded) - size + 1):
                gram = padded[start:start + size]
                # crc32 rather than hash(): the built-in is salted per process,
                # which would silently change embeddings between runs.
                buckets.append(zlib.crc32(gram.encode("utf-8")) % self.embedding_dim)
        return buckets

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        result = torch.zeros((len(texts), self.embedding_dim), dtype=torch.float32)
        for row, text in enumerate(texts):
            buckets = self._features(str(text))
            if not buckets:
                continue
            index = torch.tensor(buckets, dtype=torch.long)
            result[row].scatter_add_(
                0, index, torch.ones(len(buckets), dtype=torch.float32)
            )
        # Strings too short to yield any n-gram stay at zero; normalizing them
        # would divide by nothing.
        return functional.normalize(result, p=2, dim=1)


def build_char_ngram_encoder(
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    **params: Any,
) -> CharNGramEncoder:
    """Registry factory for the ``char_ngram`` text encoder."""

    allowed = {"embedding_dim", "min_n", "max_n", "lowercase"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"unknown char_ngram params: {', '.join(unknown)}; "
            f"supported: {', '.join(sorted(allowed))}"
        )
    return CharNGramEncoder(device=device, batch_size=batch_size, **params)


def build_multilingual_e5(
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    **params: Any,
) -> FrozenTextEncoder:
    """Registry factory for the frozen Hugging Face encoder."""

    allowed = {"model_name", "revision", "prefix", "pooling"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"unknown text encoder params: {', '.join(unknown)}; "
            f"supported: {', '.join(sorted(allowed))}"
        )
    return FrozenTextEncoder(device=device, batch_size=batch_size, **params)


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
