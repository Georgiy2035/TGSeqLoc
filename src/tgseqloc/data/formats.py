"""Canonical per-frame formats exchanged between pipeline stages.

Every OCR backend returns :class:`FrameText` and every segmentation backend
returns :class:`FrameMasks`, whatever model produced them. Stages downstream of
those two never see a model-specific structure, so a backend can be swapped in
configuration without touching its consumers.

Detections keep their confidence and, once the dynamics stage has run, the
fraction of their area covered by dynamic masks. The previous representation
was a bare ``(boxes, texts)`` pair, which discarded confidence and left the
dynamics attribute nowhere to live.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Paths and time identity for one source frame.

    Lives here rather than beside the graph schema so that discovering frames
    does not require torch: a stage that only runs a model must be usable in an
    environment built for that model alone.
    """

    sequence: str
    index: int
    timestamp: int
    stem: str
    image_path: Path
    ocr_path: Path
    graph_path: Path


def _validate_box(box: Sequence[float]) -> Box:
    """Return a normalized, non-degenerate ``xyxy`` box."""

    if len(box) != 4:
        raise ValueError(f"box must have four values, got {len(box)}")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"box must be finite, got {box}")
    if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        raise ValueError(f"box must be normalized to [0, 1], got {box}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box must be non-degenerate, got {box}")
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class TextDetection:
    """One recognized string with its normalized location on the frame."""

    box: Box
    text: str
    confidence: float
    dynamic_score: float | None = None
    """Fraction of the box area covered by dynamic masks, or ``None`` until the
    dynamics stage has run. Kept continuous so the threshold stays a
    configuration choice rather than being baked into the artifact."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "box", _validate_box(self.box))
        object.__setattr__(self, "text", str(self.text))
        confidence = float(self.confidence)
        if not math.isfinite(confidence):
            raise ValueError(f"confidence must be finite, got {self.confidence}")
        object.__setattr__(self, "confidence", confidence)
        if self.dynamic_score is not None:
            score = float(self.dynamic_score)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"dynamic_score must be in [0, 1], got {score}")
            object.__setattr__(self, "dynamic_score", score)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)


@dataclass(frozen=True, slots=True)
class FrameText:
    """All recognized strings of one frame, plus the identity of their source."""

    detections: tuple[TextDetection, ...] = ()
    image_size: tuple[int, int] = (0, 0)
    """``(width, height)`` in pixels; needed to rasterize normalized boxes
    against segmentation masks."""
    model_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detections", tuple(self.detections))
        width, height = (int(value) for value in self.image_size)
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        object.__setattr__(self, "image_size", (width, height))

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def boxes(self) -> list[list[float]]:
        """Normalized ``xyxy`` boxes in detection order."""

        return [list(detection.box) for detection in self.detections]

    @property
    def texts(self) -> list[str]:
        """Recognized strings in detection order, aligned with :attr:`boxes`."""

        return [detection.text for detection in self.detections]

    def select(self, keep: Iterable[bool]) -> FrameText:
        """Return a copy holding only the detections whose flag is true."""

        flags = tuple(bool(value) for value in keep)
        if len(flags) != len(self.detections):
            raise ValueError(
                f"keep has {len(flags)} flags for {len(self.detections)} detections"
            )
        return replace(
            self,
            detections=tuple(
                detection
                for detection, flag in zip(self.detections, flags, strict=True)
                if flag
            ),
        )

    def with_dynamic_scores(self, scores: Sequence[float]) -> FrameText:
        """Return a copy whose detections carry the given coverage scores."""

        if len(scores) != len(self.detections):
            raise ValueError(
                f"got {len(scores)} scores for {len(self.detections)} detections"
            )
        return replace(
            self,
            detections=tuple(
                replace(detection, dynamic_score=float(score))
                for detection, score in zip(self.detections, scores, strict=True)
            ),
        )


@dataclass(frozen=True, slots=True)
class FrameMasks:
    """Dynamic-object coverage of one frame.

    The mask is the union of every instance a segmentation backend considers
    ephemeral, stored as COCO run-length encoding so a frame stays small on
    disk. Encoding and decoding belong to the segmentation backend; this type
    only carries the payload.
    """

    dynamic_rle: Mapping[str, Any] | None = None
    """``None`` means the backend ran and found no dynamic objects, which is
    different from the stage not having run at all."""
    image_size: tuple[int, int] = (0, 0)
    model_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width, height = (int(value) for value in self.image_size)
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        object.__setattr__(self, "image_size", (width, height))

    @property
    def is_empty(self) -> bool:
        return self.dynamic_rle is None
