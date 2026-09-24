"""Canonical per-frame formats shared by annotation and preparation.

Every text annotation is a :class:`FrameText` and every dynamic-object
segmentation is a :class:`FrameMasks`, whatever model produced them, so a
recognizer or segmenter can be replaced without touching their consumers.

Detections keep their confidence and the fraction of their area covered by
dynamic-object masks (``dynamic_score``). The score is stored rather than a
keep/drop decision, so the threshold remains a preparation setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """Annotation paths and time identity for one frame.

    Defined apart from the graph schema so that discovering frames does not
    import torch. ``image_path`` is optional: preparation and evaluation read
    annotations only, and the images are needed only to produce them.
    """

    sequence: str
    index: int
    timestamp: int
    stem: str
    image_path: Path | None
    text_path: Path
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
    """Fraction of the box area covered by dynamic-object masks (intersection
    over the box area), or ``None`` when no segmenter was run."""

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
    """Which recognizer (and segmenter, if any) produced the annotation."""

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


def is_junk_text(text: str, min_length: int = 3, drop_numeric: bool = True) -> bool:
    """Whether a reading cannot identify a place: too short or digits only."""

    value = text.strip()
    return len(value) < min_length or (drop_numeric and value.isdigit())


def replace_frame_texts(frame: FrameText, texts: Sequence[str]) -> FrameText:
    """Return a copy whose detections carry different strings.

    Boxes, confidences and ordering are untouched, so the graph built from the
    result is structurally identical -- which is what makes it a control rather
    than a different experiment.
    """

    if len(texts) != len(frame.detections):
        raise ValueError(
            f"got {len(texts)} texts for {len(frame.detections)} detections"
        )
    return replace(
        frame,
        detections=tuple(
            replace(detection, text=str(text))
            for detection, text in zip(frame.detections, texts, strict=True)
        ),
    )


def frame_text_to_json(value: FrameText, frame_id: str | None = None) -> dict[str, Any]:
    """Serialize a text annotation (one JSON file per frame)."""

    payload: dict[str, Any] = {}
    if frame_id is not None:
        payload["frame_id"] = str(frame_id)
    payload.update(
        {
            "image_size": list(value.image_size),
            "model_identity": dict(value.model_identity),
            "detections": [
                {
                    "box": list(detection.box),
                    "text": detection.text,
                    "confidence": detection.confidence,
                    **(
                        {"dynamic_score": detection.dynamic_score}
                        if detection.dynamic_score is not None
                        else {}
                    ),
                }
                for detection in value.detections
            ],
        }
    )
    return payload


def frame_text_from_json(payload: Mapping[str, Any]) -> FrameText:
    """Read a text annotation written by :func:`frame_text_to_json`."""

    return FrameText(
        detections=tuple(
            TextDetection(
                box=tuple(row["box"]),
                text=row["text"],
                confidence=row["confidence"],
                dynamic_score=row.get("dynamic_score"),
            )
            for row in payload.get("detections", ())
        ),
        image_size=tuple(payload["image_size"]),
        model_identity=payload.get("model_identity", {}),
    )


@dataclass(frozen=True, slots=True)
class FrameMasks:
    """Dynamic-object coverage of one frame.

    The mask is the union of every instance a segmenter considers dynamic,
    stored as COCO run-length encoding. Used only while annotating text.
    """

    dynamic_rle: Mapping[str, Any] | None = None
    """``None`` means the segmenter ran and found no dynamic objects."""
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
