"""Deciding which recognized text sits on something ephemeral.

The measure is IoA — the fraction of the *text box* covered by dynamic masks —
not IoU. A sign occupies a handful of pixels next to a bus filling half the
frame, so their IoU is near zero however completely the bus carries the text.
Dividing by the text area asks the question that actually matters: how much of
this string is on that object.

The score stays continuous and the threshold is applied by the caller, so
changing where the line falls does not require running the models again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tgseqloc.data.formats import FrameMasks, FrameText


@dataclass(slots=True)
class MaskIoA:
    """Scores detections by how much of their area dynamic masks cover."""

    threshold: float = 0.5

    def __post_init__(self) -> None:
        threshold = float(self.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        self.threshold = threshold

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {"backend": "mask_ioa", "threshold": self.threshold}

    def score(self, frame_text: FrameText, frame_masks: FrameMasks) -> list[float]:
        """Return the covered fraction of every detection, in detection order."""

        if frame_text.image_size != frame_masks.image_size:
            raise ValueError(
                "text and masks describe different frames: "
                f"{frame_text.image_size} vs {frame_masks.image_size}"
            )
        if not frame_text.detections:
            return []
        if frame_masks.is_empty:
            # The segmenter ran and found nothing; every box is uncovered.
            return [0.0] * len(frame_text.detections)

        from tgseqloc.inference.segmentation import decode_mask

        width, height = frame_text.image_size
        mask = decode_mask(frame_masks.dynamic_rle, (height, width))
        return [
            _covered_fraction(detection.box, mask, width, height)
            for detection in frame_text.detections
        ]

    def classify(
        self, frame_text: FrameText, frame_masks: FrameMasks
    ) -> list[bool]:
        """Return which detections count as dynamic at the configured threshold."""

        return [value >= self.threshold for value in self.score(frame_text, frame_masks)]

    def apply(self, frame_text: FrameText, frame_masks: FrameMasks) -> FrameText:
        """Record the scores and drop what the threshold calls dynamic."""

        scores = self.score(frame_text, frame_masks)
        scored = frame_text.with_dynamic_scores(scores)
        return scored.select(value < self.threshold for value in scores)


def _covered_fraction(
    box: Sequence[float], mask: Any, width: int, height: int
) -> float:
    """Fraction of a normalized box's pixels that the mask covers."""

    x1 = max(0, int(round(box[0] * width)))
    y1 = max(0, int(round(box[1] * height)))
    x2 = min(width, int(round(box[2] * width)))
    y2 = min(height, int(round(box[3] * height)))
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        # A box thinner than a pixel after rounding covers nothing measurable;
        # calling it static leaves it in the graph, which is the safe default.
        return 0.0
    return float(patch.sum()) / float(patch.size)


def build_mask_ioa(threshold: float = 0.5, **params: Any) -> MaskIoA:
    """Registry factory for the ``mask_ioa`` text-dynamics backend."""

    if params:
        raise ValueError(
            f"unknown mask_ioa params: {', '.join(sorted(params))}; supported: threshold"
        )
    return MaskIoA(threshold=float(threshold))
