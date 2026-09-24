"""Dynamic score of a string: the fraction of its box covered by dynamic masks.

The measure is intersection over the box area (IoA), not IoU: a word is much
smaller than the vehicle carrying it, so their IoU stays near zero however
completely the vehicle carries the word (Sec. III-B). The score is stored in the
text annotation; the threshold tau is applied when the annotation is read.
"""

from __future__ import annotations

from typing import Any, Sequence

from tgseqloc.data.formats import FrameMasks, FrameText


def dynamic_scores(frame_text: FrameText, frame_masks: FrameMasks) -> list[float]:
    """IoA of every detection with the dynamic mask, in detection order."""

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

    from tgseqloc.annotation.segmentation import decode_mask

    width, height = frame_text.image_size
    mask = decode_mask(frame_masks.dynamic_rle, (height, width))
    return [
        _covered_fraction(detection.box, mask, width, height)
        for detection in frame_text.detections
    ]


def _covered_fraction(box: Sequence[float], mask: Any, width: int, height: int) -> float:
    """Fraction of a normalized box's pixels that the mask covers."""

    x1 = max(0, int(round(box[0] * width)))
    y1 = max(0, int(round(box[1] * height)))
    x2 = min(width, int(round(box[2] * width)))
    y2 = min(height, int(round(box[3] * height)))
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        # A box thinner than a pixel after rounding covers nothing measurable.
        return 0.0
    return float(patch.sum()) / float(patch.size)
