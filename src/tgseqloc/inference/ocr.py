"""OCR backends that run recognition themselves.

Ported from the standalone ocr_benchmark adapter, which is what produced every
result the project has measured so far, so behaviour stays comparable: the same
detection and recognition models, the same EXIF handling, the same treatment of
empty strings.

The eslav recognizer covers Latin and Cyrillic. Scripts outside it (Kannada,
Thai, Japanese) are recognized as garbage rather than skipped, so a dataset
containing them needs a different recognizer, not a different threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tgseqloc.data.formats import FrameText, TextDetection

DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "eslav_PP-OCRv5_mobile_rec"
ORIENTATION_MODEL = "PP-LCNet_x1_0_textline_ori"

#: PaddleOCR fetches these on first use into its own cache; the manifest entry
#: only reports whether they are already there.
PADDLE_MODELS = (DETECTION_MODEL, RECOGNITION_MODEL, ORIENTATION_MODEL)


def load_oriented_image(image_path: str | Path):
    """Open an image with EXIF rotation applied, as the benchmark did.

    Skipping this silently rotates boxes on any camera that stores orientation
    in metadata instead of in the pixels.
    """

    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


@dataclass(slots=True)
class PaddleOCRv5:
    """PP-OCRv5 detection with the eslav recognizer.

    ``load`` is separate from construction so that configuration can be checked,
    and missing weights reported, before anything reaches the GPU.
    """

    device: str = "cpu"
    text_detection_model: str = DETECTION_MODEL
    text_recognition_model: str = RECOGNITION_MODEL
    use_textline_orientation: bool = True
    confidence_threshold: float = 0.0
    noop_texts: tuple[str, ...] = ()
    _engine: Any = field(default=None, init=False, repr=False)

    @staticmethod
    def weight_requirements() -> tuple[str, ...]:
        return ("paddleocr_v5_eslav",)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        """Identity that must invalidate prepared graphs when it changes."""

        return {
            "backend": "paddleocr_v5",
            "text_detection_model": self.text_detection_model,
            "text_recognition_model": self.text_recognition_model,
            "use_textline_orientation": bool(self.use_textline_orientation),
            "confidence_threshold": float(self.confidence_threshold),
            "noop_texts": tuple(sorted(self.noop_texts)),
        }

    def load(self) -> None:
        if self._engine is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError(
                "paddleocr_v5 requires the optional paddleocr package; "
                "install it or select a different ocr backend"
            ) from error
        self._engine = PaddleOCR(
            text_detection_model_name=self.text_detection_model,
            text_recognition_model_name=self.text_recognition_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=self.use_textline_orientation,
            device=_paddle_device(self.device),
        )

    def predict(self, image_paths: Sequence[str | Path]) -> list[FrameText]:
        """Recognize every frame, one at a time, in the given order."""

        if self._engine is None:
            raise RuntimeError("call load() before predict()")
        return [self.predict_one(path) for path in image_paths]

    def predict_one(self, image_path: str | Path) -> FrameText:
        image = load_oriented_image(image_path)
        width, height = image.size
        started = time.perf_counter()

        import numpy as np

        detections: list[TextDetection] = []
        noops = {value.strip().casefold() for value in self.noop_texts}
        for result in self._engine.predict(input=np.asarray(image)):
            data = result.json["res"]
            for text, score, polygon in zip(
                data["rec_texts"], data["rec_scores"], data["rec_polys"], strict=True
            ):
                detection = _make_detection(
                    text, score, polygon, width, height,
                    threshold=self.confidence_threshold, noops=noops,
                )
                if detection is not None:
                    detections.append(detection)
        return FrameText(
            detections=tuple(detections),
            image_size=(width, height),
            model_identity={
                **self.cache_identity,
                "inference_time_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )


def _paddle_device(device: str) -> str:
    """Translate the pipeline's device name into PaddlePaddle's own.

    ``auto`` is resolved with paddle's runtime rather than torch's: this
    backend usually runs where torch is not installed at all.
    """

    value = str(device)
    if value.startswith("cuda") or value.startswith("gpu"):
        return "gpu:0"
    if value != "auto":
        return "cpu"
    try:
        import paddle

        return "gpu:0" if paddle.device.cuda.device_count() > 0 else "cpu"
    except Exception:  # noqa: BLE001 - any failure here just means no GPU
        return "cpu"


def _make_detection(
    text: str,
    score: float,
    polygon: Iterable[Sequence[float]],
    width: int,
    height: int,
    *,
    threshold: float,
    noops: set[str],
) -> TextDetection | None:
    """Normalize one recognized polygon, or drop it if it carries nothing."""

    text = str(text).strip()
    confidence = float(score)
    if not text or text.casefold() in noops or confidence < threshold:
        return None
    points = [point for point in polygon if len(point) >= 2]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    box = (
        min(1.0, max(0.0, min(xs) / width)),
        min(1.0, max(0.0, min(ys) / height)),
        min(1.0, max(0.0, max(xs) / width)),
        min(1.0, max(0.0, max(ys) / height)),
    )
    # A polygon can collapse after clipping to the frame; such a box carries no
    # location and would fail FrameText validation.
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return TextDetection(box=box, text=text, confidence=confidence)


def build_paddleocr_v5(
    device: str = "cpu",
    *,
    confidence_threshold: float = 0.0,
    noop_texts: Sequence[str] = (),
    **params: Any,
) -> PaddleOCRv5:
    """Registry factory for the ``paddleocr_v5`` OCR backend."""

    unknown = sorted(set(params) - {"text_detection_model", "text_recognition_model",
                                    "use_textline_orientation"})
    if unknown:
        raise ValueError(
            f"unknown paddleocr_v5 params: {', '.join(unknown)}; "
            "supported: text_detection_model, text_recognition_model, "
            "use_textline_orientation, confidence_threshold, noop_texts"
        )
    return PaddleOCRv5(
        device=device,
        confidence_threshold=float(confidence_threshold),
        noop_texts=tuple(noop_texts),
        **params,
    )


build_paddleocr_v5.weight_requirements = PaddleOCRv5.weight_requirements
