"""Scene-text recognizers for the text annotation (Sec. III-B, Table I).

Two back ends share one output: detections ``(string, box, confidence)`` with
boxes normalized by the frame size.

* ``qwen3_vl_4b`` -- Qwen3-VL-4B-Instruct served by vLLM with greedy decoding,
  the accurate back end used where a discrete GPU is available;
* ``paddleocr_v5`` -- PP-OCRv5 server detector with the East Slavic recognizer
  (Cyrillic and Latin), the efficient back end used on board.

Both first produce predictions with pixel boxes ``[x, y, width, height]`` and
convert them with :func:`predictions_to_frame_text`, which is the conversion
the released annotations went through.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tgseqloc.data.formats import FrameText, TextDetection

#: Readings that mean "no text" rather than text.
DEFAULT_NOOP_TEXTS = frozenset(
    {"", "none", "null", "n/a", "no text", "[no text]", "<no_text>", "<no text>"}
)


def load_oriented_image(image_path: str | Path):
    """Open an image as RGB with its EXIF orientation applied."""

    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def predictions_to_frame_text(
    predictions: Iterable[Mapping[str, Any]],
    width: int,
    height: int,
    *,
    model_identity: Mapping[str, Any] | None = None,
    noop_texts: Iterable[str] = DEFAULT_NOOP_TEXTS,
) -> FrameText:
    """Normalize pixel-space predictions into a :class:`FrameText`.

    A prediction is ``{"bbox": [x, y, w, h] | polygon, "text": str,
    "confidence": float | None}``. An absent confidence (vision-language models
    report none) is stored as 1.0. Empty and "no text" readings and boxes that
    collapse after clipping to the frame are dropped.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    noops = {str(value).strip().casefold() for value in noop_texts}
    detections: list[TextDetection] = []
    for row in predictions:
        reported = row.get("confidence", row.get("score"))
        confidence = 1.0 if reported is None else float(reported)
        text = str(row.get("text", "")).strip()
        if text.casefold() in noops:
            continue
        raw_box = row.get("bbox", row.get("box"))
        if not isinstance(raw_box, (list, tuple)):
            continue
        if len(raw_box) == 4 and all(isinstance(value, (int, float)) for value in raw_box):
            x, y, box_width, box_height = map(float, raw_box)
            x1, y1, x2, y2 = x, y, x + box_width, y + box_height
        else:
            points = [point for point in raw_box if isinstance(point, (list, tuple)) and len(point) >= 2]
            if not points:
                continue
            xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        normalized = [
            min(1.0, max(0.0, x1 / width)),
            min(1.0, max(0.0, y1 / height)),
            min(1.0, max(0.0, x2 / width)),
            min(1.0, max(0.0, y2 / height)),
        ]
        if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
            detections.append(TextDetection(box=tuple(normalized), text=text, confidence=confidence))
    return FrameText(
        detections=tuple(detections),
        image_size=(int(width), int(height)),
        model_identity=dict(model_identity or {}),
    )


# --------------------------------------------------------------------------- PaddleOCR


@dataclass(slots=True)
class PaddleOCRv5:
    """PP-OCRv5 server detection with the eslav recognizer."""

    device: str = "cpu"
    _engine: Any = field(default=None, init=False, repr=False)

    identity = {
        "ocr": "paddleocr_v5",
        "text_detection_model": "PP-OCRv5_server_det",
        "text_recognition_model": "eslav_PP-OCRv5_mobile_rec",
    }

    def load(self) -> None:
        if self._engine is not None:
            return
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            text_detection_model_name=self.identity["text_detection_model"],
            text_recognition_model_name=self.identity["text_recognition_model"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="gpu:0" if str(self.device).startswith(("cuda", "gpu")) else "cpu",
        )

    def predict(self, image_paths: Sequence[str | Path]) -> list[FrameText]:
        return [self.predict_one(path) for path in image_paths]

    def predict_one(self, image_path: str | Path) -> FrameText:
        import numpy as np

        self.load()
        image = load_oriented_image(image_path)
        started = time.perf_counter()
        predictions = []
        for result in self._engine.predict(input=np.asarray(image)):
            data = result.json["res"]
            for text, score, polygon in zip(
                data["rec_texts"], data["rec_scores"], data["rec_polys"], strict=True
            ):
                text = text.strip()
                if not text:
                    continue
                xs = [float(point[0]) for point in polygon]
                ys = [float(point[1]) for point in polygon]
                predictions.append(
                    {
                        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                        "text": text,
                        "confidence": float(score),
                    }
                )
        identity = {**self.identity, "ms": round((time.perf_counter() - started) * 1000, 2)}
        return predictions_to_frame_text(
            predictions, image.width, image.height, model_identity=identity
        )


# --------------------------------------------------------------------------- Qwen3-VL

#: The exact word-spotting prompt used for every Qwen3-VL annotation. It is in
#: Russian because the target imagery carries Russian and English signage; it
#: asks for every separate Russian or English word as a JSON array of
#: {"bbox": [x_min, y_min, x_max, y_max], "text": word} with integer coordinates
#: normalized to 0-1000, without merging words or adding text that is absent.
WORD_SPOTTING_PROMPT = """
Найди и распознай каждое отдельное русское или английское слово на изображении.
Верни только JSON-массив без Markdown и пояснений.
Каждый элемент должен иметь формат:
{"bbox": [x_min, y_min, x_max, y_max], "text": "слово"}.
Координаты должны быть целыми числами, нормализованными в диапазоне 0–1000.
Не объединяй несколько слов в одну запись. Не добавляй текст, которого нет на изображении.
""".strip()

_BOX_PATTERN = re.compile(
    r"(?:<box>)?\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*(?:</box>)?"
)
_EMPTY_ARRAY_PATTERN = re.compile(r"\s*(?:```(?:json)?\s*)?\[\s*\](?:\s*```)?\s*", re.IGNORECASE)


def parse_word_predictions(raw_output: str, image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Predictions with pixel boxes ``[x, y, w, h]`` from a 0-1000 JSON answer.

    Raises ``ValueError`` when the answer is not a JSON array of words; the frame
    is then annotated with no detections and the error is recorded.
    """

    items = _decode_json_array(raw_output)
    predictions = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = _decode_box(item.get("bbox") or item.get("box") or item.get("bbox_2d"))
        text = item.get("text") or item.get("label")
        if box is None or not isinstance(text, str):
            continue
        x_min, y_min, x_max, y_max = box
        if any(value < 0 or value > 1000 for value in box):
            raise ValueError(f"normalized coordinates outside 0-1000: {box}")
        x_min, x_max = x_min / 1000 * image_width, x_max / 1000 * image_width
        y_min, y_max = y_min / 1000 * image_height, y_max / 1000 * image_height
        x_min, x_max = min(max(x_min, 0), image_width), min(max(x_max, 0), image_width)
        y_min, y_max = min(max(y_min, 0), image_height), min(max(y_max, 0), image_height)
        if x_max <= x_min or y_max <= y_min or not text.strip():
            continue
        predictions.append(
            {"bbox": [x_min, y_min, x_max - x_min, y_max - y_min], "text": text.strip(), "confidence": None}
        )
    if not items and _EMPTY_ARRAY_PATTERN.fullmatch(raw_output) is None:
        raise ValueError("an empty JSON array is accompanied by other output")
    if items and not predictions:
        raise ValueError("the answer has items but no valid bbox and text")
    return predictions


def _decode_json_array(raw_output: str) -> list:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
    cleaned = re.sub(r",\s*]", "]", cleaned)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    raise ValueError("the model did not return a JSON array")


def _decode_box(value: object) -> list[float] | None:
    if isinstance(value, list) and len(value) == 4:
        try:
            return [float(coordinate) for coordinate in value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = _BOX_PATTERN.fullmatch(value.strip())
        if match is not None:
            return [float(coordinate) for coordinate in match.groups()]
    return None


@dataclass(slots=True)
class Qwen3VLOCR:
    """Qwen3-VL-4B-Instruct word spotting through vLLM, greedy decoding."""

    batch_size: int = 4
    gpu_memory_utilization: float = 0.85
    _llm: Any = field(default=None, init=False, repr=False)

    identity = {
        "ocr": "qwen3_vl_4b",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    }
    max_model_len = 4096
    max_tokens = 2048

    def load(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        self._llm = LLM(
            model=self.identity["model_id"],
            revision=self.identity["revision"],
            trust_remote_code=True,
            dtype="auto",
            max_model_len=self.max_model_len,
            max_num_seqs=self.batch_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1, "video": 0},
            mm_processor_cache_gb=0,
            enable_prefix_caching=False,
            # The released annotations were produced in eager mode.
            enforce_eager=True,
        )

    def predict(self, image_paths: Sequence[str | Path]) -> list[FrameText]:
        from vllm import SamplingParams

        self.load()
        sampling = SamplingParams(temperature=0.0, max_tokens=self.max_tokens, skip_special_tokens=True)
        results: list[FrameText] = []
        for offset in range(0, len(image_paths), self.batch_size):
            images = [load_oriented_image(path) for path in image_paths[offset: offset + self.batch_size]]
            conversations = [
                [{"role": "user", "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": WORD_SPOTTING_PROMPT},
                ]}]
                for image in images
            ]
            outputs = self._llm.chat(conversations, sampling_params=sampling)
            for image, output in zip(images, outputs, strict=True):
                identity: dict[str, Any] = dict(self.identity)
                try:
                    predictions = parse_word_predictions(output.outputs[0].text, image.width, image.height)
                except ValueError as error:
                    predictions, identity["error"] = [], str(error)
                results.append(
                    predictions_to_frame_text(predictions, image.width, image.height, model_identity=identity)
                )
        return results


RECOGNIZERS = {"qwen3_vl_4b": Qwen3VLOCR, "paddleocr_v5": PaddleOCRv5}


def build_recognizer(name: str, device: str = "cuda"):
    """A recognizer by name."""

    if name == "paddleocr_v5":
        return PaddleOCRv5(device=device)
    if name == "qwen3_vl_4b":
        return Qwen3VLOCR()
    raise ValueError(f"unknown recognizer {name!r}; available: {', '.join(RECOGNIZERS)}")
