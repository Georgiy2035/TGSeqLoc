"""Dynamic-object segmentation for the text filter (Sec. III-B).

The output is the union mask of every dynamic instance: the only question asked
of it is whether a string lies on something that will not be there on the next
traversal. Thirteen COCO classes count as dynamic -- carriers that move
(people, vehicles, bags) and objects whose displayed content changes (screens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tgseqloc.data.formats import FrameMasks

# Carriers that leave the scene.
EPHEMERAL = (
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "backpack", "handbag", "suitcase",
)
# Objects that stay but whose displayed text changes.
VOLATILE_CONTENT = ("tv", "laptop", "cell phone")
DYNAMIC_CLASSES = EPHEMERAL + VOLATILE_CONTENT
OPEN_VOCAB_PROMPTS = DYNAMIC_CLASSES + (
    "monitor", "information display", "screen",
)

# Checkpoints name some COCO classes differently; without these aliases
# motorcycles and screens would silently be missed.
LABEL_ALIASES = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "tvmonitor": "tv",
    "tv monitor": "tv",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "mobile phone": "cell phone",
    "monitor": "tv",
    "screen": "tv",
}


def canonical(label: str) -> str:
    key = str(label).strip().lower()
    return LABEL_ALIASES.get(key, key)


def is_dynamic(label: str, dynamic_classes: Sequence[str] = DYNAMIC_CLASSES) -> bool:
    return canonical(label) in {canonical(name) for name in dynamic_classes}


def dynamic_indices(
    names: Mapping[int, str], dynamic_classes: Sequence[str] = DYNAMIC_CLASSES
) -> set[int]:
    """Class indices of a model's label table that count as dynamic."""

    return {
        int(index)
        for index, name in names.items()
        if is_dynamic(name, dynamic_classes)
    }


def encode_mask(mask: Any) -> dict[str, Any] | None:
    """Encode a boolean mask as COCO RLE, or ``None`` when nothing is set.

    ``counts`` is decoded to ASCII so the payload survives a JSON round trip,
    which is the form the rest of the project already stores.
    """

    import numpy as np
    from pycocotools import mask as mask_utils

    if not mask.any():
        return None
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }


def decode_mask(rle: Mapping[str, Any] | None, shape: tuple[int, int]):
    """Decode COCO RLE back to a boolean mask of ``(height, width)``."""

    import numpy as np

    if rle is None:
        return np.zeros(shape, dtype=bool)
    from pycocotools import mask as mask_utils

    payload = {
        "size": list(rle["size"]),
        "counts": rle["counts"].encode("ascii")
        if isinstance(rle["counts"], str)
        else rle["counts"],
    }
    return mask_utils.decode(payload).astype(bool)


@dataclass(slots=True)
class YoloSeg:
    """YOLO11-seg (Ultralytics) with the closed COCO class set.

    ``weights_path`` is a local checkpoint or an Ultralytics model name such as
    ``yolo11x-seg.pt``, which Ultralytics downloads on first use.
    """

    weights_path: Path
    device: str = "cpu"
    confidence: float = 0.25
    dynamic_classes: tuple[str, ...] = DYNAMIC_CLASSES
    _model: Any = field(default=None, init=False, repr=False)
    _dynamic: set[int] = field(default_factory=set, init=False, repr=False)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": "yolo_seg",
            "weights": Path(self.weights_path).name,
            "confidence": float(self.confidence),
            "dynamic_classes": tuple(sorted(self.dynamic_classes)),
        }

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "yolo_seg requires the optional ultralytics package; "
                "install it or select a different segmentation backend"
            ) from error
        self._model = YOLO(str(self.weights_path))
        self._dynamic = dynamic_indices(self._model.names, self.dynamic_classes)
        if not self._dynamic:
            raise RuntimeError(
                f"none of {self.dynamic_classes} appear in the checkpoint's classes; "
                "every frame would come back with an empty mask"
            )

    def predict(self, image_paths: Sequence[str | Path]) -> list[FrameMasks]:
        if self._model is None:
            raise RuntimeError("call load() before predict()")
        return [self.predict_one(path) for path in image_paths]

    def predict_one(self, image_path: str | Path) -> FrameMasks:
        import numpy as np

        from tgseqloc.annotation.ocr import load_oriented_image

        image = load_oriented_image(image_path)
        width, height = image.size
        # Ultralytics treats an array as BGR, the order OpenCV would have given
        # it had we passed a path. Handing it RGB swaps two channels and shifts
        # the masks by a few percent of their pixels.
        result = self._model.predict(
            np.asarray(image)[:, :, ::-1],
            conf=self.confidence,
            verbose=False,
            retina_masks=True,
            device=self.device,
        )[0]

        union = np.zeros((height, width), dtype=bool)
        if result.masks is not None:
            data = result.masks.data.cpu().numpy()
            for index, class_id in enumerate(result.boxes.cls.tolist()):
                if int(class_id) not in self._dynamic:
                    continue
                union |= _fit(data[index] > 0.5, height, width)
        return FrameMasks(
            dynamic_rle=encode_mask(union),
            image_size=(width, height),
            model_identity=dict(self.cache_identity),
        )


@dataclass(slots=True)
class YoloESeg(YoloSeg):
    """YOLOE open-vocabulary segmentation prompted with the dynamic classes."""

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": "yoloe_seg",
            "weights": Path(self.weights_path).name,
            "confidence": float(self.confidence),
            "prompts": tuple(OPEN_VOCAB_PROMPTS),
            "dynamic_classes": tuple(sorted(self.dynamic_classes)),
        }

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOE
        except ImportError as error:
            raise RuntimeError("yoloe_seg requires ultralytics with YOLOE support") from error
        self._model = YOLOE(str(self.weights_path))
        prompts = list(OPEN_VOCAB_PROMPTS)
        self._model.set_classes(prompts, self._model.get_text_pe(prompts))
        self._dynamic = set(range(len(prompts)))


def _fit(mask: Any, height: int, width: int) -> Any:
    """Resize a mask to the frame, keeping it boolean.

    Nearest-neighbour interpolation is deliberate: a mask must stay a mask, and
    smoothing would invent partially covered pixels.
    """

    import numpy as np

    if mask.shape == (height, width):
        return mask.astype(bool)
    from PIL import Image

    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), Image.NEAREST
    )
    return np.asarray(resized) > 127


SEGMENTERS = {
    # name: (class, weights). yolo11x_seg is the pipeline's segmenter; the other
    # two are the substitutes of the filtering ablation (Sec. IV-E.1).
    "yolo11x_seg": (YoloSeg, "yolo11x-seg.pt"),
    "yolo11m_seg": (YoloSeg, "yolo11m-seg.pt"),
    "yoloe_11l_seg": (YoloESeg, "yoloe-11l-seg.pt"),
}


def build_segmenter(
    name: str,
    device: str = "cpu",
    *,
    confidence: float = 0.25,
    weights: str | Path | None = None,
) -> YoloSeg:
    """A segmenter by name, with the confidence threshold used throughout (0.25)."""

    if name not in SEGMENTERS:
        raise ValueError(f"unknown segmenter {name!r}; available: {', '.join(SEGMENTERS)}")
    cls, default_weights = SEGMENTERS[name]
    return cls(
        weights_path=Path(weights or default_weights), device=device, confidence=float(confidence)
    )
