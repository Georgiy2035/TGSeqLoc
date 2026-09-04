"""Segmentation backends marking pixels that belong to ephemeral objects.

Ported from the standalone dynseg_benchmark adapter. The output is the union of
every dynamic instance rather than per-instance masks: the only question asked
downstream is whether a piece of text sits on something that will not be there
next time, and the union answers it.

Two groups of classes count as dynamic, for different reasons. Ephemeral
objects physically leave within hours. Volatile-content objects stay put but the
text on them changes, which for text-based place recognition is just as useless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tgseqloc.data.formats import FrameMasks

# Объект физически исчезнет через часы.
EPHEMERAL = (
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "backpack", "handbag", "suitcase",
)
# Объект стоит, но текст на нём меняется.
VOLATILE_CONTENT = ("tv", "laptop", "cell phone")
DYNAMIC_CLASSES = EPHEMERAL + VOLATILE_CONTENT

# Разные чекпоинты используют разную номенклатуру COCO. Без этого маппинга
# мотоциклы и телевизоры молча теряются.
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
    """YOLO11-seg and compatible checkpoints with the closed COCO class set."""

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
        path = Path(self.weights_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"segmentation weights are absent: {path}; run tgseqloc weights sync"
            )
        self._model = YOLO(str(path))
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

        from tgseqloc.inference.ocr import load_oriented_image

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


def build_yolo_seg(
    device: str = "cpu",
    *,
    weights_path: str | Path,
    confidence: float = 0.25,
    dynamic_classes: Sequence[str] = DYNAMIC_CLASSES,
    **params: Any,
) -> YoloSeg:
    """Registry factory for the ``yolo_seg`` segmentation backend."""

    if params:
        raise ValueError(
            f"unknown yolo_seg params: {', '.join(sorted(params))}; "
            "supported: confidence, dynamic_classes"
        )
    return YoloSeg(
        weights_path=Path(weights_path),
        device=device,
        confidence=float(confidence),
        dynamic_classes=tuple(dynamic_classes),
    )


build_yolo_seg.weight_requirements = lambda: ("yolo11x_seg",)
