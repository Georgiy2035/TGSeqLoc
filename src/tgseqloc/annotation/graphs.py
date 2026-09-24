"""Scene-graph annotation from detector outputs (Sec. III-C).

Detector-based generators return objects only; relations are derived from the
normalized box geometry. Each object is linked to its nearest object by box
centre (``neighbours`` sets how many; the released annotations use one), and
the relation label is chosen by fixed rules:

* ``in`` -- at least 80 % of the source box lies inside the target box;
* ``standing on`` -- the source's bottom edge is within 0.05 of the target's
  top edge, the source is above it and the two overlap horizontally;
* ``beside`` -- the boxes intersect;
* ``above`` -- the source centre is more than 0.1 higher than the target's;
* ``next to`` -- otherwise.

Two generators are provided:

* ``grounding_dino`` -- Grounding DINO (``IDEA-Research/grounding-dino-base``)
  queried with twenty street classes, box threshold 0.4, text threshold 0.3;
  dynamic classes are not queried at all. Returned phrases are canonicalized to
  the query vocabulary.
* ``mask2former`` -- Mask2Former Swin-L trained on COCO panoptic, instance
  post-processing at threshold 0.5 on a quarter-resolution map; objects of
  dynamic COCO classes are dropped at the detector output.

Output is node-link JSON with coordinates normalized by the frame size.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

#: The twenty classes Grounding DINO is queried with.
STREET_VOCABULARY = (
    "building", "road", "sidewalk", "tree", "sign", "wall", "fence", "traffic light",
    "pole", "window", "door", "sky", "shop window", "awning", "stairs", "bench",
    "trash can", "plant", "banner", "floor",
)

#: COCO thing classes removed from Mask2Former graphs as dynamic or movable.
MASK2FORMER_DYNAMIC_CLASSES = frozenset({
    "airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove", "bear",
    "bicycle", "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus", "cake", "car",
    "carrot", "cat", "cell phone", "clock", "cow", "cup", "dog", "donut", "elephant", "fork",
    "frisbee", "giraffe", "hair drier", "handbag", "horse", "hot dog", "keyboard", "kite",
    "knife", "laptop", "motorcycle", "mouse", "orange", "person", "pizza", "remote",
    "sandwich", "scissors", "sheep", "skateboard", "skis", "snowboard", "spoon",
    "sports ball", "suitcase", "surfboard", "teddy bear", "tennis racket", "tie",
    "toothbrush", "train", "truck", "tv", "umbrella", "vase", "wine glass", "zebra",
})


def relation_label(source: Sequence[float], target: Sequence[float]) -> str:
    """Relation of ``source`` to ``target`` from their normalized xyxy boxes."""

    ax0, ay0, ax1, ay1 = source
    bx0, by0, bx1, by1 = target
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    area = (ax1 - ax0) * (ay1 - ay0)
    inter = ix * iy
    acy, bcy = (ay0 + ay1) / 2, (by0 + by1) / 2
    if area and inter / area >= 0.8:
        return "in"
    if abs(ay1 - by0) <= 0.05 and acy < bcy and ix > 0:
        return "standing on"
    if inter > 0:
        return "beside"
    return "above" if acy - bcy < -0.1 else "next to"


def build_graph(
    objects: Sequence[tuple[str, Sequence[float]]],
    meta: Mapping[str, Any],
    *,
    neighbours: int = 1,
    raw_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Node-link graph of ``(class name, normalized xyxy box)`` objects."""

    centers = [((x0 + x1) / 2, (y0 + y1) / 2) for _, (x0, y0, x1, y1) in objects]
    nodes = []
    for index, (name, box) in enumerate(objects, 1):
        data: dict[str, Any] = {
            "global_id": index,
            "class_name": name,
            "class_id": -1,
            "bbox_2d": {"xyxy": list(box), "center": list(centers[index - 1])},
        }
        if raw_labels is not None and raw_labels[index - 1] != name:
            data["class_name_raw"] = raw_labels[index - 1]
        nodes.append({"id": index, "data": data})
    links = []
    for i in range(len(objects)):
        others = sorted(
            (k for k in range(len(objects)) if k != i),
            key=lambda k: (math.hypot(centers[i][0] - centers[k][0], centers[i][1] - centers[k][1]), k),
        )
        for j in others[: max(0, int(neighbours))]:
            links.append(
                {"source": i + 1, "target": j + 1, "label": relation_label(objects[i][1], objects[j][1])}
            )
    return {"directed": True, "multigraph": True, "graph": dict(meta), "nodes": nodes, "links": links}


def canonical_label(label: str, vocabulary: Sequence[str] = STREET_VOCABULARY) -> str:
    """Map a Grounding DINO phrase to a vocabulary class.

    The earliest vocabulary class occurring in the phrase wins (the longer one
    on a tie); a truncated phrase that begins exactly one class is completed.
    """

    label = label.strip()
    hits = [(label.find(name), -len(name), name) for name in vocabulary if name in label]
    if hits:
        return min(hits)[2]
    prefixed = [name for name in vocabulary if label and name.startswith(label)]
    return prefixed[0] if len(prefixed) == 1 else label


class GroundingDINOGraphs:
    """Objects from Grounding DINO queried with :data:`STREET_VOCABULARY`."""

    model_id = "IDEA-Research/grounding-dino-base"

    def __init__(self, device: str = "cuda", *, box_threshold: float = 0.4,
                 text_threshold: float = 0.3, min_area: float = 0.001) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).eval().to(device)
        self.prompt = ". ".join(STREET_VOCABULARY) + "."
        self.box_threshold, self.text_threshold, self.min_area = box_threshold, text_threshold, min_area

    @property
    def identity(self) -> dict[str, Any]:
        return {"generator": "grounding_dino", "model_id": self.model_id,
                "box_threshold": self.box_threshold, "text_threshold": self.text_threshold,
                "min_area": self.min_area, "vocabulary": list(STREET_VOCABULARY)}

    def objects(self, image) -> tuple[list[tuple[str, tuple[float, ...]]], list[str]]:
        """Canonicalized objects and the raw phrases they came from."""

        width, height = image.size
        with self.torch.inference_mode():
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[(height, width)],
            )[0]
        labels = result.get("text_labels") or result["labels"]
        objects, raw = [], []
        for box, label in zip(result["boxes"].tolist(), labels):
            x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in
                              (box[0] / width, box[1] / height, box[2] / width, box[3] / height))
            # Boxes are clipped to the frame; boxes under 0.1 % of it are dropped.
            if (x1 - x0) * (y1 - y0) < self.min_area:
                continue
            objects.append((canonical_label(label), (x0, y0, x1, y1)))
            raw.append(label)
        return objects, raw


class Mask2FormerGraphs:
    """Objects from Mask2Former Swin-L (COCO panoptic), dynamic classes dropped."""

    model_id = "facebook/mask2former-swin-large-coco-panoptic"

    def __init__(self, device: str = "cuda", *, threshold: float = 0.5,
                 downscale: int = 4, min_pixels: int = 50, keep_dynamic: bool = False) -> None:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch, self.device = torch, device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id).eval().to(device)
        self.id2label = self.model.config.id2label
        self.threshold, self.downscale, self.min_pixels = threshold, downscale, min_pixels
        self.keep_dynamic = keep_dynamic

    @property
    def identity(self) -> dict[str, Any]:
        return {"generator": "mask2former", "model_id": self.model_id, "threshold": self.threshold,
                "downscale": self.downscale, "min_pixels": self.min_pixels,
                "keep_dynamic": self.keep_dynamic}

    def objects(self, image) -> tuple[list[tuple[str, tuple[float, ...]]], list[str]]:
        import numpy as np

        width, height = image.size
        with self.torch.inference_mode():
            inputs = {k: v.to(self.device) for k, v in self.processor(images=image, return_tensors="pt").items()}
            outputs = self.model(**inputs)
            map_height, map_width = height // self.downscale, width // self.downscale
            result = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=[(map_height, map_width)], threshold=self.threshold
            )[0]
        segmentation = result["segmentation"]
        objects: list[tuple[str, tuple[float, ...]]] = []
        if segmentation is None:
            return objects, []
        segmentation = segmentation.cpu().numpy()
        for info in result["segments_info"]:
            name = self.id2label[info["label_id"]]
            if not self.keep_dynamic and name in MASK2FORMER_DYNAMIC_CLASSES:
                continue
            ys, xs = np.nonzero(segmentation == info["id"])
            # Fragments under ~50 pixels of the quarter-resolution map are dropped.
            if len(ys) == 0 or len(ys) < self.min_pixels:
                continue
            box = (xs.min() / map_width, ys.min() / map_height,
                   (xs.max() + 1) / map_width, (ys.max() + 1) / map_height)
            objects.append((name, tuple(float(v) for v in box)))
        return objects, [name for name, _ in objects]


GENERATORS = {"grounding_dino": GroundingDINOGraphs, "mask2former": Mask2FormerGraphs}
