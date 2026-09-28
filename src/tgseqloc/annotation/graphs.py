"""Scene-graph annotation from detector outputs (Sec. III-C).

Detector-based generators return objects only; relations are derived from the
normalized box geometry. Each object is linked to its nearest objects by the
distance between box centres (``neighbours`` sets how many; the released
annotations use one), and the relation of source A to target B is chosen by
the first rule that holds:

* ``in`` -- the intersection covers more than 80 % of the area of A;
* ``standing on`` -- the boxes overlap horizontally and the bottom edge of A
  lies within 0.05 of the top edge of B;
* ``beside`` -- the boxes intersect;
* ``above`` -- the centre of A is more than 0.1 higher than the centre of B;
* ``next to`` -- otherwise.

Two generators are provided:

* ``grounding_dino`` -- Grounding DINO (``IDEA-Research/grounding-dino-base``)
  queried with twenty street classes, box threshold 0.4, text threshold 0.3.
  Dynamic classes are not queried at all. Labels are kept as returned, so a
  box matched by two query phrases carries both (``window shop window``).
* ``mask2former`` -- Mask2Former Swin-L trained on COCO panoptic, instance
  post-processing at threshold 0.5 on the full frame; objects whose box covers
  less than 0.1 % of the frame are dropped, and objects of dynamic COCO
  classes are dropped at the detector output, before edges are built.

Output is node-link JSON with coordinates normalized by the frame size.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: The twenty classes Grounding DINO is queried with, in prompt order.
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

    ax1, ay1, ax2, ay2 = source
    bx1, by1, bx2, by2 = target
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    if inter / area > 0.8:
        return "in"
    if ix > 0 and abs(ay2 - by1) < 0.05:
        return "standing on"
    if inter > 0:
        return "beside"
    if (ay1 + ay2) / 2 < (by1 + by2) / 2 - 0.1:
        return "above"
    return "next to"


def build_graph(
    objects: Sequence[tuple[str, Sequence[float]]],
    meta: Mapping[str, Any],
    *,
    neighbours: int = 1,
) -> dict[str, Any]:
    """Node-link graph of ``(class name, normalized xyxy box)`` objects."""

    nodes = []
    for index, (name, (x1, y1, x2, y2)) in enumerate(objects, 1):
        nodes.append({
            "id": index,
            "data": {
                "global_id": index,
                "class_name": name,
                "class_id": -1,
                "bbox_2d": {"xyxy": [x1, y1, x2, y2], "center": [(x1 + x2) / 2, (y1 + y2) / 2]},
            },
        })
    links = []
    if len(objects) > 1:
        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for _, (x1, y1, x2, y2) in objects]
        for i, (cx, cy) in enumerate(centers):
            # Nearest by squared centre distance; ties keep the node order.
            others = sorted(
                (j for j in range(len(objects)) if j != i),
                key=lambda j: (centers[j][0] - cx) ** 2 + (centers[j][1] - cy) ** 2,
            )
            for j in others[: max(0, int(neighbours))]:
                links.append(
                    {"source": i + 1, "target": j + 1, "label": relation_label(objects[i][1], objects[j][1])}
                )
    return {"directed": True, "multigraph": True, "graph": dict(meta), "nodes": nodes, "links": links}


class GroundingDINOGraphs:
    """Objects from Grounding DINO queried with :data:`STREET_VOCABULARY`."""

    model_id = "IDEA-Research/grounding-dino-base"

    def __init__(self, device: str = "cuda", *, box_threshold: float = 0.4,
                 text_threshold: float = 0.3) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device).eval()
        self.prompt = ". ".join(STREET_VOCABULARY) + "."
        self.box_threshold, self.text_threshold = box_threshold, text_threshold

    def objects(self, image) -> list[tuple[str, tuple[float, ...]]]:
        width, height = image.size
        with self.torch.inference_mode():
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(height, width)],
        )[0]
        labels = result.get("text_labels", result.get("labels"))
        objects = []
        for label, box in zip(labels, result["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box)
            objects.append((str(label), (max(0.0, x1 / width), max(0.0, y1 / height),
                                         min(1.0, x2 / width), min(1.0, y2 / height))))
        return objects


class Mask2FormerGraphs:
    """Objects from Mask2Former Swin-L (COCO panoptic), dynamic classes dropped."""

    model_id = "facebook/mask2former-swin-large-coco-panoptic"

    def __init__(self, device: str = "cuda", *, threshold: float = 0.5,
                 min_area: float = 0.001, keep_dynamic: bool = False) -> None:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch, self.device = torch, device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id).to(device).eval()
        self.id2label = self.model.config.id2label
        self.threshold, self.min_area, self.keep_dynamic = threshold, min_area, keep_dynamic

    def objects(self, image) -> list[tuple[str, tuple[float, ...]]]:
        torch = self.torch
        width, height = image.size
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=[(height, width)], threshold=self.threshold
        )[0]
        segmentation = result["segmentation"]
        objects = []
        for info in result["segments_info"]:
            mask = segmentation == info["id"]
            if not bool(mask.any()):
                continue
            ys, xs = torch.where(mask)
            x1, x2 = float(xs.min()) / width, float(xs.max() + 1) / width
            y1, y2 = float(ys.min()) / height, float(ys.max() + 1) / height
            # Boxes under 0.1 % of the frame are dropped.
            if (x2 - x1) * (y2 - y1) < self.min_area:
                continue
            objects.append((self.id2label.get(info["label_id"], "unknown"), (x1, y1, x2, y2)))
        if not self.keep_dynamic:
            objects = [obj for obj in objects if obj[0].lower() not in MASK2FORMER_DYNAMIC_CLASSES]
        return objects


GENERATORS = {"grounding_dino": GroundingDINOGraphs, "mask2former": Mask2FormerGraphs}
