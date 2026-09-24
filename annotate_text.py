"""Text annotation of a folder of frames (Fig. 4, text branch).

For every image: recognize scene text, segment dynamic objects, and record for
each string the fraction of its box covered by dynamic-object masks
(``dynamic_score``). With filtering (the default) the junk filter drops strings
shorter than three characters and digit-only strings, and the dynamic-text
filter drops strings with ``dynamic_score >= 0.5``. ``--no_filter`` keeps every
string with its score; preparation then applies the same filters, so both
outputs lead to the same graphs. The released annotations are unfiltered.

One JSON file per frame is written to ``--output_folder/{frame_id}.json``,
where the frame identifier is the image file name without extension.

    python annotate_text.py --images_folder datasets/tgseqloc_street/27-08-26 \\
        --output_folder annotations/tgseqloc_street/text/paddleocr_v5-yolo11x_seg/27-08-26 \\
        --ocr paddleocr_v5 --segmenter yolo11x_seg --stride 10

Environment: PaddleOCR needs ``paddlepaddle-gpu`` and ``paddleocr``; Qwen3-VL
needs ``vllm``; segmentation needs ``ultralytics`` and ``pycocotools`` (see
requirements_annotation.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgseqloc.annotation.dynamics import dynamic_scores
from tgseqloc.annotation.ocr import RECOGNIZERS, build_recognizer
from tgseqloc.annotation.segmentation import SEGMENTERS, build_segmenter
from tgseqloc.data.formats import frame_text_to_json, is_junk_text

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--ocr", choices=sorted(RECOGNIZERS), default="qwen3_vl_4b")
    parser.add_argument("--segmenter", choices=sorted(SEGMENTERS) + ["none"], default="yolo11x_seg")
    parser.add_argument("--no_filter", action="store_true",
                        help="keep junk and dynamic strings (with their dynamic_score)")
    parser.add_argument("--dynamic_threshold", type=float, default=0.5, help="tau of the dynamic-text filter")
    parser.add_argument("--stride", type=int, default=1, help="annotate every N-th image")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    images = sorted(p for p in args.images_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = images[:: max(1, args.stride)]
    pending = [p for p in images if not (args.output_folder / f"{p.stem}.json").exists()]
    args.output_folder.mkdir(parents=True, exist_ok=True)
    recognizer = build_recognizer(args.ocr, device=args.device)
    segmenter = None if args.segmenter == "none" else build_segmenter(args.segmenter, device=args.device)
    if segmenter is not None:
        segmenter.load()

    written = kept = total = 0
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset: offset + args.batch_size]
        for path, frame_text in zip(batch, recognizer.predict(batch), strict=True):
            identity = {**frame_text.model_identity, "segmenter": args.segmenter}
            if segmenter is not None:
                frame_text = frame_text.with_dynamic_scores(
                    dynamic_scores(frame_text, segmenter.predict_one(path))
                )
            total += len(frame_text)
            if not args.no_filter:
                frame_text = frame_text.select(
                    not is_junk_text(d.text)
                    and (d.dynamic_score is None or d.dynamic_score < args.dynamic_threshold)
                    for d in frame_text.detections
                )
            kept += len(frame_text)
            payload = frame_text_to_json(frame_text, frame_id=path.stem)
            payload["model_identity"] = identity
            (args.output_folder / f"{path.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            written += 1
    print(json.dumps({"frames": len(images), "written": written, "strings": total, "kept": kept}))


if __name__ == "__main__":
    main()
