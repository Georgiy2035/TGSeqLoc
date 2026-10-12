"""YOLO11x-seg по mono-камерам RobotCar: те же кадры, что у PaddleOCR (симлинки на кадры графов).

Формат выхода совпадает с run_robotcar_seg.py (фронтальная камера): <кадр>/yolo11x_seg.json с масками
динамических классов в RLE, так что compute_ioa-логика применяется без изменений.
"""
import argparse, json, statistics, time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from benchmark.classes import canonical, is_dynamic

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "weights" / "yolo11x-seg.pt"))
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    times, saved, with_dyn = [], 0, 0
    for path in paths:
        target = args.output / path.stem / "yolo11x_seg.json"
        if target.exists():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        started = time.perf_counter()
        result = model.predict(image, conf=args.confidence, verbose=False, retina_masks=True)[0]
        times.append((time.perf_counter() - started) * 1000)
        instances = []
        if result.masks is not None:
            data = result.masks.data.cpu().numpy()
            for i, (class_id, score) in enumerate(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())):
                label = model.names[int(class_id)]
                if not is_dynamic(label):
                    continue
                mask = data[i] > 0.5
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                if not mask.any():
                    continue
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                ys, xs = np.nonzero(mask)
                instances.append({
                    "label": canonical(label), "score": round(float(score), 4),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "area": int(mask.sum()),
                    "rle": {"size": rle["size"], "counts": rle["counts"].decode("ascii")},
                })
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "model": "yolo11x_seg", "image": path.name, "image_width": width, "image_height": height,
            "inference_time_ms": round(times[-1], 3), "instances": instances,
        }, ensure_ascii=False), encoding="utf-8")
        saved += 1; with_dyn += int(bool(instances))
        if saved % 1000 == 0:
            print(f"  {args.images.parent.name}/{args.images.name}: кадров {saved}, медиана {statistics.median(times):.0f} мс", flush=True)
    print(json.dumps({"images": str(args.images), "saved": saved, "with_dynamic": with_dyn,
                      "median_ms": round(statistics.median(times), 1) if times else None}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
