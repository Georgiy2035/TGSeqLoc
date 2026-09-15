"""Маски динамики без чтения кадра: кадры заранее в памяти (cv2 BGR); YOLO11x-seg + объединение масок + RLE."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from pycocotools import mask as mask_utils
sys.path.insert(0, str(Path.home() / "dynseg_benchmark"))
from benchmark.classes import is_dynamic
from ultralytics import YOLO
log = Path(sys.argv[1])
frames = json.load(open(Path(__file__).with_name("frames.json")))
images = [cv2.imread(f["image"]) for f in frames]                                  # чтение — вне замера
model = YOLO(str(Path.home() / "dynseg_benchmark/weights/yolo11x-seg.pt"))
def run(img):
    h, w = img.shape[:2]
    r = model.predict(img, conf=0.25, verbose=False, retina_masks=True)[0]
    union = np.zeros((h, w), np.uint8)
    if r.masks is not None:
        data = r.masks.data.cpu().numpy()
        for i, c in enumerate(r.boxes.cls.tolist()):
            if is_dynamic(model.names[int(c)]):
                m = data[i] > 0.5
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                union |= m
    mask_utils.encode(np.asfortranarray(union))
for img in images[:5]:
    run(img)
torch.cuda.reset_peak_memory_stats(); start = time.time(); times = []
for img in images[5:]:
    t0 = time.perf_counter(); run(img); torch.cuda.synchronize(); times.append(1000 * (time.perf_counter() - t0))
with open(log, "a") as fh:
    fh.write(json.dumps({"stage": "Маски динамики YOLO11x-seg (без чтения кадра)", "device": "GPU", "start": start, "end": time.time(),
                         "times_ms": times, "torch_peak_mb": torch.cuda.max_memory_allocated() / 2**20}) + "\n")
print("yolo готово", len(times))
