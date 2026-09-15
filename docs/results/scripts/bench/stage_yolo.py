"""Маски динамики: YOLO11x-seg + объединение масок динамических классов + RLE, как в пайплайне."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from pycocotools import mask as mask_utils
sys.path.insert(0, str(Path.home() / "dynseg_benchmark"))
from benchmark.classes import is_dynamic
from ultralytics import YOLO
log = Path(sys.argv[1])
frames = json.load(open(Path(__file__).with_name("frames.json")))
model = YOLO(str(Path.home() / "dynseg_benchmark/weights/yolo11x-seg.pt"))
def run(path):
    img = cv2.imread(path); h, w = img.shape[:2]
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
for f in frames[:5]:
    run(f["image"])
torch.cuda.reset_peak_memory_stats(); start = time.time(); times = []
for f in frames[5:]:
    img_t0 = time.perf_counter(); run(f["image"]); torch.cuda.synchronize(); times.append(1000 * (time.perf_counter() - img_t0))
with open(log, "a") as fh:
    fh.write(json.dumps({"stage": "Маски динамики YOLO11x-seg", "device": "GPU", "start": start, "end": time.time(), "times_ms": times,
                         "torch_peak_mb": torch.cuda.max_memory_allocated() / 2**20}) + "\n")
print("yolo готово", len(times))
