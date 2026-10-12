"""Динамический текст на Первомайской: IoA строк трёх распознавателей с масками YOLO11x-seg.

Кадры — каждый 10-й кадр двух прогулок (27-08-26, 28-08-26), 1280×720, те же, что прошли OCR.
IoA = доля площади рамки строки под объединением масок динамических классов, порог 0.5 (как на RobotCar).
Фильтр мусора как в пайплайне: после нормализации длина ≥ 3 и не только цифры.
Выход: JSONL на слово (прогулка, кадр, модель, текст, рамка, IoA, класс, dynamic, junk) — готовый фильтр.
"""
import json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

OCR = Path("/mnt/external_usb_hdd/6YL/Datasets/pervomay_ocr")
SEG = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/pervomay_seg")
OUT = Path("/mnt/external_usb_hdd/6YL/Datasets/pervomay_ocr/text_dynamic_yolo11x.jsonl")
MODELS = ("paddleocr_v5", "easyocr", "qwen3_vl_4b")
TAU = 0.5
norm = lambda t: re.sub(r"[^0-9a-zа-яё]", "", t.casefold())
bins = [(0, 0), (0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0001)]

frames = Counter(); frames_dyn = Counter(); missing = Counter()
stat = defaultdict(Counter); classes = defaultdict(Counter); hist = defaultdict(Counter)
length = defaultdict(lambda: {True: [], False: []}); per_frame = defaultdict(lambda: [0, 0])
with open(OUT, "w") as out:
    for walk in ("27-08-26", "28-08-26"):
        for frame_dir in sorted((OCR / walk).iterdir()):
            if not frame_dir.is_dir():
                continue
            seg_p = SEG / walk / frame_dir.name / "yolo11x_seg.json"
            if not seg_p.exists():
                missing[walk] += 1
                continue
            seg = json.loads(seg_p.read_text())
            h, w = seg["image_height"], seg["image_width"]
            frames[walk] += 1
            frames_dyn[walk] += int(bool(seg["instances"]))
            masks, union = [], None
            for inst in seg["instances"]:
                m = mask_utils.decode({"size": inst["rle"]["size"], "counts": inst["rle"]["counts"].encode("ascii")}).astype(bool)
                masks.append((inst["label"], m))
                union = m.copy() if union is None else (union | m)
            for model in MODELS:
                p = frame_dir / f"{model}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                sx, sy = w / float(d.get("image_width") or w), h / float(d.get("image_height") or h)
                static_in_frame = 0
                for pred in d.get("predictions") or []:
                    x, y, bw, bh = pred["bbox"]
                    x1, y1 = max(0, int(round(x * sx))), max(0, int(round(y * sy)))
                    x2, y2 = min(w, int(round((x + bw) * sx))), min(h, int(round((y + bh) * sy)))
                    area = max(1, (x2 - x1) * (y2 - y1))
                    ioa = float(union[y1:y2, x1:x2].sum()) / area if union is not None and x2 > x1 and y2 > y1 else 0.0
                    cls, best = None, 0.0
                    if ioa > 0:
                        for label, m in masks:
                            ov = float(m[y1:y2, x1:x2].sum()) / area
                            if ov > best:
                                cls, best = label, ov
                    n = norm(pred.get("text", ""))
                    junk = len(n) < 3 or n.isdigit()
                    dyn = ioa >= TAU
                    out.write(json.dumps({"walk": walk, "frame": frame_dir.name, "model": model, "text": pred.get("text", ""),
                                          "bbox": pred["bbox"], "ioa": round(ioa, 4), "class": cls, "dynamic": dyn, "junk": junk},
                                         ensure_ascii=False) + "\n")
                    stat[model]["слов"] += 1
                    if junk:
                        continue
                    stat[model]["после мусора"] += 1
                    stat[model]["динамических"] += int(dyn)
                    if dyn:
                        classes[model][cls] += 1
                    else:
                        static_in_frame += 1
                    length[model][dyn].append(len(n))
                    for lo, hi in bins:
                        if (lo == hi == 0 and ioa == 0) or (hi > 0 and lo <= ioa < hi):
                            hist[model][(lo, hi)] += 1
                            break
                per_frame[model][0] += int(static_in_frame > 0)
                per_frame[model][1] += 1

total = sum(frames.values())
print("кадров с масками: %s (всего %d); кадров с динамикой: %s; без масок: %s" % (
    dict(frames), total, {k: "%.1f%%" % (100 * frames_dyn[k] / frames[k]) for k in frames}, dict(missing)))
print("\n=== Первомайская: динамический текст (YOLO11x-seg, IoA ≥ 0.5), после фильтра мусора ===")
print("%-14s %7s %8s %13s %10s %20s %14s" % ("модель", "слов", "после", "динамических", "статичных", "кадров со статичным", "длина дин/стат"))
for m in MODELS:
    s = stat[m]; n = s["после мусора"]
    print("%-14s %7d %8d %12.1f%% %10d %19.1f%% %7.1f / %.1f" % (m, s["слов"], n, 100 * s["динамических"] / max(1, n), n - s["динамических"],
          100 * per_frame[m][0] / max(1, per_frame[m][1]), np.mean(length[m][True]) if length[m][True] else 0, np.mean(length[m][False]) if length[m][False] else 0))
print("\nраспределение IoA (доля слов после фильтра мусора):")
print("%-14s %7s %9s %10s %10s %9s" % ("модель", "=0", "(0,0.1)", "[0.1,0.5)", "[0.5,0.9)", "[0.9,1]"))
for m in MODELS:
    n = max(1, stat[m]["после мусора"])
    print("%-14s %6.1f%% %8.1f%% %9.1f%% %9.1f%% %8.1f%%" % (m, *[100 * hist[m][b] / n for b in bins]))
print("\nклассы-носители динамического текста:")
for m in MODELS:
    t = max(1, sum(classes[m].values()))
    print("  %-13s " % m + ", ".join("%s %.1f%%" % (k, 100 * v / t) for k, v in classes[m].most_common(6)))
print("\nзаписано:", OUT)
