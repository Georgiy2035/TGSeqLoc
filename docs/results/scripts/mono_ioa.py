"""IoA строк PaddleOCR с масками динамики YOLO11x-seg для всех четырёх камер RobotCar, кадры графов.

Логика как в dynseg_benchmark/compute_ioa.py (фронтальная камера): IoA = доля площади бокса строки под
объединением масок динамических классов; порог 0.5; класс — экземпляр, накрывающий бокс сильнее всех.
Для фронта берутся маски из dynseg_benchmark/results/robotcar_seg_<проезд>, для mono — robotcar_mono_seg.
Выход: JSONL по строке на слово (камера, проезд, кадр, текст, IoA, класс, динамическое) — для фильтра.
"""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

H = Path.home()
MONO_OCR = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr")
MONO_SEG = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_seg")
OUT = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_text_dynamic_4cams.jsonl")
rows = list(csv.DictReader(open(MONO_OCR / "frames_for_graphs.csv")))
TAU = 0.5


def pairs():
    for r in rows:
        t, s = r["traversal"], r["stereo_centre"]
        yield "front", t, s, H / f"ocr_benchmark/results/robotcar_{t}/{s}/paddleocr_v5.json", H / f"dynseg_benchmark/results/robotcar_seg_{t}/{s}/yolo11x_seg.json"
        for c in ("mono_left", "mono_right", "mono_rear"):
            if float(r[c + "_dt_s"]) > 0.5:
                continue
            m = r[c]
            yield c, t, m, MONO_OCR / f"robotcar_{t}_{c}/{m}/paddleocr_v5.json", MONO_SEG / f"robotcar_seg_{t}_{c}/{m}/yolo11x_seg.json"


seen = set()
stats = defaultdict(Counter)
classes = defaultdict(Counter)
lengths = defaultdict(lambda: {"dyn": [], "stat": []})
conf = defaultdict(lambda: {"dyn": [], "stat": []})
ioa_hist = defaultdict(Counter)
bins = [(0, 0), (0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0)]
with open(OUT, "w") as out:
    for cam, t, stem, ocr_p, seg_p in pairs():
        key = (cam, t, stem)
        if key in seen:
            continue
        seen.add(key)
        if not ocr_p.exists() or not seg_p.exists():
            stats[cam]["кадров без OCR или масок"] += 1
            continue
        stats[cam]["кадров"] += 1
        preds = json.loads(ocr_p.read_text()).get("predictions") or []
        seg = json.loads(seg_p.read_text())
        if seg["instances"]:
            stats[cam]["кадров с динамикой"] += 1
        if not preds:
            continue
        h, w = seg["image_height"], seg["image_width"]
        masks = [(i["label"], mask_utils.decode({"size": i["rle"]["size"], "counts": i["rle"]["counts"].encode("ascii")}).astype(bool)) for i in seg["instances"]]
        union = np.zeros((h, w), bool)
        for _, m in masks:
            union |= m
        for p in preds:
            x, y, bw, bh = p["bbox"]
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            area = max(1, (x2 - x1) * (y2 - y1))
            patch = union[y1:y2, x1:x2]
            ioa = float(patch.sum()) / area if patch.size else 0.0
            dominant, best = None, 0.0
            if ioa > 0:
                for label, m in masks:
                    ov = float(m[y1:y2, x1:x2].sum()) / area
                    if ov > best:
                        dominant, best = label, ov
            dyn = ioa >= TAU
            text = p.get("text", "")
            norm = re.sub(r"[^0-9a-zа-яё]", "", text.casefold())
            junk = len(norm) < 3 or norm.isdigit()
            out.write(json.dumps({"camera": cam, "traversal": t, "frame": stem, "text": text, "ioa": round(ioa, 4),
                                  "class": dominant, "dynamic": dyn, "junk": junk}, ensure_ascii=False) + "\n")
            stats[cam]["слов"] += 1
            if junk:
                stats[cam]["мусор"] += 1
                continue
            stats[cam]["слов после фильтра мусора"] += 1
            stats[cam]["динамических"] += int(dyn)
            if dyn:
                classes[cam][dominant] += 1
            lengths[cam]["dyn" if dyn else "stat"].append(len(norm))
            c = p.get("confidence")
            if c is not None:
                conf[cam]["dyn" if dyn else "stat"].append(float(c))
            for lo, hi in bins:
                if (lo == hi == 0 and ioa == 0) or (hi > 0 and lo <= ioa < hi) or (hi == 1.0 and ioa == 1.0):
                    ioa_hist[cam][(lo, hi)] += 1
                    break

cams = ("front", "mono_left", "mono_right", "mono_rear")
names = {"front": "фронт", "mono_left": "левая", "mono_right": "правая", "mono_rear": "задняя"}
print("\n=== динамический текст по камерам (PaddleOCR v5, YOLO11x-seg, IoA ≥ 0.5; после фильтра мусора) ===")
print("%-8s %7s %13s %7s %9s %13s %11s %13s" % ("камера", "кадров", "с динамикой", "слов", "после", "динамических", "статичных", "длина дин/стат"))
for c in cams:
    s = stats[c]; n = s["слов после фильтра мусора"]
    ld, ls = lengths[c]["dyn"], lengths[c]["stat"]
    print("%-8s %7d %12.1f%% %7d %9d %12.1f%% %11d %6.1f / %.1f" % (
        names[c], s["кадров"], 100 * s["кадров с динамикой"] / max(1, s["кадров"]), s["слов"], n,
        100 * s["динамических"] / max(1, n), n - s["динамических"], np.mean(ld) if ld else 0, np.mean(ls) if ls else 0))
print("\nуверенность OCR, динамические / статичные:", "; ".join(
    "%s %.2f / %.2f" % (names[c], np.mean(conf[c]["dyn"]) if conf[c]["dyn"] else 0, np.mean(conf[c]["stat"]) if conf[c]["stat"] else 0) for c in cams))
print("\nраспределение IoA (доля слов после фильтра мусора):")
print("%-8s %8s %9s %9s %9s %9s" % ("камера", "=0", "(0,0.1)", "[0.1,0.5)", "[0.5,0.9)", "[0.9,1]"))
for c in cams:
    n = max(1, stats[c]["слов после фильтра мусора"])
    print("%-8s %7.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%%" % (names[c], *[100 * ioa_hist[c][b] / n for b in bins]))
print("\nклассы-носители динамического текста:")
for c in cams:
    tot = max(1, sum(classes[c].values()))
    print("  %-7s " % names[c] + ", ".join("%s %.1f%%" % (k, 100 * v / tot) for k, v in classes[c].most_common(6)))
miss = {names[c]: stats[c]["кадров без OCR или масок"] for c in cams if stats[c]["кадров без OCR или масок"]}
if miss:
    print("\nкадров без OCR или масок:", miss)
print("\nзаписано:", OUT)
