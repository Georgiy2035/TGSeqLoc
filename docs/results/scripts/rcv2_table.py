"""RobotCar на новом разбиении: валидация — отдельный сегмент, отделённый теми же 25 м."""
import glob, itertools, json, math, os
from pathlib import Path
import torch

H = Path.home()
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def load(pattern, key=lambda name: name):
    out = {}
    for f in glob.glob(str(pattern)):
        txt = open(f, encoding="utf-8", errors="replace").read()
        i = txt.find("{")
        try:
            obj, _ = json.JSONDecoder().raw_decode(txt[i:])
        except Exception:
            continue
        m = obj.get("final_metrics") or obj.get("training", {}).get("final_metrics")
        if m:
            out[int(os.path.basename(f)[:-5].rsplit("_", 1)[1])] = (m["R@1"], m["R@5"], m["R@10"])
    return out


NAMES = [
    ("без текста", "rcg_notext"),
    ("PaddleOCR, фильтр", "rcgf_real_paddle"),
    ("Qwen, фильтр", "rcgf_real_qwen"),
    ("EasyOCR, фильтр", "rcgf_real_easyocr"),
    ("перемешанный Paddle, фильтр", "rcgf_shuffled_paddle"),
    ("перемешанный Qwen, фильтр", "rcgf_shuffled_qwen"),
    ("PaddleOCR, без фильтра", "rcg_real_paddle"),
    ("Qwen, без фильтра", "rcg_real_qwen"),
    ("EasyOCR, без фильтра", "rcg_real_easyocr"),
    ("перемешанный Paddle, без фильтра", "rcg_shuffled_paddle"),
]
rows = {name: load(H / f"tgseqloc_data/rcv2_runs/{key}_*.json") for name, key in NAMES}

# старое разбиение — для справки; тестовый набор там другой, сравнение только качественное
old = {"без текста": load(H / "tgseqloc_data/rcg_runs_filter_off/notext_*.json"),
       "PaddleOCR, фильтр": load(H / "tgseqloc_data/rcgf_runs/real_paddle_*.json"),
       "Qwen, фильтр": load(H / "tgseqloc_data/rcgf_runs/real_qwen_*.json"),
       "EasyOCR, фильтр": load(H / "tgseqloc_data/rcgf_runs/real_easyocr_*.json"),
       "перемешанный Paddle, фильтр": load(H / "tgseqloc_data/rcgf_runs/shuffled_paddle_*.json"),
       "перемешанный Qwen, фильтр": load(H / "tgseqloc_data/rcgf_runs/shuffled_qwen_*.json"),
       "PaddleOCR, без фильтра": load(H / "tgseqloc_data/rcg_runs_filter_off/real_paddle_*.json"),
       "Qwen, без фильтра": load(H / "tgseqloc_data/rcg_runs_filter_off/real_qwen_*.json"),
       "EasyOCR, без фильтра": load(H / "tgseqloc_data/rcg_runs_filter_off/real_easyocr_*.json"),
       "перемешанный Paddle, без фильтра": load(H / "tgseqloc_data/rcg_runs_filter_off/shuffled_paddle_*.json")}


def ci(v):
    n = len(v); mu = sum(v) / n
    s = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return "%5.2f ± %5.2f" % (mu, T.get(n, 2.262) * s / math.sqrt(n) if n > 1 else 0.0)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]; n = len(d); o = sum(d) / n
    hits = sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(si * x for si, x in zip(s, d)) / n) >= abs(o) - 1e-12)
    return o, hits / 2 ** n, sum(1 for x in d if x > 0), n


split = json.load(open(H / "tgseqloc_data/rcgf_real_paddle/robotcar/mappings/temporal_split.json"))
print("=== RobotCar, новое разбиение: обучение %d / валидация %d / тест %d ===" % (
    len(split["train_query_indices"]), len(split["validation_query_indices"]), len(split["test_query_indices"])))
print("%-34s %3s %-15s %-15s %-15s   %s" % ("ветка", "n", "R@1", "R@5", "R@10", "R@5 на старом разбиении"))
for name, _ in NAMES:
    r = rows[name]
    if not r:
        print("%-34s  —" % name); continue
    s = sorted(r)
    o = old.get(name, {})
    ref = "%5.2f" % mean([o[x][1] for x in sorted(o)]) if o else "—"
    print("%-34s %3d %s  %s  %s   %s" % (name, len(s), *(ci([r[x][k] for x in s]) for k in range(3)), ref))

print("\nпарные сравнения (точный знаковый тест, одни и те же сиды):")
pairs = [("PaddleOCR, фильтр", "без текста"), ("Qwen, фильтр", "без текста"), ("EasyOCR, фильтр", "без текста"),
         ("PaddleOCR, фильтр", "перемешанный Paddle, фильтр"), ("Qwen, фильтр", "перемешанный Qwen, фильтр"),
         ("PaddleOCR, фильтр", "PaddleOCR, без фильтра"), ("Qwen, фильтр", "Qwen, без фильтра"),
         ("EasyOCR, фильтр", "EasyOCR, без фильтра"), ("Qwen, фильтр", "PaddleOCR, фильтр"),
         ("PaddleOCR, без фильтра", "без текста"), ("PaddleOCR, без фильтра", "перемешанный Paddle, без фильтра")]
for x, y in pairs:
    s = sorted(set(rows.get(x, {})) & set(rows.get(y, {})))
    if len(s) < 2:
        continue
    res = [paired([rows[x][i][k] for i in s], [rows[y][i][k] for i in s]) for k in range(3)]
    print("  %-30s − %-30s R@5 Δ=%+6.2f p=%.3f %d/%d | R@1 Δ=%+6.2f p=%.3f | R@10 Δ=%+6.2f p=%.3f"
          % (x, y, res[1][0], res[1][1], res[1][2], res[1][3], res[0][0], res[0][1], res[2][0], res[2][1]))


def breakdown(arm_root, prefix):
    root = H / f"tgseqloc_data/{arm_root}/robotcar"
    sp = json.load(open(root / "mappings/temporal_split.json"))
    pos = {int(k): set(v) for k, v in sp["positives"].items()}
    has = {qi: bool(torch.load(root / sp["query_paths"][qi], weights_only=False).is_text.any())
           for qi in sp["test_query_indices"]}

    def r5(pfx, want):
        vals = []
        for seed in range(42, 52):
            p = H / f"TGSeqLoc/outputs/{pfx}-s{seed}/test_retrievals.json"
            if not p.exists():
                continue
            r = json.load(open(p)); hits = tot = 0
            for qi, near in zip(r["query_indices"], r["nearest_database_indices"]):
                if has.get(qi) != want:
                    continue
                tot += 1; hits += any(d in pos.get(qi, ()) for d in near[:5])
            if tot:
                vals.append(100 * hits / tot)
        return mean(vals)

    return sum(has.values()), len(has), r5(prefix, True), r5("v2-rcg-notext", True), r5(prefix, False), r5("v2-rcg-notext", False)


print("\nR@5 на тестовых запросах с текстом и без (по графам своего распознавателя):")
for arm_root, prefix, label in (("rcgf_real_paddle", "v2-rcgf-real_paddle", "PaddleOCR"),
                                ("rcgf_real_qwen", "v2-rcgf-real_qwen", "Qwen"),
                                ("rcgf_real_easyocr", "v2-rcgf-real_easyocr", "EasyOCR")):
    try:
        k, n, a, b, c, d = breakdown(arm_root, prefix)
        print("  %-9s запросов с текстом %3d из %d: с текстом %.1f против %.1f без текста | остальные %.1f против %.1f"
              % (label, k, n, a, b, c, d))
    except Exception as e:
        print("  %-9s не посчитано: %s %s" % (label, type(e).__name__, e))
