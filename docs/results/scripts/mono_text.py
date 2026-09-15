"""Сколько текста дают боковые и задняя камеры RobotCar против фронтальной, без обучения.

Кадры: ~7 000 фронтальных кадров графов и ближайшие к ним кадры mono-камер (разрыв ≤ 0.5 с).
OCR одинаковый для всех камер — PaddleOCR v5; фронтальный Qwen3-VL (им сейчас обучаемся) — для ориентира.
Фильтр как в пайплайне: длина ≥ 3, не только цифры. Масок динамики на mono нет, поэтому главный показатель —
повторяемость: совпадает ли строка запроса (проезд 09-02) со строкой базы (08-12) в радиусе 25 м.
Совпадение: нормализованная строка (регистр, только буквы и цифры), точно или с отношением Левенштейна ≥ 0.8.
"""
import csv, json, math, re, random
from difflib import SequenceMatcher
from pathlib import Path

H = Path.home()
SPLIT = json.load(open(H / "tgseqloc_data/rcg_notext/robotcar/mappings/split_fold0.json"))
MAP = list(csv.DictReader(open("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr/frames_for_graphs.csv")))
MONO = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr")
FRONT = H / "ocr_benchmark/results"
TRAV = {"base": "2015-08-12-15-04-18", "query": "2015-09-02-10-37-32"}
row_of = {r["stereo_centre"]: r for r in MAP}


def norm(t):
    return re.sub(r"[^0-9a-zа-яё]", "", t.casefold())


def strings(path):
    try:
        preds = json.load(open(path)).get("predictions") or []
    except (OSError, ValueError):
        return None
    out = []
    for p in preds:
        n = norm(p.get("text", ""))
        if len(n) >= 3 and not n.isdigit():
            out.append(n)
    return out


def frame_strings(stem, trav, source):
    if source == "front_paddle":
        return strings(FRONT / f"robotcar_{trav}/{stem}/paddleocr_v5.json")
    if source == "front_qwen":
        return strings(FRONT / f"robotcar_{trav}/{stem}/qwen3_vl_4b.json")
    r = row_of.get(stem)
    if r is None or float(r[source + "_dt_s"]) > 0.5:
        return None
    return strings(MONO / f"robotcar_{trav}_{source}/{r[source]}/paddleocr_v5.json")


SOURCES = ["front_qwen", "front_paddle", "mono_left", "mono_right", "mono_rear"]
db_stems = [Path(p).stem for p in SPLIT["database_paths"]]
q_stems = [Path(p).stem for p in SPLIT["query_paths"]]
positives = {int(k): v for k, v in SPLIT["positives"].items()}
qpos = {int(k): v for k, v in SPLIT["query_positions"].items()}
dpos = {int(k): v for k, v in SPLIT["database_positions"].items()}

data = {}
for src in SOURCES:
    data[src] = ({i: frame_strings(s, TRAV["base"], src) for i, s in enumerate(db_stems)},
                 {i: frame_strings(s, TRAV["query"], src) for i, s in enumerate(q_stems)})
combos = {"фронт (Paddle)": ["front_paddle"], "бока (L+R)": ["mono_left", "mono_right"],
          "фронт + бока": ["front_paddle", "mono_left", "mono_right"],
          "все 4 (Paddle)": ["front_paddle", "mono_left", "mono_right", "mono_rear"]}
for name, parts in combos.items():
    db = {i: (sum((data[p][0][i] or [] for p in parts), []) if all(data[p][0][i] is not None for p in parts) else None) for i in range(len(db_stems))}
    q = {i: (sum((data[p][1][i] or [] for p in parts), []) if all(data[p][1][i] is not None for p in parts) else None) for i in range(len(q_stems))}
    data[name] = (db, q)

cache = {}
def match(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.25:
        return False
    k = (a, b) if a < b else (b, a)
    if k not in cache:
        cache[k] = SequenceMatcher(None, a, b).ratio() >= 0.8
    return cache[k]


def shares(qs, ds):
    return any(match(x, y) for x in set(qs) for y in set(ds))


rng = random.Random(0)
# общие кадры: все источники доступны (иначе камеры сравнивались бы на разных кадрах)
common_q = [i for i in range(len(q_stems)) if all(data[s][1][i] is not None for s in data)]
common_d = [i for i in range(len(db_stems)) if all(data[s][0][i] is not None for s in data)]
cd = set(common_d)
print(f"общих кадров: база {len(common_d)} из {len(db_stems)}, запросов {len(common_q)} из {len(q_stems)}\n")
print("%-16s %8s %8s %9s %8s | %9s %11s %12s %11s" % ("источник", "кадров с", "строк на", "уник.", "текст у", "запросов с", "из них: есть", "доля всех", "ложные: >100 м"))
print("%-16s %8s %8s %9s %8s | %9s %11s %12s %11s" % ("", "текстом", "кадр", "строк", "позитив.", "текстом", "совпадение", "запросов", "совпадения"))
for src in data:
    db, q = data[src]
    fr = [db[i] for i in common_d] + [q[i] for i in common_q]
    with_text = sum(1 for x in fr if x)
    n_str = sum(len(x) for x in fr)
    uniq = len({s for x in fr for s in x})
    qt = [i for i in common_q if q[i]]
    hit = 0; false_hit = 0; pos_text = 0
    for i in qt:
        pos = [j for j in positives.get(i, []) if j in cd]
        if any(db[j] for j in pos):
            pos_text += 1
        if any(db[j] and shares(q[i], db[j]) for j in pos):
            hit += 1
        # ложные совпадения: столько же случайных кадров базы дальше 100 м
        far = [j for j in rng.sample(common_d, min(len(common_d), 60)) if math.dist(qpos[i], dpos[j]) > 100 and db[j]]
        if any(shares(q[i], db[j]) for j in far[:max(1, len(pos))]):
            false_hit += 1
    print("%-16s %7.1f%% %8.2f %9d %7.1f%% | %8.1f%% %10.1f%% %11.1f%% %10.1f%%" % (
        src, 100 * with_text / len(fr), n_str / len(fr), uniq, 100 * pos_text / max(1, len(qt)),
        100 * len(qt) / len(common_q), 100 * hit / max(1, len(qt)), 100 * hit / len(common_q), 100 * false_hit / max(1, len(qt))))
print("\nколонки: «текст у позитив.» — доля запросов с текстом, у которых хоть один правильный кадр базы (≤25 м) тоже с текстом;")
print("«есть совпадение» — среди запросов с текстом доля, где строка совпала со строкой правильного кадра базы;")
print("«доля всех запросов» — то же относительно всех запросов: потолок того, скольким запросам текст может помочь напрямую;")
print("«ложные» — доля запросов с текстом, у которых строка совпала со случайным кадром базы дальше 100 м (столько же кадров, сколько позитивов).")
