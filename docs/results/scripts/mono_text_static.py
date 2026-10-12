"""То же, что mono_text.py, но только статичный текст (IoA < 0.5), PaddleOCR на всех камерах."""
import csv, json, math, random, re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

H = Path.home()
SPLIT = json.load(open(H / "tgseqloc_data/rcg_notext/robotcar/mappings/split_fold0.json"))
MAP = list(csv.DictReader(open("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr/frames_for_graphs.csv")))
TRAV = {"base": "2015-08-12-15-04-18", "query": "2015-09-02-10-37-32"}
norm = lambda t: re.sub(r"[^0-9a-zа-яё]", "", t.casefold())

words = defaultdict(lambda: {"all": [], "static": []})
for line in open("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_text_dynamic_4cams.jsonl"):
    r = json.loads(line)
    if r["junk"]:
        continue
    k = (r["camera"], r["traversal"], r["frame"])
    n = norm(r["text"])
    words[k]["all"].append(n)
    if not r["dynamic"]:
        words[k]["static"].append(n)

row_of = {r["stereo_centre"]: r for r in MAP}
CAMS = ("front", "mono_left", "mono_right", "mono_rear")


def strings(stem, trav, cam, kind):
    r = row_of.get(stem)
    if r is None:
        return None
    if cam == "front":
        frame = stem
    else:
        if float(r[cam + "_dt_s"]) > 0.5:
            return None
        frame = r[cam]
    return words.get((cam, trav, frame), {"all": [], "static": []})[kind]


db_stems = [Path(p).stem for p in SPLIT["database_paths"]]
q_stems = [Path(p).stem for p in SPLIT["query_paths"]]
positives = {int(k): v for k, v in SPLIT["positives"].items()}
qpos = {int(k): v for k, v in SPLIT["query_positions"].items()}
dpos = {int(k): v for k, v in SPLIT["database_positions"].items()}
combos = {"фронт": ["front"], "левая": ["mono_left"], "правая": ["mono_right"], "задняя": ["mono_rear"],
          "бока (L+R)": ["mono_left", "mono_right"], "фронт + бока": ["front", "mono_left", "mono_right"],
          "все 4": list(CAMS)}
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


def get(stems, trav, parts, kind):
    out = {}
    for i, s in enumerate(stems):
        vals = [strings(s, trav, c, kind) for c in parts]
        out[i] = None if any(v is None for v in vals) else sum(vals, [])
    return out


for kind, title in (("all", "весь текст"), ("static", "только статичный (IoA < 0.5)")):
    print(f"\n=== {title}; PaddleOCR на всех камерах, фильтр мусора ===")
    print("%-14s %9s %8s | %9s %12s %12s %10s" % ("источник", "кадров с", "строк на", "запросов", "совпадение", "доля всех", "ложные"))
    print("%-14s %9s %8s | %9s %12s %12s %10s" % ("", "текстом", "кадр", "с текстом", "с позитивом", "запросов", ">100 м"))
    rng = random.Random(0)
    base = {n: (get(db_stems, TRAV["base"], p, kind), get(q_stems, TRAV["query"], p, kind)) for n, p in combos.items()}
    common_d = [i for i in range(len(db_stems)) if all(base[n][0][i] is not None for n in base)]
    common_q = [i for i in range(len(q_stems)) if all(base[n][1][i] is not None for n in base)]
    cd = set(common_d)
    for name, (db, q) in base.items():
        fr = [db[i] for i in common_d] + [q[i] for i in common_q]
        qt = [i for i in common_q if q[i]]
        hit = false = 0
        for i in qt:
            pos = [j for j in positives.get(i, []) if j in cd]
            qs = set(q[i])
            if any(db[j] and any(match(x, y) for x in qs for y in set(db[j])) for j in pos):
                hit += 1
            far = [j for j in rng.sample(common_d, 60) if math.dist(qpos[i], dpos[j]) > 100 and db[j]]
            if any(any(match(x, y) for x in qs for y in set(db[j])) for j in far[:max(1, len(pos))]):
                false += 1
        print("%-14s %8.1f%% %8.2f | %8.1f%% %11.1f%% %11.1f%% %9.1f%%" % (
            name, 100 * sum(1 for x in fr if x) / len(fr), sum(len(x) for x in fr) / len(fr),
            100 * len(qt) / len(common_q), 100 * hit / max(1, len(qt)), 100 * hit / len(common_q), 100 * false / max(1, len(qt))))
    print(f"(общих кадров: база {len(common_d)}, запросов {len(common_q)})")
