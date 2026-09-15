"""Сид 42, 5 фолдов: тройки с любой камеры (final4) против троек внутри камеры (final4s), слагаемое и без текста.
Метрика а — правильный ответ любой камеры ≤ 25 м; метрика б — только кадр той же камеры ≤ 25 м (по 100 сохранённым соседям).
Статистика по участкам маршрута, как в final4_table.py."""
import bisect, json, math, random
from collections import defaultdict
from pathlib import Path

H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
KS = (1, 5, 10, 25, 100)
rnd = random.Random(0)
cam_of = lambda p: Path(p).stem.split("-", 1)[0] if "-" in Path(p).stem else "stereo_centre"
ts_of = lambda p: int(Path(p).stem.split("-")[-1])
ref = json.load(open(D / "rcg4_m2f_add_paddle/robotcar/mappings/split_fold0.json"))
qpaths, dpaths = ref["query_paths"], ref["database_paths"]
dcam = [cam_of(p) for p in dpaths]
qpos = {int(k): v for k, v in ref["query_positions"].items()}
front = sorted((i for i in range(len(qpaths)) if cam_of(qpaths[i]) == "stereo_centre" and i in qpos), key=lambda i: ts_of(qpaths[i]))
blk, b, trav, prev = {}, 0, 0.0, None
for i in front:
    if prev is not None:
        st = math.dist(qpos[i], prev); trav += st if st < 50 else 0.0
    if trav >= 200:
        b += 1; trav = 0.0
    blk[i] = b; prev = qpos[i]
fts = [ts_of(qpaths[i]) for i in front]
def block_of(i):
    t = ts_of(qpaths[i]); k = bisect.bisect_left(fts, t)
    return blk[front[min((j for j in (k - 1, k) if 0 <= j < len(front)), key=lambda j: abs(fts[j] - t))]]

ARMS = {"любая камера / слагаемое": ("final4", "add", "rcg4_m2f_add_paddle"), "любая камера / без текста": ("final4", "notext", "rcg4_m2f_notext"),
        "та же камера / слагаемое": ("final4s", "add", "rcg4_m2f_add_paddle"), "та же камера / без текста": ("final4s", "notext", "rcg4_m2f_notext")}
data = {m: {a: defaultdict(list) for a in ARMS} for m in ("а", "б")}
for a, (pref, key, root) in ARMS.items():
    for f in range(5):
        split = json.load(open(D / root / f"robotcar/mappings/split_fold{f}.json"))
        pos = {int(k): set(v) for k, v in split["positives"].items()}
        r = json.load(open(OUT / f"{pref}-{key}-f{f}-s42" / "test_retrievals.json"))
        for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
            pq = pos.get(q, set()); qc = cam_of(qpaths[q]); pq_same = {j for j in pq if dcam[j] == qc}
            data["а"][a][block_of(q)].append(tuple(int(any(c in pq for c in near[:k])) for k in KS))
            if pq_same:
                data["б"][a][block_of(q)].append(tuple(int(any(c in pq_same for c in near[:k])) for k in KS))

for m, title in (("а", "метрика а: верный ответ — любая камера ≤ 25 м"), ("б", "метрика б: верный ответ — та же камера ≤ 25 м")):
    print(f"\n=== {title}; сид 42, 5 фолдов ===")
    print("%-28s " % "вариант" + "  ".join("%-21s" % f"R@{k}" for k in KS))
    rates = {}
    for a in ARMS:
        blocks = sorted(data[m][a]); n = {x: len(data[m][a][x]) for x in blocks}
        val = {x: [100 * sum(h[j] for h in data[m][a][x]) / n[x] for j in range(len(KS))] for x in blocks}
        rates[a] = (blocks, n, val)
        cells = []
        for j in range(len(KS)):
            W = lambda smp: sum(val[x][j] * n[x] for x in smp) / sum(n[x] for x in smp)
            bs = sorted(W([rnd.choice(blocks) for _ in blocks]) for _ in range(2000))
            cells.append("%5.2f [%5.2f; %5.2f]" % (W(blocks), bs[50], bs[1949]))
        print("%-28s " % a + "  ".join("%-21s" % c for c in cells))
    for x, y in (("любая камера / слагаемое", "любая камера / без текста"), ("та же камера / слагаемое", "та же камера / без текста"),
                 ("та же камера / слагаемое", "любая камера / слагаемое"), ("та же камера / без текста", "любая камера / без текста")):
        bx, nx, vx = rates[x]; by, ny, vy = rates[y]; blocks = sorted(set(bx) & set(by))
        parts = []
        for j, k in enumerate(KS):
            d = {z: vx[z][j] - vy[z][j] for z in blocks}; w = nx
            obs = sum(d[z] * w[z] for z in blocks) / sum(w[z] for z in blocks)
            ext = sum(1 for _ in range(5000) if abs(sum(d[z] * w[z] * (1 if rnd.random() < .5 else -1) for z in blocks) / sum(w[z] for z in blocks)) >= abs(obs) - 1e-12) / 5000
            parts.append("R@%d %+.2f (p=%.3f)" % (k, obs, ext))
        print("  %s − %s: %s" % (x, y, "; ".join(parts)))
