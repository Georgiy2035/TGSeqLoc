"""Финальный эксперимент, 4 камеры: R@1/5/10/25/100 по всем запросам и по камерам; статистика по участкам маршрута.

Участки — как во всех таблицах RobotCar: 200 м пройденного пути по фронтальным запросам. Запрос другой камеры
относится к участку фронтального запроса, ближайшего по времени.
"""
import bisect, json, math, random, sys
from collections import defaultdict
from pathlib import Path

H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
KS = (1, 5, 10, 25, 100)
CAMS = ("stereo_centre", "mono_left", "mono_right", "mono_rear")
ARMS = {"слагаемое": ("add", "rcg4_m2f_add_paddle"), "без текста": ("notext", "rcg4_m2f_notext")}
SEEDS = (42, 43, 44)
rnd = random.Random(0)
cam_of = lambda p: Path(p).stem.split("-", 1)[0] if "-" in Path(p).stem else "stereo_centre"
ts_of = lambda p: int(Path(p).stem.split("-")[-1])

# участки маршрута по фронтальным запросам (разбиение одно для всех фолдов по составу запросов и координатам)
ref = json.load(open(D / "rcg4_m2f_add_paddle/robotcar/mappings/split_fold0.json"))
qpaths = ref["query_paths"]; qpos = {int(k): v for k, v in ref["query_positions"].items()}
front = [i for i in range(len(qpaths)) if cam_of(qpaths[i]) == "stereo_centre" and i in qpos]
front.sort(key=lambda i: ts_of(qpaths[i]))
block_front, cur_block, travelled, prev = {}, 0, 0.0, None
for i in front:
    p = qpos[i]
    if prev is not None:
        step = math.dist(p, prev); travelled += step if step < 50 else 0.0
    if travelled >= 200:
        cur_block += 1; travelled = 0.0
    block_front[i] = cur_block; prev = p
front_ts = [ts_of(qpaths[i]) for i in front]
def block_of(i):
    t = ts_of(qpaths[i]); k = bisect.bisect_left(front_ts, t)
    best = min((j for j in (k - 1, k) if 0 <= j < len(front)), key=lambda j: abs(front_ts[j] - t))
    return block_front[front[best]]

hits = {a: defaultdict(lambda: defaultdict(list)) for a in ARMS}   # arm -> block -> cam -> [(hit per K)]
runs = {a: 0 for a in ARMS}; epochs = {a: [] for a in ARMS}
for arm, (key, root) in ARMS.items():
    for f in range(5):
        split = json.load(open(D / root / f"robotcar/mappings/split_fold{f}.json"))
        assert split["query_paths"] == qpaths
        positives = {int(k): set(v) for k, v in split["positives"].items()}
        for s in SEEDS:
            run = OUT / f"final4-{key}-f{f}-s{s}"
            if not (run / "test_retrievals.json").exists():
                continue
            fm = json.load(open(run / "final_metrics.json")); epochs[arm].append(fm.get("best_epoch"))
            r = json.load(open(run / "test_retrievals.json"))
            depth = len(r["nearest_database_indices"][0]); assert depth >= max(KS), depth
            runs[arm] += 1
            for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
                pq = positives.get(q, ())
                hits[arm][block_of(q)][cam_of(qpaths[q])].append(tuple(int(any(c in pq for c in near[:k])) for k in KS))

print("=== RobotCar, 4 камеры, Mask2Former, PaddleOCR, 10/25, GAT 1 слой, прогрев; 5 фолдов × сиды", SEEDS, "===")
for a in ARMS:
    print("  %-11s прогонов %d/15, лучшие эпохи %s" % (a, runs[a], epochs[a]))
ok = [a for a in ARMS if runs[a] == 15]

def table(sel_cams, title):
    print(f"\n{title}")
    print("%-11s " % "вариант" + "  ".join("%-22s" % f"R@{k}" for k in KS))
    rate = {}
    for a in ok:
        blocks = sorted(hits[a])
        n = {b: sum(len(hits[a][b][c]) for c in sel_cams) for b in blocks}
        blocks = [b for b in blocks if n[b]]
        val = {b: [100 * sum(h[j] for c in sel_cams for h in hits[a][b][c]) / n[b] for j in range(len(KS))] for b in blocks}
        rate[a] = (blocks, n, val)
        cells = []
        for j in range(len(KS)):
            W = lambda smp: sum(val[b][j] * n[b] for b in smp) / sum(n[b] for b in smp)
            bs = sorted(W([rnd.choice(blocks) for _ in blocks]) for _ in range(3000))
            cells.append("%5.2f [%5.2f; %5.2f]" % (W(blocks), bs[75], bs[2924]))
        print("%-11s " % a + "  ".join("%-22s" % c for c in cells))
    if len(ok) == 2:
        (bx, nx, vx), (by, ny, vy) = rate[ok[0]], rate[ok[1]]
        blocks = sorted(set(bx) & set(by))
        line = []
        for j, k in enumerate(KS):
            d = {b: vx[b][j] - vy[b][j] for b in blocks}; w = {b: nx[b] for b in blocks}
            obs = sum(d[b] * w[b] for b in blocks) / sum(w.values())
            ext = sum(1 for _ in range(10000) if abs(sum(d[b] * w[b] * (1 if rnd.random() < .5 else -1) for b in blocks) / sum(w.values())) >= abs(obs) - 1e-12) / 10000
            line.append("R@%d %+.2f (p=%.3f)" % (k, obs, ext))
        print("%s − %s: " % (ok[0], ok[1]) + ";  ".join(line))

table(CAMS, "все запросы (среднее по запросам, 95% интервал по участкам маршрута)")
for c in CAMS:
    table((c,), f"запросы камеры {c}")
