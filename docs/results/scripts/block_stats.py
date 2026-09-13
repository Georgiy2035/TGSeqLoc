"""Статистика по участкам маршрута (кластерам) вместо фолдов: RobotCar, кросс-валидация 5×5, 128/64."""
import json, math, random
from pathlib import Path

H = Path.home()
OUT = H / "TGSeqLoc/outputs"
D = H / "tgseqloc_data"
ARMS = {"без текста": "cv-notext-f{f}-s{s}", "узел": "cv-real_qwen-f{f}-s{s}",
        "слагаемое": "add-rc-p0-f{f}-s{s}", "слагаемое + dropout": "add-rc-p05-f{f}-s{s}",
        "перемешанный (узел)": "cv-shuffled_qwen-f{f}-s{s}"}
KS = (1, 5, 10)

# участки: та же нарезка, что при построении фолдов (200 м пройденного пути, скачки ≥ 50 м не считаются)
ref = json.load(open(D / "rcg_notext/robotcar/mappings/split_fold0.json"))
pos = {int(k): v for k, v in ref["query_positions"].items()}
blocks, cur, travelled, prev = [], [], 0.0, None
for i in sorted(pos):
    p = pos[i]
    if prev is not None:
        step = math.dist(p, prev)
        travelled += step if step < 50 else 0.0
    if travelled >= 200 and cur:
        blocks.append(cur); cur = []; travelled = 0.0
    cur.append(i); prev = p
if cur:
    blocks.append(cur)
block_of = {i: b for b, idx in enumerate(blocks) for i in idx}
print("участков: %d" % len(blocks))

# попадания каждого тестового запроса, по веткам и сидам
hits = {a: {} for a in ARMS}         # hits[arm][(block)] -> list of per-(query,seed) tuples
fold_of_block = {}
for f in range(5):
    split = json.load(open(D / f"cv_runs/split_notext_f{f}.json"))
    positives = {int(k): set(v) for k, v in split["positives"].items()}
    for i in split["test_query_indices"]:
        b = block_of[i]
        if fold_of_block.setdefault(b, f) != f:
            raise SystemExit("участок %d попал в тест двух фолдов" % b)
    for arm, pat in ARMS.items():
        for s in range(42, 47):
            r = json.load(open(OUT / pat.format(f=f, s=s) / "test_retrievals.json"))
            for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
                pos_q = positives.get(q, ())
                hits[arm].setdefault(block_of[q], []).append(tuple(int(any(c in pos_q for c in near[:k])) for k in KS))
print("участков в тесте: %d, каждый ровно в одном фолде — проверено" % len(fold_of_block))

blocks_used = sorted(fold_of_block)
n_q = {b: len(hits["без текста"][b]) for b in blocks_used}      # (запрос × сид)
rate = {a: {b: [100 * sum(h[j] for h in hits[a][b]) / len(hits[a][b]) for j in range(3)] for b in blocks_used} for a in ARMS}


def weighted(values, weights):
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


rnd = random.Random(0)
BOOT = 10000


def boot_ci(stat):
    vals = []
    for _ in range(BOOT):
        sample = [rnd.choice(blocks_used) for _ in blocks_used]
        vals.append(stat(sample))
    vals.sort()
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT)]


print("\n=== абсолютные значения (взвешено по запросам; 95 %% ДИ — бутстрэп по %d участкам) ===" % len(blocks_used))
print("%-22s %-24s %-24s %-24s" % ("ветка", "R@1", "R@5", "R@10"))
for a in ARMS:
    cells = []
    for j in range(3):
        stat = lambda sample, j=j: weighted([rate[a][b][j] for b in sample], [n_q[b] for b in sample])
        lo, hi = boot_ci(stat)
        cells.append("%5.2f [%5.2f; %5.2f]" % (stat(blocks_used), lo, hi))
    print("%-22s %s" % (a, "  ".join(cells)))

print("\n=== парные разницы по участкам ===")
print("%-40s %-6s %-30s %-10s %s" % ("сравнение", "", "Δ [95 % ДИ]", "p", "участков +/−/0"))
for x, y in (("слагаемое", "узел"), ("слагаемое", "без текста"), ("слагаемое + dropout", "слагаемое"),
             ("узел", "перемешанный (узел)"), ("без текста", "перемешанный (узел)"), ("узел", "без текста")):
    for j, k in enumerate(KS):
        d = {b: rate[x][b][j] - rate[y][b][j] for b in blocks_used}
        stat = lambda sample: weighted([d[b] for b in sample], [n_q[b] for b in sample])
        obs = stat(blocks_used)
        lo, hi = boot_ci(stat)
        w = [n_q[b] for b in blocks_used]; dv = [d[b] for b in blocks_used]
        trials = 20000
        extreme = sum(1 for _ in range(trials)
                      if abs(sum(v * ww * (1 if rnd.random() < 0.5 else -1) for v, ww in zip(dv, w)) / sum(w)) >= abs(obs) - 1e-12)
        pos_n = sum(1 for v in dv if v > 1e-9); neg_n = sum(1 for v in dv if v < -1e-9)
        label = ("%s − %s" % (x, y)) if j == 0 else ""
        print("%-40s R@%-3d %+6.2f [%+6.2f; %+6.2f]      %.4f    %d/%d/%d" % (label, k, obs, lo, hi, extreme / trials, pos_n, neg_n, len(dv) - pos_n - neg_n))
print("\np — перестановочный знаковый тест по участкам (веса — число запросов), 20 000 жеребьёвок")
