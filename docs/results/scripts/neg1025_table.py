"""Выбор негативов при 10/25, GAT 1 слой: RobotCar, 5 фолдов, статистика по участкам маршрута."""
import json, math, random, sys
from pathlib import Path

H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
KS = (1, 5, 10)
SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42"])]
rnd = random.Random(0)


def blocks_of_route():
    ref = json.load(open(D / "rcg_notext/robotcar/mappings/split_fold0.json"))
    pos = {int(k): v for k, v in ref["query_positions"].items()}
    blocks, cur, travelled, prev = [], [], 0.0, None
    for i in sorted(pos):
        p = pos[i]
        if prev is not None:
            step = math.dist(p, prev); travelled += step if step < 50 else 0.0
        if travelled >= 200 and cur:
            blocks.append(cur); cur = []; travelled = 0.0
        cur.append(i); prev = p
    if cur:
        blocks.append(cur)
    return {i: b for b, idx in enumerate(blocks) for i in idx}


ARMS = {f"{a} / {m}": f"neg1025-{a}-{m}-f{{f}}-s{{s}}" for a in ("notext", "add", "shufadd") for m in ("hard", "warm")}
block_of = blocks_of_route()
hits, epochs = {a: {} for a in ARMS}, {a: [] for a in ARMS}
for f in range(5):
    split = json.load(open(D / f"cv_runs/split_notext_f{f}.json"))
    positives = {int(k): set(v) for k, v in split["positives"].items()}
    for a, pat in ARMS.items():
        for s in SEEDS:
            run = OUT / pat.format(f=f, s=s)
            p = run / "test_retrievals.json"
            if not p.exists():
                continue
            fm = json.load(open(run / "final_metrics.json")); epochs[a].append(fm.get("best_epoch"))
            r = json.load(open(p))
            for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
                pq = positives.get(q, ())
                hits[a].setdefault(block_of[q], {}).setdefault((f, s), []).append(tuple(int(any(c in pq for c in near[:k])) for k in KS))
need = 5 * len(SEEDS)
print(f"=== RobotCar, 10/25, GAT 1 слой, сиды {SEEDS}; статистика по участкам маршрута ===")
done = {a: len({fs for v in hits[a].values() for fs in v}) for a in ARMS}
for a in ARMS:
    print("  %-18s прогонов %d/%d  лучшие эпохи %s" % (a, done[a], need, epochs[a]))
ok = [a for a in ARMS if done[a] == need]
if not ok:
    sys.exit()
blocks = sorted(set.intersection(*[set(hits[a]) for a in ok]))
n_q = {b: sum(len(v) for v in hits[ok[0]][b].values()) for b in blocks}
rate = {a: {b: [100 * sum(h[j] for v in hits[a][b].values() for h in v) / n_q[b] for j in range(3)] for b in blocks} for a in ok}
W = lambda vals, smp: sum(vals[b] * n_q[b] for b in smp) / sum(n_q[b] for b in smp)
def boot(fn, B=5000):
    vals = sorted(fn([rnd.choice(blocks) for _ in blocks]) for _ in range(B)); return vals[int(.025 * B)], vals[int(.975 * B)]
print("\n%-18s %-24s %-24s %-24s" % ("вариант", "R@1", "R@5", "R@10"))
for a in ok:
    cells = []
    for j in range(3):
        vals = {b: rate[a][b][j] for b in blocks}; lo, hi = boot(lambda smp: W(vals, smp))
        cells.append("%5.2f [%5.2f; %5.2f]" % (W(vals, blocks), lo, hi))
    print("%-18s %s" % (a, "  ".join(cells)))
print("\nпарные разницы по участкам (%d):" % len(blocks))
for x, y in (("notext / warm", "notext / hard"), ("add / warm", "add / hard"), ("shufadd / warm", "shufadd / hard"),
             ("add / hard", "notext / hard"), ("add / warm", "notext / warm"), ("add / hard", "shufadd / hard"), ("add / warm", "shufadd / warm")):
    if x not in ok or y not in ok:
        continue
    for j, k in enumerate(KS):
        d = {b: rate[x][b][j] - rate[y][b][j] for b in blocks}; obs = W(d, blocks); lo, hi = boot(lambda smp: W(d, smp))
        w = [n_q[b] for b in blocks]; dv = [d[b] for b in blocks]
        ext = sum(1 for _ in range(20000) if abs(sum(v * ww * (1 if rnd.random() < .5 else -1) for v, ww in zip(dv, w)) / sum(w)) >= abs(obs) - 1e-12)
        print("  %-36s R@%-3d %+6.2f [%+6.2f; %+6.2f]  p=%.4f  участков %d/%d" % (("%s − %s" % (x, y)) if j == 0 else "", k, obs, lo, hi, ext / 20000,
              sum(1 for v in dv if v > 1e-9), sum(1 for v in dv if v < -1e-9)))
