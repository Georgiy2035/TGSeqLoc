"""Итог проверки «сцена + набор строк»: RobotCar 256 по участкам маршрута, V4RL 256 парно по сидам."""
import itertools, json, math, random
from pathlib import Path

H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
KS = (1, 5, 10)
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


def robotcar(arms):
    block_of = blocks_of_route()
    hits = {a: {} for a in arms}
    for f in range(5):
        split = json.load(open(D / f"cv_runs/split_notext_f{f}.json"))
        positives = {int(k): set(v) for k, v in split["positives"].items()}
        for a, pat in arms.items():
            for s in range(42, 47):
                p = OUT / pat.format(f=f, s=s) / "test_retrievals.json"
                if not p.exists():
                    continue
                r = json.load(open(p))
                for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
                    pq = positives.get(q, ())
                    hits[a].setdefault(block_of[q], {}).setdefault((f, s), []).append(
                        tuple(int(any(c in pq for c in near[:k])) for k in KS))
    complete = {a for a in arms if sum(len(v) for v in hits[a].values()) == 5 * 0 + sum(len(v) for v in hits[a].values()) and
                len({fs for v in hits[a].values() for fs in v}) == 25}
    print("=== RobotCar, 256/256, кросс-валидация 5×5, статистика по участкам маршрута ===")
    for a in arms:
        n = len({fs for v in hits[a].values() for fs in v})
        print("  %-26s прогонов %d/25%s" % (a, n, "" if n == 25 else "  (неполная, в сравнениях не участвует)"))
    arms_ok = [a for a in arms if a in complete]
    if not arms_ok:
        return
    blocks = sorted(set.intersection(*[set(hits[a]) for a in arms_ok]))
    n_q = {b: sum(len(v) for v in hits[arms_ok[0]][b].values()) for b in blocks}
    rate = {a: {b: [100 * sum(h[j] for v in hits[a][b].values() for h in v) / n_q[b] for j in range(3)] for b in blocks} for a in arms_ok}
    W = lambda vals, sample: sum(vals[b] * n_q[b] for b in sample) / sum(n_q[b] for b in sample)

    def boot(fn, B=5000):
        vals = sorted(fn([rnd.choice(blocks) for _ in blocks]) for _ in range(B))
        return vals[int(.025 * B)], vals[int(.975 * B)]

    print("\n%-26s %-24s %-24s %-24s" % ("ветка", "R@1", "R@5", "R@10"))
    for a in arms_ok:
        cells = []
        for j in range(3):
            vals = {b: rate[a][b][j] for b in blocks}
            lo, hi = boot(lambda smp: W(vals, smp))
            cells.append("%5.2f [%5.2f; %5.2f]" % (W(vals, blocks), lo, hi))
        print("%-26s %s" % (a, "  ".join(cells)))
    return rate, blocks, n_q, W, boot, arms_ok


def compare(ctx, pairs):
    rate, blocks, n_q, W, boot, arms_ok = ctx
    print("\nпарные разницы по участкам (%d участков):" % len(blocks))
    for x, y in pairs:
        if x not in arms_ok or y not in arms_ok:
            continue
        for j, k in enumerate(KS):
            d = {b: rate[x][b][j] - rate[y][b][j] for b in blocks}
            obs = W(d, blocks); lo, hi = boot(lambda smp: W(d, smp))
            w = [n_q[b] for b in blocks]; dv = [d[b] for b in blocks]
            ext = sum(1 for _ in range(20000) if abs(sum(v * ww * (1 if rnd.random() < .5 else -1) for v, ww in zip(dv, w)) / sum(w)) >= abs(obs) - 1e-12)
            print("  %-44s R@%-3d %+6.2f [%+6.2f; %+6.2f]  p=%.4f  участков %d/%d" % (
                ("%s − %s" % (x, y)) if j == 0 else "", k, obs, lo, hi, ext / 20000,
                sum(1 for v in dv if v > 1e-9), sum(1 for v in dv if v < -1e-9)))


ctx = robotcar({"без текста": "f256-notext-f{f}-s{s}", "слагаемое": "f256-add-f{f}-s{s}",
                "набор строк": "f256-set-f{f}-s{s}", "перемешанный набор": "f256-shufset-f{f}-s{s}",
                "без текста + буфер": "f256-notextbuf-f{f}-s{s}", "слагаемое + буфер": "f256-addbuf-f{f}-s{s}",
                "перемешанное слаг. + буфер": "f256-shufaddbuf-f{f}-s{s}",
                "слагаемое + буфер 35": "f256-addbuf35-f{f}-s{s}", "слагаемое + буфер 75": "f256-addbuf75-f{f}-s{s}",
                "слагаемое 10/25": "f256-addp10-f{f}-s{s}", "без текста 10/25": "f256-notextp10-f{f}-s{s}"})
if ctx:
    compare(ctx, [("набор строк", "без текста"), ("набор строк", "слагаемое"), ("набор строк", "перемешанный набор"),
                  ("слагаемое", "без текста"), ("слагаемое + буфер", "слагаемое"), ("без текста + буфер", "без текста"),
                  ("слагаемое + буфер", "без текста + буфер"), ("слагаемое + буфер", "перемешанное слаг. + буфер"),
                  ("слагаемое + буфер 35", "слагаемое"), ("слагаемое + буфер 75", "слагаемое"),
                  ("слагаемое + буфер 75", "слагаемое + буфер"), ("слагаемое + буфер", "слагаемое + буфер 35"),
                  ("слагаемое + буфер 35", "без текста"), ("слагаемое + буфер 75", "без текста"),
                  ("слагаемое 10/25", "слагаемое"), ("без текста 10/25", "без текста"),
                  ("слагаемое 10/25", "без текста 10/25"), ("слагаемое 10/25", "слагаемое + буфер")])


def metrics(p):
    try:
        t = open(p, errors="replace").read(); o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):])
        m = o.get("final_metrics") or o.get("training", {}).get("final_metrics")
        return (m["R@1"], m["R@5"], m["R@10"]) if m else None
    except Exception:
        return None


print("\n=== V4RL, 256/256, 10 сидов ===")
V = {a: {s: m for s in range(42, 52) if (m := metrics(D / f"v256_runs/{a}_{s}.json"))} for a in ("notext", "node", "set")}
names = {"notext": "без текста", "node": "узел", "set": "набор строк"}
for a, r in V.items():
    if r:
        vals = list(r.values()); n = len(vals)
        cells = []
        for j in range(3):
            mu = sum(v[j] for v in vals) / n
            sd = math.sqrt(sum((v[j] - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0
            cells.append("%5.2f ± %5.2f" % (mu, 2.262 * sd / math.sqrt(n) if n > 1 else 0))
        print("  %-12s n=%2d  %s" % (names[a], n, "  ".join(cells)))
for x, y in (("set", "node"), ("set", "notext"), ("node", "notext")):
    keys = sorted(set(V[x]) & set(V[y]))
    if len(keys) < 2:
        continue
    cells = []
    for j, k in enumerate(KS):
        d = [V[x][s][j] - V[y][s][j] for s in keys]; n = len(d); mu = sum(d) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in d) / (n - 1))
        ext = sum(1 for sg in itertools.product((1, -1), repeat=n) if abs(sum(a * b for a, b in zip(sg, d)) / n) >= abs(mu) - 1e-12) / 2 ** n
        cells.append("R@%d Δ=%+5.2f ±%4.2f p=%.3f %d/%d" % (k, mu, 2.262 * sd / math.sqrt(n), ext, sum(1 for v in d if v > 0), n))
    print("  %-26s %s" % ("%s − %s" % (names[x], names[y]), "   ".join(cells)))
print("\nкритерии: RobotCar — набор строк ≥ слагаемого и значимо > без текста, набор > перемешанного;"
      " V4RL — набор не хуже узла более чем на 1 R@5 и значимо > без текста")
