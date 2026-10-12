"""Первомайская без обучения: веса финального RobotCar (4 камеры, 10/25, прогрев, сид 42, фолды 0–4).

Запросы 27-08-26, база 28-08-26. Правильный кадр — ≤ 25 м (основная метрика) и ≤ 10 м (дополнительная), 3D.
Попадание запроса усредняется по 5 моделям-фолдам; разброс — стандартное отклонение R@K между фолдами.
Статистика — по участкам 200 м пройденного пути прогулки-запроса: бутстрэп 95 % ДИ и знаковый перестановочный тест.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

H = Path.home()
D = H / "tgseqloc_data"
RUNS = D / "pmtransfer_runs"
ROOTS = {"add": D / "pm_m2f_paddle", "notext": D / "pm_m2f_notext", "shuf": D / "pm_m2f_shuf_paddle"}
ARMS = {"слагаемое": "add", "без текста": "notext", "текст удалён": "textoff", "перемешанный текст": "shuf"}
WEIGHTS = {"final4": "тройки с любой камеры (final4)", "final4s": "тройки с той же камеры (final4s)"}
KS = (1, 5, 10, 25, 100)
PAIRS = (("слагаемое", "без текста"), ("слагаемое", "текст удалён"), ("слагаемое", "перемешанный текст"))

splits = {arm: json.load(open(root / "pervomay/mappings/temporal_split.json")) for arm, root in ROOTS.items()}
split = splits["add"]
for arm, other in splits.items():
    for key in ("database_paths", "query_paths", "positives", "test_query_indices"):
        assert other[key] == split[key], (arm, key)
qpos = {int(k): v for k, v in split["query_positions"].items()}
dpos = {int(k): v for k, v in split["database_positions"].items()}
pos25 = {int(k): set(v) for k, v in split["positives"].items()}
db_index = sorted(dpos)
tree = cKDTree(np.array([dpos[i] for i in db_index]))
pos10 = {}
for q in sorted(qpos):
    near = tree.query_ball_point(qpos[q], 10.0)
    if near:
        pos10[q] = {db_index[j] for j in near}
test = split["test_query_indices"]
print(f"запросов {len(split['query_paths'])}, база {len(split['database_paths'])}; "
      f"с позитивом ≤25 м {len(test)}, ≤10 м {len(pos10)}; позитивов на запрос (медиана) "
      f"{int(np.median([len(pos25[q]) for q in test]))} / {int(np.median([len(v) for v in pos10.values()]))}")

# текст в графе запроса: строка, привязанная к объекту (только она меняет дескриптор слагаемого)
text_linked, text_nodes = {}, {}
for i, rel in enumerate(split["query_paths"]):
    g = torch.load(ROOTS["add"] / "pervomay" / rel, map_location="cpu", weights_only=False)
    text_nodes[i] = int(g.is_text.sum())
    text_linked[i] = bool(g.is_text_edge.any()) if g.is_text_edge.numel() else False
unknown = total = 0
for rel in split["query_paths"] + split["database_paths"]:
    g = torch.load(ROOTS["notext"] / "pervomay" / rel, map_location="cpu", weights_only=False)
    classes = g.node_class[~g.is_text.bool()]
    unknown += int((classes == 0).sum()); total += int(classes.numel())
print(f"запросов со строкой в графе: {100 * np.mean([text_nodes[q] > 0 for q in test]):.1f} %, "
      f"с привязанной к объекту: {100 * np.mean([text_linked[q] for q in test]):.1f} %; "
      f"узлов-объектов вне словаря RobotCar (unknown): {100 * unknown / total:.1f} % из {total}")

# участки: 200 м пройденного пути, скачки ≥ 50 м не считаются
blocks, cur, travelled, prev = [], [], 0.0, None
for i in sorted(qpos):
    p = qpos[i]
    if prev is not None:
        step = math.dist(p, prev)
        travelled += step if step < 50 else 0.0
    if travelled >= 200 and cur:
        blocks.append(cur); cur = []; travelled = 0.0
    cur.append(i); prev = p
if cur:
    blocks.append(cur)
block_of = {i: b for b, idx in enumerate(blocks) for i in idx}
print(f"участков по 200 м: {len(blocks)}")


def load_hits(weights, evaluation, positives, queries):
    """hits[q][fold][k] и R@K по фолдам."""

    per_fold = []
    hits = {q: [] for q in queries}
    for f in range(5):
        r = json.load(open(RUNS / f"pm-{weights}-{evaluation}-f{f}" / "test_retrievals.json"))
        assert r["query_indices"] == test
        fold = {}
        for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
            if q not in hits:
                continue
            ps = positives[q]
            first = next((rank for rank, c in enumerate(near) if c in ps), None)
            row = [int(first is not None and first < k) for k in KS]
            hits[q].append(row); fold[q] = row
        per_fold.append([100 * np.mean([fold[q][j] for q in queries]) for j in range(len(KS))])
    return {q: np.mean(v, axis=0) for q, v in hits.items()}, np.array(per_fold)


rnd = random.Random(0)
BOOT, TRIALS = 10000, 20000


def block_rates(hits, queries):
    used = sorted({block_of[q] for q in queries})
    members = {b: [q for q in queries if block_of[q] == b] for b in used}
    return used, {b: len(members[b]) for b in used}, {b: np.mean([hits[q] for q in members[b]], axis=0) * 100 for b in used}


def ci(values, weights):
    stats = []
    keys = list(values)
    for _ in range(BOOT):
        sample = [rnd.choice(keys) for _ in keys]
        w = np.array([weights[b] for b in sample], float)
        stats.append((np.array([values[b] for b in sample]) * w[:, None]).sum(0) / w.sum())
    stats = np.sort(np.array(stats), axis=0)
    return stats[int(0.025 * BOOT)], stats[int(0.975 * BOOT)]


def report(title, positives, queries):
    print(f"\n=== {title}: запросов {len(queries)} ===")
    for weights, label in WEIGHTS.items():
        print(f"\n--- веса {label} ---")
        data = {arm: load_hits(weights, ev, positives, queries) for arm, ev in ARMS.items()}
        print("%-20s" % "вариант" + "".join("%-26s" % f"R@{k}" for k in KS) + "sd по фолдам R@1/R@5")
        rates = {}
        for arm, (hits, folds) in data.items():
            used, n_q, rates[arm] = block_rates(hits, queries)
            mean = np.mean([hits[q] for q in queries], axis=0) * 100
            lo, hi = ci(rates[arm], n_q)
            cells = "".join("%-26s" % f"{mean[j]:5.2f} [{lo[j]:5.2f}; {hi[j]:5.2f}]" for j in range(len(KS)))
            print("%-20s%s%.2f / %.2f" % (arm, cells, folds[:, 0].std(ddof=1), folds[:, 1].std(ddof=1)))
        print("по фолдам R@5: " + "; ".join(f"{arm} " + " ".join(f"{v:.1f}" for v in data[arm][1][:, 1]) for arm in ARMS))
        for x, y in PAIRS:
            diffs = {b: rates[x][b] - rates[y][b] for b in used}
            w = np.array([n_q[b] for b in used], float)
            dv = np.array([diffs[b] for b in used])
            obs = (dv * w[:, None]).sum(0) / w.sum()
            lo, hi = ci(diffs, n_q)
            parts = []
            for j, k in enumerate(KS):
                extreme = 0
                for _ in range(TRIALS):
                    signs = np.array([1 if rnd.random() < 0.5 else -1 for _ in used])
                    if abs((dv[:, j] * w * signs).sum() / w.sum()) >= abs(obs[j]) - 1e-12:
                        extreme += 1
                parts.append(f"R@{k} {obs[j]:+.2f} [{lo[j]:+.2f}; {hi[j]:+.2f}] p={extreme / TRIALS:.3f}")
            print(f"  {x} − {y}: " + ";  ".join(parts))


report("все запросы, правильный кадр ≤ 25 м", pos25, test)
report("все запросы, правильный кадр ≤ 10 м", pos10, [q for q in test if q in pos10])
with_text = [q for q in test if text_linked[q]]
without = [q for q in test if not text_linked[q]]
report("запросы с текстом, привязанным к объекту, ≤ 25 м", pos25, with_text)
report("запросы без текста в графе, ≤ 25 м", pos25, without)
