"""Текст как добавка к оценке кандидата, без вентилей и без обучения.

    score = порядок_по_сцене + λ · Σ_надписей idf · max(0, cos − фон)

Пустая сумма для кадра без текста — ровно ноль, поэтому такие кадры и такие пары
обрабатываются в точности как бестекстовой моделью. Считается поверх сохранённых
списков кандидатов: модель не запускается, поэтому recall на размере списка (20)
неизменен по построению, а меняться могут только R@1, R@5 и R@10.

λ на RobotCar выбирается по четырём фолдам и применяется к пятому — на тестовых
запросах фолда она не подбирается. На V4RL тестовый набор один, подбирать λ на нём
значило бы выбирать по тесту, поэтому берётся λ, выбранная на RobotCar.
"""
import itertools, json, math, random, statistics as st, sys
from collections import Counter
from pathlib import Path
import torch

H = Path.home()
DATASET = sys.argv[1]
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)

if DATASET == "robotcar":
    TEXT_ROOT = H / "tgseqloc_data/rcgf_real_qwen/robotcar"
    GROUPS = [(f, s) for f in range(5) for s in range(42, 47)]
    split_of = lambda f: json.load(open(H / f"tgseqloc_data/cv_runs/split_notext_f{f}.json"))
    runs_of = lambda f, s: H / f"TGSeqLoc/outputs/cv-notext-f{f}-s{s}/test_retrievals.json"
    REFERENCE = None
else:
    TEXT_ROOT = H / "tgseqloc_data/prepared/v4rl"
    GROUPS = [(0, s) for s in range(42, 52)]
    split_of = lambda f: json.load(open(TEXT_ROOT / "mappings/temporal_split.json"))
    runs_of = lambda f, s: H / f"TGSeqLoc/outputs/c10-notext-s{s}/test_retrievals.json"
    REFERENCE = lambda s: H / f"TGSeqLoc/outputs/c10-real-s{s}/test_retrievals.json"

split0 = split_of(0)
texts = {}
for rel in split0["database_paths"] + split0["query_paths"]:
    g = torch.load(TEXT_ROOT / rel, weights_only=False)
    strings = [t.strip().lower() for t in (getattr(g, "text_strings", None) or []) if t.strip()]
    emb = torch.nn.functional.normalize(g.text_emb[g.is_text], dim=1) if strings else None
    texts[Path(rel).stem] = (strings, emb)

df = Counter()
for rel in split0["database_paths"]:
    df.update(set(texts[Path(rel).stem][0]))
N = len(split0["database_paths"])
IDF_MAX = math.log(N / 1)
idf = {s: math.log(N / (1 + c)) for s, c in df.items()}

rnd = random.Random(0)
vecs = [(s, e[i]) for ss, e in texts.values() if e is not None for i, s in enumerate(ss)]
rnd.shuffle(vecs)
sample = vecs[:1500]
bg = [float(sample[a][1] @ sample[b][1]) for a, b in
      ((rnd.randrange(len(sample)), rnd.randrange(len(sample))) for _ in range(20000))
      if sample[a][0] != sample[b][0]] if sample else [0.0]
B = st.median(bg)
print("=== %s: кадров базы %d, различных строк %d, фон %.3f ===" % (DATASET, N, len(df), B))


def text_score(q_stem, c_stem):
    qs, qe = texts.get(q_stem, ([], None))
    cs, ce = texts.get(c_stem, ([], None))
    if qe is None or ce is None:
        return 0.0
    sims = qe @ ce.T
    total = 0.0
    for i, s in enumerate(qs):
        j = int(sims[i].argmax())
        total += min(idf.get(s, IDF_MAX), idf.get(cs[j], IDF_MAX)) * max(0.0, float(sims[i][j]) - B)
    return total


def recalls(fold, seed, lam):
    split = split_of(fold)
    q, d = split["query_paths"], split["database_paths"]
    positives = {int(k): set(v) for k, v in split["positives"].items()}
    r = json.load(open(runs_of(fold, seed)))
    hits = {1: 0, 5: 0, 10: 0}; hits_text = {1: 0, 5: 0, 10: 0}; n_text = 0; moved = 0
    for qi, cands in zip(r["query_indices"], r["nearest_database_indices"]):
        stem = Path(q[qi]).stem
        has_text = texts.get(stem, ([], None))[1] is not None
        K = len(cands)
        if lam and has_text:
            scores = [text_score(stem, Path(d[c]).stem) for c in cands]
            order = sorted(range(K), key=lambda i: -((K - i) / K + lam * scores[i]))
        else:
            order = list(range(K))                      # нет текста — порядок не тронут
        moved += order != list(range(K))
        pos = positives.get(qi, ())
        n_text += has_text
        for k in hits:
            got = any(cands[i] in pos for i in order[:k])
            hits[k] += got
            if has_text:
                hits_text[k] += got
    n = len(r["query_indices"])
    return ({k: 100 * v / n for k, v in hits.items()},
            {k: 100 * v / max(1, n_text) for k, v in hits_text.items()}, n, n_text, moved)


def sign_test(d):
    n = len(d); obs = sum(d) / n
    if n <= 20:
        hits = sum(1 for s in itertools.product((1, -1), repeat=n)
                   if abs(sum(si * x for si, x in zip(s, d)) / n) >= abs(obs) - 1e-12)
        return obs, hits / 2 ** n
    rr = random.Random(0)
    trials = 20000
    hits = sum(1 for _ in range(trials)
               if abs(sum(x if rr.random() < 0.5 else -x for x in d) / n) >= abs(obs) - 1e-12)
    return obs, hits / trials


cache = {}
for fold, seed in GROUPS:
    for lam in LAMBDAS:
        cache[(fold, seed, lam)] = recalls(fold, seed, lam)

if DATASET == "robotcar":
    print("\nвыбор λ: по четырём фолдам, применяется к пятому")
    chosen = {}
    for fold in range(5):
        best, best_v = None, -1
        for lam in LAMBDAS:
            v = st.mean([cache[(f, s, lam)][0][5] for f, s in GROUPS if f != fold])
            if v > best_v:
                best, best_v = lam, v
        chosen[fold] = best
        print("  фолд %d → λ=%.2f" % (fold, best))
else:
    chosen = {0: float(sys.argv[2]) if len(sys.argv) > 2 else 0.5}
    print("\nλ перенесена с RobotCar: %.2f" % chosen[0])

base = [cache[(f, s, 0.0)] for f, s in GROUPS]
tuned = [cache[(f, s, chosen[f])] for f, s in GROUPS]
print("\n%-28s %-9s %-9s %-9s" % ("", "R@1", "R@5", "R@10"))
for label, rows, idx in (("все запросы, без текста в оценке", base, 0),
                         ("все запросы, с текстом в оценке", tuned, 0),
                         ("запросы с текстом, без текста в оценке", base, 1),
                         ("запросы с текстом, с текстом в оценке", tuned, 1)):
    vals = {k: st.mean([r[idx][k] for r in rows]) for k in (1, 5, 10)}
    print("%-38s %6.2f    %6.2f    %6.2f" % (label, vals[1], vals[5], vals[10]))

print("\nпарные сравнения «с текстом» против «без текста», по %d прогонам:" % len(GROUPS))
for idx, what in ((0, "все запросы"), (1, "запросы с текстом")):
    line = []
    for k in (1, 5, 10):
        d = [t[idx][k] - b[idx][k] for b, t in zip(base, tuned)]
        obs, p = sign_test(d)
        line.append("R@%d Δ=%+5.2f p=%.3f %d/%d" % (k, obs, p, sum(1 for x in d if x > 0), len(d)))
    print("  %-20s %s" % (what, "   ".join(line)))

print("\nзапросов с текстом %d из %d; порядок менялся у %d запросов на прогон"
      % (tuned[0][3], tuned[0][2], st.mean([r[4] for r in tuned])))

if REFERENCE:
    ref = []
    for _, s in GROUPS:
        r = json.load(open(REFERENCE(s)))
        split = split_of(0); positives = {int(k): set(v) for k, v in split["positives"].items()}
        hits = {1: 0, 5: 0, 10: 0}
        for qi, cands in zip(r["query_indices"], r["nearest_database_indices"]):
            for k in hits:
                hits[k] += any(c in positives.get(qi, ()) for c in cands[:k])
        ref.append({k: 100 * v / len(r["query_indices"]) for k, v in hits.items()})
    print("\nдля сравнения — текст, вшитый в дескриптор (ветка c10-real): R@1 %.2f  R@5 %.2f  R@10 %.2f"
          % tuple(st.mean([r[k] for r in ref]) for k in (1, 5, 10)))
