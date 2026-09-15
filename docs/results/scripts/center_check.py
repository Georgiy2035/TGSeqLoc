"""Схлопывание дескрипторов и центрирование при поиске, без обучения.

Для каждого сохранённого чекпойнта: средний попарный косинус дескрипторов базы, recall как есть,
после вычитания среднего дескриптора базы и после PCA-whitening по базе. База известна при
развёртывании (это карта), так что статистика берётся только из неё, запросы в неё не входят.
Для моделей с текстовой веткой — ещё доля кадров с текстом в top-5.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

from tgseqloc.config import load_config
from tgseqloc.pipeline import PipelineRunner

H = Path.home()
KEEP = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/zero")
KS = (1, 5, 10)
GROUPS = {
    "notext": [(f"keep-notext-f{f}-s42", f"keep_notext_f{f}_s42") for f in range(5)],
    "set0": [(f"f256-set0-f{f}-s42", f"f256_set0_f{f}_s42") for f in range(5)],
    "v4rl": [("v256-set0-s42", "v256_set0_s42")],
}
norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)


def ranks(db, q):
    return np.argsort(-(q @ db.T), axis=1)[:, :max(KS)]


def recall(order, qidx, positives):
    hit = np.array([[any(c in positives.get(i, ()) for c in order[r, :k]) for k in KS] for r, i in enumerate(qidx)])
    return 100 * hit.mean(0), hit


def variants(db, q):
    mu = db.mean(0, keepdims=True)
    yield "как есть", db, q
    yield "центрирование", norm(db - mu), norm(q - mu)
    x = db - mu
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    keep = s > s[0] * 1e-6
    W = vt[keep].T / (s[keep] / np.sqrt(len(db)))
    yield "PCA-whitening", norm((db - mu) @ W), norm((q - mu) @ W)


def mean_pair(a, n=1500):
    idx = np.random.default_rng(0).choice(len(a), size=min(n, len(a)), replace=False)
    g = a[idx] @ a[idx].T; m = len(idx)
    return (g.sum() - m) / (m * (m - 1))


hits_all = {}
for group in sys.argv[1:]:
    for name, cfgname in GROUPS[group]:
        path = KEEP / name / "best_model.pth"
        if not path.exists():
            print(f"{name}: чекпойнта нет, пропущен"); continue
        trainer = PipelineRunner(load_config(H / f"TGSeqLoc/configs/{cfgname}.yaml")).trainer()
        trainer.load_checkpoint_for_evaluation(path)
        model = trainer.model
        text_mask = []
        if getattr(model, "text_out", None) is not None:
            flag = {}
            model.text_out.register_forward_hook(lambda m, i, o: flag.__setitem__("t", (o.detach().norm(dim=1) > 0).cpu()))
            model.register_forward_hook(lambda m, i, o: text_mask.append(flag.pop("t", torch.zeros(o.shape[0], dtype=torch.bool))))
        qidx = list(trainer.test_query_indices)
        db = trainer._encode(trainer.database_paths)
        q = trainer._encode([trainer.query_paths[i] for i in qidx])
        tdb = tq = None
        if text_mask:
            m = torch.cat(text_mask).numpy(); tdb, tq = m[:len(db)], m[len(db):]
        ref = json.load(open(KEEP / name / "final_metrics.json"))
        print(f"\n=== {name}: база {len(db)}, запросов {len(q)}, попарный cos базы {mean_pair(db):.4f} ===")
        for label, d, qq in variants(db, q):
            order = ranks(d, qq)
            r, hit = recall(order, qidx, trainer.positives)
            hits_all.setdefault((group, label), []).append(hit)
            extra = ""
            if tdb is not None:
                extra = "  текст в top-5: у запросов с текстом %5.1f%%, без текста %5.1f%% (в базе %.1f%%)" % (
                    100 * tdb[order[tq, :5]].mean(), 100 * tdb[order[~tq, :5]].mean(), 100 * tdb.mean())
            line = "  %-14s попарный cos %.4f  R@1 %5.2f  R@5 %5.2f  R@10 %5.2f%s" % (label, mean_pair(d), *r, extra)
            if label == "как есть":
                same = abs(r[1] - ref["R@5"]) < 1e-6
                line += "  (= final_metrics)" if same else "  (final_metrics R@5 %.2f — FAISS разошёлся с numpy)" % ref["R@5"]
            print(line, flush=True)

print("\n=== итого по всем запросам (сид 42) ===")
for (group, label), hits in hits_all.items():
    h = np.concatenate(hits)
    print("  %-7s %-14s R@1 %5.2f  R@5 %5.2f  R@10 %5.2f  (запросов %d)" % (group, label, *(100 * h.mean(0)), len(h)))
