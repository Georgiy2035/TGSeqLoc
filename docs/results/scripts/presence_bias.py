"""Тянет ли текст к кадрам с текстом: доля кадров базы с текстом в топ-5 у запросов с текстом (RobotCar, разбиение v2)."""
import json, statistics as st
from pathlib import Path
import torch

H = Path.home()
for arm, root in (("PaddleOCR", "rcgf_real_paddle"), ("Qwen", "rcgf_real_qwen")):
    R = H / f"tgseqloc_data/{root}/robotcar"
    s = json.load(open(R / "mappings/temporal_split.json"))
    has_db = [bool(torch.load(R / p, weights_only=False).is_text.any()) for p in s["database_paths"]]
    has_q = {q: bool(torch.load(R / s["query_paths"][q], weights_only=False).is_text.any()) for q in s["test_query_indices"]}
    tq = [q for q in s["test_query_indices"] if has_q[q]]
    pos_text = st.mean(sum(has_db[d] for d in s["positives"][str(q)]) / len(s["positives"][str(q)]) for q in tq)
    print("%s: кадров базы с текстом %.1f %%, тестовых запросов с текстом %d из %d, среди их правильных кадров с текстом %.0f %%"
          % (arm, 100 * sum(has_db) / len(has_db), len(tq), len(has_q), 100 * pos_text))
    for label, prefix in ((f"{arm}, узел", f"v2-rcgf-real_{'paddle' if arm == 'PaddleOCR' else 'qwen'}"), ("без текста", "v2-rcg-notext")):
        top5 = []; top1 = []
        for seed in range(42, 52):
            p = H / f"TGSeqLoc/outputs/{prefix}-s{seed}/test_retrievals.json"
            if not p.exists():
                continue
            r = json.load(open(p))
            for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
                if has_q.get(q):
                    top5.append(sum(has_db[d] for d in near[:5]) / 5); top1.append(has_db[near[0]])
        if top5:
            print("  %-18s у запросов с текстом в топ-5 кадров с текстом %.0f %%, топ-1 с текстом %.0f %%" % (label, 100 * st.mean(top5), 100 * st.mean(top1)))
