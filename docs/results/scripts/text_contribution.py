"""Вклад текстовой ветки в дескриптор: набор строк с нулевым стартом, RobotCar, сид 42, пять фолдов.

По лучшему чекпойнту каждого фолда: норма текстовой добавки W·t против нормы сценовой части s
на кадрах с текстом, угол, на который текст поворачивает дескриптор, и recall того же чекпойнта
с выключенной текстовой веткой (W = 0) — отдельно для запросов с текстом и без.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from tgseqloc.config import load_config
from tgseqloc.pipeline import PipelineRunner

H = Path.home()
KEEP = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/zero")
KS = (1, 5, 10)
arm = sys.argv[1] if len(sys.argv) > 1 else "set0"


def encode(trainer, paths, store):
    store.clear()
    emb = trainer._encode(paths)
    s = torch.cat([b[0] for b in store]); wt = torch.cat([b[1] for b in store])
    return emb, s.numpy(), wt.numpy()


def recall(db, q, qidx, positives, mask=None):
    sims = q @ db.T
    order = np.argsort(-sims, axis=1)[:, :max(KS)]
    hit = np.array([[any(c in positives.get(i, ()) for c in order[r, :k]) for k in KS] for r, i in enumerate(qidx)])
    if mask is not None:
        hit = hit[mask]
    return 100 * hit.mean(0), len(hit)


rows = []
RUNS = [("v256-set0-s42", "v256_set0_s42")] if arm == "v4rl" else [(f"f256-{arm}-f{f}-s42", f"f256_{arm}_f{f}_s42") for f in range(5)]
for f, (name, cfgname) in enumerate(RUNS):
    cfg = load_config(H / f"TGSeqLoc/configs/{cfgname}.yaml")
    runner = PipelineRunner(cfg)
    trainer = runner.trainer()
    payload = trainer.load_checkpoint_for_evaluation(KEEP / name / "best_model.pth")
    model = trainer.model
    store, cur = [], {}
    model.proj.register_forward_hook(lambda m, i, o: cur.__setitem__("s", o.detach().cpu()))
    model.text_out.register_forward_hook(lambda m, i, o: cur.__setitem__("wt", o.detach().cpu()))

    def on_forward(m, i, o):
        s = cur.pop("s"); wt = cur.pop("wt", torch.zeros_like(s))
        store.append((s, wt))
    model.register_forward_hook(on_forward)

    qidx = list(trainer.test_query_indices)
    db, s_db, wt_db = encode(trainer, trainer.database_paths, store)
    q, s_q, wt_q = encode(trainer, [trainer.query_paths[i] for i in qidx], store)

    s_all = np.concatenate([s_db, s_q]); wt_all = np.concatenate([wt_db, wt_q])
    has = np.linalg.norm(wt_all, axis=1) > 0
    ns, nw = np.linalg.norm(s_all[has], axis=1), np.linalg.norm(wt_all[has], axis=1)
    cos_sw = (s_all[has] * wt_all[has]).sum(1) / (ns * nw)
    full = s_all[has] + wt_all[has]
    turn = np.degrees(np.arccos(np.clip((full * s_all[has]).sum(1) / (np.linalg.norm(full, axis=1) * ns), -1, 1)))

    qtext = np.linalg.norm(wt_q, axis=1) > 0
    # без текстовой ветки: дескриптор = normalize(s), ровно как при W = 0
    norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    db0, q0 = norm(s_db), norm(s_q)
    assert np.allclose(norm(s_db + wt_db), db, atol=1e-5), "перехваченные s и W·t не собираются в дескриптор"
    r_all, n_all = recall(db, q, qidx, trainer.positives)
    r_all0, _ = recall(db0, q0, qidx, trainer.positives)
    r_t, n_t = recall(db, q, qidx, trainer.positives, qtext)
    r_t0, _ = recall(db0, q0, qidx, trainer.positives, qtext)
    r_n, n_n = recall(db, q, qidx, trainer.positives, ~qtext)
    r_n0, _ = recall(db0, q0, qidx, trainer.positives, ~qtext)
    rows.append(dict(f=f, epoch=payload.get("epoch"), share=has.mean(), ratio=np.median(nw / ns),
                     ratio_q=np.percentile(nw / ns, [10, 90]), cos=np.median(cos_sw), turn=np.median(turn),
                     turn_q=np.percentile(turn, [10, 90]), r=(r_all, r_all0), rt=(r_t, r_t0, n_t), rn=(r_n, r_n0, n_n),
                     wnorm=float(model.text_out.weight.norm()), wabs=float(model.text_out.weight.abs().mean())))
    print(f"фолд {f} готов", flush=True)

print(f"\n=== {arm}: вклад текста в дескриптор, RobotCar, сид 42, лучший чекпойнт ===")
print("фолд эпоха  кадров с текстом  |W·t|/|s| медиана [p10;p90]  cos(s,W·t)  поворот, ° медиана [p10;p90]  |W|_F  mean|w|")
for r in rows:
    print("  %d    %s      %5.1f %%          %.3f [%.3f; %.3f]        %+.3f       %5.1f [%4.1f; %5.1f]          %.3f  %.5f" % (
        r["f"], r["epoch"], 100 * r["share"], r["ratio"], *r["ratio_q"], r["cos"], r["turn"], *r["turn_q"], r["wnorm"], r["wabs"]))
print("\nrecall того же чекпойнта с текстом / без текстовой ветки (W = 0):")
print("фолд  все запросы R@1 R@5 R@10                     запросы с текстом (n)                          запросы без текста (n)")
fmt = lambda a, b: " ".join("%5.1f/%5.1f" % (x, y) for x, y in zip(a, b))
for r in rows:
    print("  %d   %s   %s (%d)   %s (%d)" % (r["f"], fmt(*r["r"]), fmt(r["rt"][0], r["rt"][1]), r["rt"][2], fmt(r["rn"][0], r["rn"][1]), r["rn"][2]))
tot = lambda key, j: (sum(r[key][0][j] * r[key][2] for r in rows) / sum(r[key][2] for r in rows),
                      sum(r[key][1][j] * r[key][2] for r in rows) / sum(r[key][2] for r in rows))
print("\nпо пяти фолдам (взвешено по запросам), R@5 с текстом → без ветки:")
print("  запросы с текстом:  %.2f → %.2f" % tot("rt", 1))
print("  запросы без текста: %.2f → %.2f" % tot("rn", 1))
