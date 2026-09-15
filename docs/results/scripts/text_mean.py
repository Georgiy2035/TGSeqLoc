"""Гипотеза «текст = признак наличия»: набор строк с нулевым стартом, RobotCar, сид 42, пять фолдов.

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
norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
RUNS = [("v256-set0-s42", "v256_set0_s42")] if arm == "v4rl" else [(f"f256-{arm}-f{f}-s42", f"f256_{arm}_f{f}_s42") for f in range(5)]
for f, (name, cfgname) in enumerate(RUNS):
    cfg = load_config(H / f"TGSeqLoc/configs/{cfgname}.yaml")
    trainer = PipelineRunner(cfg).trainer()
    trainer.load_checkpoint_for_evaluation(KEEP / name / "best_model.pth")
    model = trainer.model
    store, cur = [], {}
    model.proj.register_forward_hook(lambda m, i, o: cur.__setitem__("s", o.detach().cpu()))
    model.text_out.register_forward_hook(lambda m, i, o: cur.__setitem__("wt", o.detach().cpu()))
    model.register_forward_hook(lambda m, i, o: store.append((cur.pop("s"), cur.pop("wt", torch.zeros_like(cur.get("s", o.detach().cpu()))))))
    qidx = list(trainer.test_query_indices)
    db, s_db, wt_db = encode(trainer, trainer.database_paths, store)
    q, s_q, wt_q = encode(trainer, [trainer.query_paths[i] for i in qidx], store)
    db0, q0 = norm(s_db), norm(s_q)
    tdb = np.linalg.norm(wt_db, axis=1) > 0; tq = np.linalg.norm(wt_q, axis=1) > 0

    mean_all = norm(db0.mean(0, keepdims=True))[0]          # общее направление базы (без текста)
    mean_notext = norm(db0[~tdb].mean(0, keepdims=True))[0]
    w = norm(wt_db[tdb]); sd = norm(s_db[tdb])
    cos_w_mean = w @ mean_all
    cos_w_mean_nt = w @ mean_notext
    cos_s_mean = sd @ mean_all
    # случайный базис: какой косинус с тем же средним у случайного направления той же размерности
    rnd = norm(np.random.default_rng(0).standard_normal((2000, db.shape[1]))) @ mean_all
    # дескрипторы кадров с текстом стягиваются друг к другу?
    def mean_pair(a):
        g = a @ a.T; n = len(a); return (g.sum() - n) / (n * (n - 1))
    sub = np.random.default_rng(1).choice(np.flatnonzero(tdb), size=min(1500, tdb.sum()), replace=False)
    subn = np.random.default_rng(2).choice(np.flatnonzero(~tdb), size=min(1500, (~tdb).sum()), replace=False)
    pair_t_on, pair_t_off = mean_pair(db[sub]), mean_pair(db0[sub])
    pair_n = mean_pair(db[subn])
    cross_on = float((db[sub] @ db[subn].T).mean()); cross_off = float((db0[sub] @ db0[subn].T).mean())
    # доля кадров с текстом в top-5 у запросов с текстом и без
    def share(dbm, qm, mask):
        order = np.argsort(-(qm[mask] @ dbm.T), axis=1)[:, :5]
        return 100 * tdb[order].mean()
    rows.append(dict(f=f, cwm=np.median(cos_w_mean), cwm_q=np.percentile(cos_w_mean, [10, 90]), cwnt=np.median(cos_w_mean_nt),
                     csm=np.median(cos_s_mean), rnd=np.percentile(np.abs(rnd), 90),
                     pt_on=pair_t_on, pt_off=pair_t_off, pn=pair_n, cr_on=cross_on, cr_off=cross_off,
                     sh_t=(share(db, q, tq), share(db0, q0, tq)), sh_n=(share(db, q, ~tq), share(db0, q0, ~tq)),
                     base=100 * tdb.mean()))
    print(f"фолд {f} готов", flush=True)

print(f"\n=== {arm}: куда направлена текстовая добавка, RobotCar, сид 42 ===")
print("фолд  cos(W·t, средний дескр. базы) медиана [p10;p90]  то же к среднему кадров без текста  cos(s, средний)  |cos| случайного направления p90")
for r in rows:
    print("  %d        %+.3f [%+.3f; %+.3f]                    %+.3f                        %+.3f            %.3f" % (
        r["f"], r["cwm"], *r["cwm_q"], r["cwnt"], r["csm"], r["rnd"]))
print("\nсредний попарный косинус дескрипторов базы:")
print("фолд  текст↔текст: с веткой / без   без текста↔без текста   текст↔без текста: с веткой / без")
for r in rows:
    print("  %d        %.4f / %.4f                %.4f                    %.4f / %.4f" % (r["f"], r["pt_on"], r["pt_off"], r["pn"], r["cr_on"], r["cr_off"]))
print("\nдоля кадров с текстом в top-5 (в базе их %s):" % ", ".join("%.1f%%" % r["base"] for r in rows))
print("фолд  запросы с текстом: с веткой / без    запросы без текста: с веткой / без")
for r in rows:
    print("  %d          %5.1f / %5.1f                       %5.1f / %5.1f" % (r["f"], *r["sh_t"], *r["sh_n"]))
