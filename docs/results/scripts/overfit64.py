"""Может ли модель выучить 64 фиксированные тройки: проверка метода отдельно от объёма данных.

RobotCar без текста, фолд 0, только обучающая часть. Тройки отбираются автоматически:
  лёгкие — позитив ≤ 10 м, негатив > 200 м; трудные — позитив ≤ 10 м, негатив 25–50 м.
Негативы берутся из кадров базы, которые сами являются позитивами обучающих запросов (не из теста).
Архитектуры: пулинг + MLP (GAT без слоёв), GAT 1 слой, GAT 2 слоя; каждая как сейчас и с BatchNorm
перед проекцией (проекция без bias). Dropout 0, без майнинга, все 64 тройки одним батчем,
Adam lr 1e-3, 500 шагов. Метрика — доля троек, где cos(запрос, позитив) > cos(запрос, негатив),
на обученных тройках и на 64 отложенных тройках того же типа.
"""
import json, math, random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from tgseqloc.models import GATGraphEncoder
from tgseqloc.training.trainer import load_graph, EdgeAttrNormalizer, _torch_load

torch.backends.cudnn.benchmark = False
ROOT = Path.home() / "tgseqloc_data/rcg_notext/robotcar"
KEEP = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/zero/keep-notext-f0-s42/best_model.pth")
split = json.load(open(ROOT / "mappings/split_fold0.json"))
manifest = json.load(open(ROOT / "manifest.json"))
normalizer = EdgeAttrNormalizer.from_state_dict(_torch_load(KEEP)["normalizer_state"])  # обучен на train фолда 0
device = torch.device("cuda")
qpos = {int(k): np.array(v) for k, v in split["query_positions"].items()}
dpos = {int(k): np.array(v) for k, v in split["database_positions"].items()}
train_q = [int(i) for i in split["train_query_indices"]]
train_db = sorted({int(j) for i in train_q for j in split["positives"].get(str(i), [])})
db_xy = np.stack([dpos[j] for j in train_db])


def triples(kind, count, rng, exclude=()):
    out, pool = [], [i for i in train_q if i not in exclude]
    rng.shuffle(pool)
    for i in pool:
        d = np.linalg.norm(db_xy - qpos[i], axis=1)
        pos = [train_db[k] for k in np.flatnonzero(d <= 10)]
        lo, hi = (200, 1e9) if kind == "лёгкие" else (25, 50)
        neg = [train_db[k] for k in np.flatnonzero((d > lo) & (d <= hi))]
        if pos and neg:
            out.append((i, rng.choice(pos), rng.choice(neg)))
        if len(out) == count:
            break
    return out


cache = {}
def graph(path):
    if path not in cache:
        cache[path] = load_graph(ROOT / path, normalizer, manifest["text_embedding_dim"], edge_attr_dim=manifest["edge_attr_dim"])
    return cache[path]


def batch(ts):
    gs = []
    for q, p, n in ts:
        gs += [graph(split["query_paths"][q]), graph(split["database_paths"][p]), graph(split["database_paths"][n])]
    return Batch.from_data_list(gs).to(device)


def build(layers, norm):
    torch.manual_seed(42)
    m = GATGraphEncoder(in_dim=4, hidden_dim=256, n_layers=max(1, layers), proj_dim=256,
                        num_node_classes=manifest["num_obj_classes"], node_emb_dim=64,
                        num_edge_classes=manifest["num_edge_classes"], edge_emb_dim=64, edge_cont_dim=10,
                        dropout=0.0, heads=4, use_text_nodes=False, text_emb_dim=manifest["text_embedding_dim"],
                        use_edge_geometry=True, descriptor_norm="pooled" if norm else "none", proj_bias=not norm)
    if layers == 0:
        m.convs = nn.ModuleList()   # пулинг + MLP: признаки узлов → input_mlp → mean/max → проекция
    return m.to(device)


def accuracy(m, b, train_mode=False):
    m.train(train_mode)
    with torch.no_grad():
        e = m(b).reshape(-1, 3, m.out_dim)
    sp, sn = (e[:, 0] * e[:, 1]).sum(1), (e[:, 0] * e[:, 2]).sum(1)
    return 100 * float((sp > sn).float().mean()), float((sp - sn).mean())


rng = random.Random(0)
print("%-8s %-20s %-5s | %-26s | %-26s | %s" % ("тройки", "модель", "", "обучение: доля верных до→после", "отложенные: до→после", "loss до→после, cos(q,·) разброс"))
for kind in ("лёгкие", "трудные"):
    tr = triples(kind, 64, rng)
    ho = triples(kind, 64, rng, exclude={t[0] for t in tr})
    btr, bho = batch(tr), batch(ho)
    for layers, name in ((0, "пулинг + MLP"), (1, "GAT 1 слой"), (2, "GAT 2 слоя")):
        for norm in (False, True):
            m = build(layers, norm)
            crit = nn.TripletMarginLoss(margin=0.3, p=2)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            a0, _ = accuracy(m, btr); h0, _ = accuracy(m, bho)
            m.train()
            losses = []
            for step in range(500):
                e = m(btr).reshape(-1, 3, m.out_dim)
                loss = crit(e[:, 0], e[:, 1], e[:, 2])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
            a1, gap = accuracy(m, btr); h1, _ = accuracy(m, bho)
            with torch.no_grad():
                m.eval(); e = m(btr).reshape(-1, 3, m.out_dim)
                spread = float(1 - (e[:, 0] @ e[:, 0].T).mean())
            print("%-8s %-20s %-5s | %6.1f%% → %6.1f%% (Δcos %+.3f) | %6.1f%% → %6.1f%%          | %.3f → %.3f, 1−cos %.4f" % (
                kind, name, "BN" if norm else "", a0, a1, gap, h0, h1, losses[0], np.mean(losses[-20:]), spread), flush=True)
