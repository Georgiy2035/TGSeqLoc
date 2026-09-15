"""Этапы TGSeqLoc на кадр: фильтр текста (разбор OCR, мусор, IoA), кодирование строк, сборка графа,
GAT (GPU, пакет 1), поиск FAISS top-100 по базе финального эксперимента (16 714 кадров, 4 камеры)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch, faiss
from torch_geometric.data import Batch
from tgseqloc.config import load_config
from tgseqloc.pipeline import PipelineRunner
from tgseqloc.data.v4rl import parse_paddleocr, parse_scene_graph
from tgseqloc.preparation.v4rl import _drop_junk_text
from tgseqloc.preparation.text import build_char_ngram_encoder
from tgseqloc.preparation.fusion import build_fused_graph
from tgseqloc.inference.dynamics import build_mask_ioa
from tgseqloc.stages import frame_masks_from_json

H = Path.home(); log = Path(sys.argv[1])
frames = json.load(open(Path(__file__).with_name("frames.json")))
cfg = load_config(H / "TGSeqLoc/configs/final4s_add_f0_s42.yaml")
trainer = PipelineRunner(cfg).trainer()
ck = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/final4s/final4s-add-f0-s42/best_model.pth")
trainer.load_checkpoint_for_evaluation(ck); model = trainer.model.eval()
manifest = trainer.manifest; normalizer = trainer.normalizer
database = trainer._encode(trainer.database_paths).astype(np.float32)          # база кодируется заранее, не учитывается
faiss.normalize_L2(database); index = faiss.IndexFlatIP(database.shape[1]); index.add(database)
encoder = build_char_ngram_encoder(embedding_dim=64); ioa = build_mask_ioa(threshold=0.5)
junk_cfg = {"text_min_length": 3, "text_drop_numeric": True}
class_to_idx, edge_to_idx = manifest["node_class_to_idx"], manifest["edge_label_to_idx"]

def text_filter(f):
    ft = _drop_junk_text(junk_cfg, parse_paddleocr(f["ocr"], 0.0))
    if len(ft):
        ft = ioa.apply(ft, frame_masks_from_json(json.load(open(f["mask"]))))
    return ft
def encode(ft):
    return encoder.encode(list(ft.texts))
def graph(f, ft, emb):
    nodes, edges, _ = parse_scene_graph(f["graph"], class_to_idx, ())
    g = build_fused_graph(nodes, edges, ft.boxes, list(ft.texts), emb, edge_to_idx)
    g.edge_attr = normalizer.transform(g.edge_attr.float()); g.x = g.x.float()
    return g
def gat(g):
    with torch.no_grad():
        d = model(Batch.from_data_list([g]).to(trainer.device)); torch.cuda.synchronize()
    return d.cpu().numpy().astype(np.float32)
def search(d):
    faiss.normalize_L2(d); return index.search(d, 100)

for f in frames[:5]:
    ft = text_filter(f); g = graph(f, ft, encode(ft)); search(gat(g))
res = {k: [] for k in ("Фильтр текста (разбор OCR, мусор, IoA)", "Кодирование строк (char n-grams)", "Сборка графа", "GAT", "Поиск FAISS top-100")}
bounds = {k: [None, None] for k in res}
torch.cuda.reset_peak_memory_stats()
for f in frames[5:]:
    t = time.perf_counter(); ft = text_filter(f); a = time.perf_counter(); emb = encode(ft); b = time.perf_counter()
    g = graph(f, ft, emb); c = time.perf_counter(); d = gat(g); e = time.perf_counter(); search(d); z = time.perf_counter()
    for k, (x0, x1) in zip(res, ((t, a), (a, b), (b, c), (c, e), (e, z))):
        res[k].append(1000 * (x1 - x0))
now = time.time()
with open(log, "a") as fh:
    for k, device in zip(res, ("CPU", "CPU", "CPU", "GPU", "CPU")):
        fh.write(json.dumps({"stage": k, "device": device, "start": None, "end": None, "times_ms": res[k],
                             "torch_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if k == "GAT" else None,
                             "shared_window": "tgseqloc"}) + "\n")
    fh.write(json.dumps({"stage": "_tgseqloc_window", "start": now - sum(sum(v) for v in res.values()) / 1000, "end": now}) + "\n")
print("tgseqloc готово; строк на кадр в среднем:", "n/a")
