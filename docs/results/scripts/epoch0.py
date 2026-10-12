"""Recall необученной модели (эпоха 0): те же веса, с которых стартует обучение, RobotCar без текста, сид 42."""
from pathlib import Path

import numpy as np

from tgseqloc.config import load_config
from tgseqloc.pipeline import PipelineRunner
from tgseqloc.training.trainer import EdgeAttrNormalizer, _torch_load

H = Path.home()
KEEP = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/zero")
for f in (0, 3):
    for label, cfg in (("как сейчас", f"keep_notext_f{f}_s42"), ("output-норм.", f"nrm_notext_f{f}_output"), ("pooled-норм.", f"nrm_notext_f{f}_pooled")):
        tr = PipelineRunner(load_config(H / f"TGSeqLoc/configs/{cfg}.yaml")).trainer()
        # нормировщик рёбер фиксирован обучающими данными фолда; берём его из чекпойнта того же фолда
        tr.normalizer = EdgeAttrNormalizer.from_state_dict(_torch_load(KEEP / f"keep-notext-f{f}-s42/best_model.pth")["normalizer_state"])
        tr.model.eval()
        val, _ = tr.evaluate(tr.validation_query_indices) if getattr(tr, "validation_query_indices", None) else ({}, None)
        test, _ = tr.evaluate(tr.test_query_indices)
        db = tr._encode(tr.database_paths)
        idx = np.random.default_rng(0).choice(len(db), 1500, replace=False)
        g = db[idx] @ db[idx].T
        cos = (g.sum() - 1500) / (1500 * 1499)
        print(f"фолд {f} {label:13s} эпоха 0: вал R@5 {val.get('R@5', float('nan')):5.1f}  тест R@1 {test['R@1']:5.2f} R@5 {test['R@5']:5.2f} R@10 {test['R@10']:5.2f}  попарный cos базы {cos:.4f}", flush=True)
