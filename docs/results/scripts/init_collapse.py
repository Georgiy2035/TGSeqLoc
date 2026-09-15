"""Схлопнуты ли дескрипторы уже у необученной модели: RobotCar и V4RL, без текста/набор, сид 42."""
from pathlib import Path
import numpy as np, torch
from tgseqloc.config import load_config
from tgseqloc.pipeline import PipelineRunner
from tgseqloc.training.trainer import EdgeAttrNormalizer, _torch_load
H = Path.home(); KEEP = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/zero")
def pair(a, n=1500):
    i = np.random.default_rng(0).choice(len(a), size=min(n, len(a)), replace=False); g = a[i] @ a[i].T; m = len(i)
    return (g.sum() - m) / (m * (m - 1))
for cfgname, ck in (("keep_notext_f0_s42", "keep-notext-f0-s42"), ("v256_set0_s42", "v256-set0-s42")):
    tr = PipelineRunner(load_config(H / f"TGSeqLoc/configs/{cfgname}.yaml")).trainer()
    payload = _torch_load(KEEP / ck / "best_model.pth")
    tr.normalizer = EdgeAttrNormalizer.from_state_dict(payload["normalizer_state"])
    db0 = tr._encode(tr.database_paths)                      # необученная модель (веса от сида)
    tr.model.load_state_dict(payload["model_state_dict"]); db1 = tr._encode(tr.database_paths)
    b = tr.model.proj[2].bias.detach().cpu().numpy(); s_norm = None
    print(f"{ck}: попарный cos базы — до обучения {pair(db0):.4f}, после (эпоха {payload.get('epoch')}) {pair(db1):.4f}; |bias proj| = {np.linalg.norm(b):.3f}")
