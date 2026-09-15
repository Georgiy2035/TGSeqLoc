"""Финальный эксперимент, 4 камеры, тройки внутри одной камеры: те же конфиги final4 + training.same_camera_triplets.

Позитив и негативы тройки берутся из кадров той же камеры, что запрос; оценка не меняется (любая камера ≤ 25 м),
дополнительно считается recall «та же камера» по сохранённым 100 соседям.
"""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"
n = 0
for arm in ("add", "notext"):
    for f in range(5):
        for s in (42, 43, 44):
            lines = (C / f"final4_{arm}_f{f}_s{s}.yaml").read_text().splitlines()
            out = []
            for l in lines:
                if l.startswith("  experiment_name:"):
                    l = f"  experiment_name: final4s-{arm}-f{f}-s{s}"
                out.append(l)
                if l.startswith("  miner_warmup_epochs:"):
                    out.append("  same_camera_triplets: true")
            body = "\n".join(out) + "\n"
            assert body.count("same_camera_triplets: true") == 1 and "top_k: 100" in body
            (C / f"final4s_{arm}_f{f}_s{s}.yaml").write_text(body)
            n += 1
print("конфигов:", n)
