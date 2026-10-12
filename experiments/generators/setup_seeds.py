"""Сиды 43 и 44 для пары «набор строк / слагаемое» — проверка главного утверждения на разбросе сидов."""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"
BASE = {"set": C / "v4abl_set_ref_s42.yaml", "add": C / "v4abl_add_qwen_s42.yaml"}
n = 0
for arm, src in BASE.items():
    for seed in (43, 44):
        out = []
        for line in src.read_text().splitlines():
            if line.startswith("  seed:"):
                line = f"  seed: {seed}"
            elif line.startswith("  experiment_name:"):
                line = f"  experiment_name: v4abl-{arm}_ref-s{seed}"
            out.append(line)
        body = "\n".join(out) + "\n"
        assert f"seed: {seed}" in body and ("text_fusion: set" if arm == "set" else "text_fusion: additive") in body
        (C / f"v4abl_{arm}_ref_s{seed}.yaml").write_text(body)
        n += 1
print("конфигов сидов:", n)
