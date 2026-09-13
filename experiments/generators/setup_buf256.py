"""Буферная зона 50 м на 256/256: конфиги ветвей с буфером из уже существующих f256_*."""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"


def with_buffer(text: str, name: str) -> str:
    lines = text.splitlines()
    i = lines.index("training:")
    lines.insert(i + 1, "  negative_min_distance_m: 50.0")
    k = next(k for k, line in enumerate(lines) if line.startswith("  experiment_name:"))
    lines[k] = f"  experiment_name: {name}"
    body = "\n".join(lines) + "\n"
    assert "hidden_dim: 256" in body and "proj_dim: 256" in body
    return body


n = 0
for fold in range(5):
    for seed in range(42, 47):
        for src, dst in (("notext", "notextbuf"), ("add", "addbuf"), ("shufadd", "shufaddbuf")):
            t = (C / f"f256_{src}_f{fold}_s{seed}.yaml").read_text(encoding="utf-8")
            (C / f"f256_{dst}_f{fold}_s{seed}.yaml").write_text(
                with_buffer(t, f"f256-{dst}-f{fold}-s{seed}"), encoding="utf-8")
            n += 1
print("конфигов с буфером:", n)
