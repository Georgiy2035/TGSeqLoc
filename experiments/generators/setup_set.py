"""Проверка схемы «сцена + набор строк»: RobotCar 256 (кросс-валидация) и V4RL 256 (10 сидов)."""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"


def edit(text, name, *, size=False, fusion=None):
    lines = text.splitlines()
    if size:
        lines = ["  hidden_dim: 256" if l == "  hidden_dim: 128" else "  proj_dim: 256" if l == "  proj_dim: 64" else l for l in lines]
    if fusion:
        i = lines.index("model:")
        j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("  use_text_nodes:"))
        lines = [l for l in lines if not l.startswith(("  text_fusion:", "  text_dropout:", "  text_centering:"))]
        j = next(k for k in range(lines.index("model:") + 1, len(lines)) if lines[k].startswith("  use_text_nodes:"))
        lines[j + 1:j + 1] = [f"  text_fusion: {fusion}", "  text_dropout: 0.0", "  text_centering: true"]
    k = next(k for k, l in enumerate(lines) if l.startswith("  experiment_name:"))
    lines[k] = f"  experiment_name: {name}"
    body = "\n".join(lines) + "\n"
    assert "hidden_dim: 256" in body and "proj_dim: 256" in body, name
    return body


n = 0
for f in range(5):
    for s in range(42, 47):
        (C / f"f256_set_f{f}_s{s}.yaml").write_text(edit((C / f"f256_add_f{f}_s{s}.yaml").read_text(), f"f256-set-f{f}-s{s}", fusion="set"))
        (C / f"f256_shufset_f{f}_s{s}.yaml").write_text(edit((C / f"f256_shufadd_f{f}_s{s}.yaml").read_text(), f"f256-shufset-f{f}-s{s}", fusion="set"))
        n += 2
for s in range(42, 52):
    (C / f"v256_notext_s{s}.yaml").write_text(edit((C / f"c10_notext_s{s}.yaml").read_text(), f"v256-notext-s{s}", size=True))
    (C / f"v256_node_s{s}.yaml").write_text(edit((C / f"c10_real_s{s}.yaml").read_text(), f"v256-node-s{s}", size=True))
    (C / f"v256_set_s{s}.yaml").write_text(edit((C / f"c10_real_s{s}.yaml").read_text(), f"v256-set-s{s}", size=True, fusion="set"))
    n += 3
print("конфигов:", n)
