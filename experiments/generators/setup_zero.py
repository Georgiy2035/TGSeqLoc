"""Набор строк с нулевым стартом текстовой ветки: те же конфиги, что f256/v256 set, плюс text_zero_init."""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"


def edit(text, name):
    lines = [l for l in text.splitlines() if not l.startswith("  text_zero_init:")]
    j = lines.index("  text_centering: true")
    lines.insert(j + 1, "  text_zero_init: true")
    k = next(k for k, l in enumerate(lines) if l.startswith("  experiment_name:"))
    lines[k] = f"  experiment_name: {name}"
    return "\n".join(lines) + "\n"


n = 0
for f in range(5):
    for s in range(42, 47):
        for arm in ("set", "shufset"):
            (C / f"f256_{arm}0_f{f}_s{s}.yaml").write_text(edit((C / f"f256_{arm}_f{f}_s{s}.yaml").read_text(), f"f256-{arm}0-f{f}-s{s}"))
            n += 1
for s in range(42, 52):
    (C / f"v256_set0_s{s}.yaml").write_text(edit((C / f"v256_set_s{s}.yaml").read_text(), f"v256-set0-s{s}"))
    n += 1
print("конфигов:", n)
