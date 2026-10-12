"""Конфиги аддитивного текста: RobotCar (кросс-валидация) и V4RL, без dropout и с 0.5."""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"
TAGS = {"p0": 0.0, "p05": 0.5}


def additive(text: str, dropout: float, name: str) -> str:
    lines = text.splitlines()
    i = lines.index("model:")
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("  use_text_nodes:"))
    assert lines[j].strip() == "use_text_nodes: true", lines[j]
    lines[j + 1:j + 1] = ["  text_fusion: additive", f"  text_dropout: {dropout}"]
    k = next(k for k, line in enumerate(lines) if line.startswith("  experiment_name:"))
    lines[k] = f"  experiment_name: {name}"
    return "\n".join(lines) + "\n"


n = 0
for tag, p in TAGS.items():
    for fold in range(5):
        for seed in range(42, 47):
            src = (C / f"cv_real_qwen_f{fold}_s{seed}.yaml").read_text(encoding="utf-8")
            (C / f"add_rc_{tag}_f{fold}_s{seed}.yaml").write_text(
                additive(src, p, f"add-rc-{tag}-f{fold}-s{seed}"), encoding="utf-8")
            n += 1
    for seed in range(42, 52):
        src = (C / f"c10_real_s{seed}.yaml").read_text(encoding="utf-8")
        (C / f"add_v4rl_{tag}_s{seed}.yaml").write_text(
            additive(src, p, f"add-v4rl-{tag}-s{seed}"), encoding="utf-8")
        n += 1
print("конфигов:", n)
