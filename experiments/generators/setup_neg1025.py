"""Выбор негативов при позитивах 10/25 на однослойном GAT: RobotCar, фронтальная камера, графы Gemini, текст Qwen.

Фиксировано: GAT 1 слой, 256/256, позитивы для обучения ≤ 10 м (10–25 м не используются), оценка 25 м,
lr 3e-5, batch 8, dropout 0.1, triplet 0.3, 2 негатива, 30 эпох без ранней остановки (patience 30), эпоха по валидации.
Варьируется: hard — ближайшие негативы с первой эпохи; warm — 10 эпох случайных, затем ближайшие.
Варианты: без текста, Qwen-слагаемое, перемешанный Qwen-слагаемое. 5 фолдов × сиды 42–44.
"""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"
ARMS = {"notext": "f256_notextp10", "add": "f256_addp10", "shufadd": "f256_shufadd"}
n = 0
for arm, prefix in ARMS.items():
    for f in range(5):
        for s in (42, 43, 44):
            lines = (C / f"{prefix}_f{f}_s{s}.yaml").read_text().splitlines()
            if arm == "shufadd":   # в перемешанном контроле 10/25 ещё не было — добавляем как в addp10
                i = next(k for k, l in enumerate(lines) if l.startswith("  learning_rate:"))
                lines.insert(i + 1, "  positive_max_distance_m: 10.0")
            assert any(l == "  positive_max_distance_m: 10.0" for l in lines), (arm, f, s)
            for mode in ("hard", "warm"):
                out = []
                for l in lines:
                    if l.startswith("  epochs:"):
                        l = "  epochs: 30"
                    elif l.startswith("  patience:"):
                        l = "  patience: 30"
                    elif l.startswith("  experiment_name:"):
                        l = f"  experiment_name: neg1025-{arm}-{mode}-f{f}-s{s}"
                    out.append(l)
                    if l == "  proj_dim: 256":
                        out.append("  n_layers: 1")
                    if l.startswith("  learning_rate:") and mode == "warm":
                        out.append("  miner_warmup_epochs: 10")
                body = "\n".join(out) + "\n"
                assert body.count("n_layers: 1") == 1 and body.count("epochs: 30") == 1
                (C / f"neg1025_{arm}_{mode}_f{f}_s{s}.yaml").write_text(body)
                n += 1
print("конфигов:", n)
