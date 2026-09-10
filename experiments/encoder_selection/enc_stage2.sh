#!/bin/sh
# Финал выбора энкодера: 10 сидов на собственном оптимуме каждой строки.
#
# Ждёт освобождения видеопамяти: чужой процесс на карте периодически забирает
# её целиком, и три прогона скана уже погибли на CUBLAS_STATUS_ALLOC_FAILED.
# Сначала добираются те три, потом лучший шаг определяется ПО ВАЛИДАЦИИ и
# запускается финал.
cd "$HOME/TGSeqLoc"
mkdir -p /tmp/encsel

wait_gpu() {
    while :; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        # Финал идёт через train: графы уже собраны, обучению хватает
        # порядка 400 МБ, а порог в 8 ГБ стоил семи часов простоя.
        [ "$free" -ge 2500 ] && return 0
        sleep 120
    done
}

# 1. Добрать упавшие прогоны скана.
for seed in 42 43 44; do
    f=/tmp/encsel/scan_rumodernbert_1.0e-4_${seed}.json
    grep -q final_metrics "$f" 2>/dev/null && continue
    wait_gpu
    echo "=== добор rumodernbert 1.0e-4 сид ${seed} $(date +%H:%M) ==="
    ~/.local/bin/uv run tgseqloc run -c "configs/es_rumodernbert_1.0e-4_s${seed}.yaml" \
        > "$f" 2>&1 || echo "  ОШИБКА"
done

# 2. Лучший шаг по валидации.
~/.local/bin/uv run python - > /tmp/encsel/best_lr.txt <<'PY'
import json, glob, os
from collections import defaultdict
best = defaultdict(dict)
for f in sorted(glob.glob("/tmp/encsel/scan_*.json")):
    txt = open(f, encoding="utf-8", errors="replace").read()
    i = txt.find("{")
    if i < 0: continue
    try: obj, _ = json.JSONDecoder().raw_decode(txt[i:])
    except Exception: continue
    h = (obj.get("training", obj)).get("history") or []
    if not h: continue
    p = os.path.basename(f)[:-5].split("_")
    best[p[1]].setdefault(p[2], []).append(max(e["R@5"] for e in h))
for name, rows in best.items():
    full = {lr: sum(v)/len(v) for lr, v in rows.items() if len(v) == 3}
    if full:
        print(name, max(full, key=full.get))
PY
cat /tmp/encsel/best_lr.txt

# 3. Финал: 10 сидов на выбранном шаге.
while read name lr; do
    [ -z "$name" ] && continue
    for seed in 42 43 44 45 46 47 48 49 50 51; do
        out="/tmp/encsel/final_${name}_${seed}.json"
        grep -q final_metrics "$out" 2>/dev/null && continue
        sed "s|^  seed: .*|  seed: ${seed}|; s|experiment_name: .*|experiment_name: esf-${name}-s${seed}|" \
            "configs/es_${name}_${lr}_s42.yaml" > "configs/esf_${name}_s${seed}.yaml"
        wait_gpu
        echo "=== финал ${name} lr ${lr} сид ${seed} $(date +%H:%M) ==="
        ~/.local/bin/uv run tgseqloc train -c "configs/esf_${name}_s${seed}.yaml" \
            > "$out" 2>&1 || echo "  ОШИБКА"
    done
done < /tmp/encsel/best_lr.txt

echo ENC_FINAL_DONE
