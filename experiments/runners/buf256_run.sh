#!/bin/sh
# Буферная зона для негативов, RobotCar, 256/256, кросс-валидация 5×5.
# Ветки: без текста и Qwen (слагаемое) — без буфера и с буфером 50 м;
# перемешанный Qwen (слагаемое) с буфером 50 м — контроль содержания.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/f256_runs"; mkdir -p "$OUT"
UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1

while [ -f "$SC/buf_prep.pid" ] && kill -0 "$(cat "$SC/buf_prep.pid")" 2>/dev/null; do sleep 30; done
grep -q PREP_OK "$SC/buf_prep.log" || { echo "подготовка не прошла — не запускаю"; tail -5 "$SC/buf_prep.log"; echo BUF_FAIL; exit 1; }
echo "подготовка прошла $(date +%H:%M)"

echo "=== координаты в разбиения перемешанной ветки ==="
for fold in 0 1 2 3 4; do
  $UV run tgseqloc prepare -c "configs/cv_shuffled_qwen_f${fold}_s42.yaml" > "$SC/bufprep_shuffled_f${fold}.log" 2>&1 \
    || { echo "ОШИБКА подготовки перемешанной ветки, фолд $fold"; echo BUF_FAIL; exit 1; }
done
python3 - <<'PY' || { echo BUF_FAIL; exit 1; }
import json, sys
from pathlib import Path
D = Path.home() / "tgseqloc_data"; bad = 0
for f in range(5):
    new = json.load(open(D / f"rcgf_shuffled_qwen/robotcar/mappings/split_fold{f}.json"))
    old = json.load(open(D / f"cv_runs/split_shuffled_qwen_f{f}.json"))
    ok = all(new[k] == old[k] for k in ("train_query_indices", "validation_query_indices", "test_query_indices")) and new.get("query_positions")
    print("  фолд %d: %s" % (f, "ок" if ok else "НЕ ТО")); bad += not ok
sys.exit(1 if bad else 0)
PY

wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
for fold in 0 1 2 3 4; do
  for arm in notext notextbuf add addbuf shufaddbuf; do
    for seed in 42 43 44 45 46; do
      out="$OUT/${arm}_f${fold}_${seed}.json"
      grep -q final_metrics "$out" 2>/dev/null && continue
      wait_gpu
      echo "=== ${arm} фолд ${fold} сид ${seed} $(date +%H:%M) ==="
      $UV run tgseqloc train -c "configs/f256_${arm}_f${fold}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
      rm -f "outputs/f256-${arm}-f${fold}-s${seed}/best_model.pth" "outputs/f256-${arm}-f${fold}-s${seed}/last_model.pth"
    done
  done
done
echo BUF_DONE
