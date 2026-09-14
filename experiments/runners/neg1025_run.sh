#!/bin/sh
# Выбор негативов при 10/25, GAT 1 слой. $1 = gate | stream. Порядок: сид 42 целиком по всем фолдам и вариантам, затем 43, 44.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/neg1025_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
mkdir -p "$OUT"
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
if [ "$1" = gate ]; then
  python3 experiments/generators/setup_neg1025.py || { echo NEG_FAIL; exit 1; }
  sed "s/^\(  experiment_name:\).*/\1 repro-neg1025-addp10-f0/" configs/f256_addp10_f0_s42.yaml > configs/repro_neg1025_addp10_f0.yaml
  $UV run tgseqloc train -c configs/repro_neg1025_addp10_f0.yaml > "$SC/repro_neg1025.log" 2>&1
  a=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/f256_runs/addp10_f0_42.json"); b=$(python3 "$SC/metrics_of.py" outputs/repro-neg1025-addp10-f0/final_metrics.json)
  rm -f outputs/repro-neg1025-addp10-f0/*.pth
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение: совпало побитно ($b)"; touch "$OUT/ready"; else echo "воспроизведение: РАСХОЖДЕНИЕ [$a] → [$b]"; echo NEG_FAIL; fi
  exit 0
fi
while [ ! -f "$OUT/ready" ]; do sleep 20; done
for s in 42 43 44; do for f in 0 1 2 3 4; do for arm in add notext shufadd; do for mode in warm hard; do
  key="${arm}_${mode}_f${f}_s${s}"
  grep -q final_metrics "$OUT/$key.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$key" 2>/dev/null || continue
  wait_gpu; echo "=== $key $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/neg1025_$key.yaml" > "$OUT/$key.json" 2>&1 || echo "  ОШИБКА"
  rm -f "outputs/neg1025-$arm-$mode-f$f-s$s/best_model.pth" "outputs/neg1025-$arm-$mode-f$f-s$s/last_model.pth"
done; done; done; done
echo STREAM_DONE
