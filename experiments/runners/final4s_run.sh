#!/bin/sh
# Тройки внутри камеры, после текущей очереди final4. $1 = gate | stream.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/final4s_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/final4s; mkdir -p "$OUT" "$KEEP"
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
if [ "$1" = gate ]; then
  # ждём, пока досчитается вариант без текста текущего эксперимента
  while [ "$(cat $SC/final4_nt_a.log $SC/final4_nt_b.log $SC/final4_nt_c.log 2>/dev/null | grep -c STREAM_DONE)" -lt 3 ]; do sleep 120; done
  echo "текущая очередь завершена $(date +%H:%M)"
  git merge --ff-only feature/same-camera-triplets || { echo "НЕ СЛИЛОСЬ"; echo FINAL_FAIL; exit 1; }
  git log --oneline -1
  python3 experiments/generators/setup_final_samecam.py || { echo FINAL_FAIL; exit 1; }
  for pair in "f256_notext_f0_s42 f256_runs/notext_f0_42.json" "f256_addp10_f0_s42 f256_runs/addp10_f0_42.json"; do
    set -- $pair
    sed "s/^\(  experiment_name:\).*/\1 repro-samecam-$1/" "configs/$1.yaml" > "configs/repro_samecam_$1.yaml"
    $UV run tgseqloc train -c "configs/repro_samecam_$1.yaml" > "$SC/repro_samecam_$1.log" 2>&1
    a=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/$2"); b=$(python3 "$SC/metrics_of.py" "outputs/repro-samecam-$1/final_metrics.json")
    rm -f outputs/repro-samecam-$1/*.pth
    if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение $1: совпало побитно ($b)"; else echo "воспроизведение $1: РАСХОЖДЕНИЕ [$a] → [$b]"; echo FINAL_FAIL; exit 1; fi
  done
  touch "$OUT/ready"; exit 0
fi
while [ ! -f "$OUT/ready" ]; do sleep 60; done
for s in ${FINAL_SEEDS:-42}; do for f in 0 1 2 3 4; do for arm in add notext; do
  key="${arm}_f${f}_s${s}"; name="final4s-$arm-f$f-s$s"
  grep -q final_metrics "$OUT/$key.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$key" 2>/dev/null || continue
  wait_gpu; echo "=== $key $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/final4s_$key.yaml" > "$OUT/$key.json" 2>&1 || echo "  ОШИБКА $key"
  if [ "$s" = 42 ]; then mkdir -p "$KEEP/$name" && cp "outputs/$name/best_model.pth" "outputs/$name/final_metrics.json" "$KEEP/$name/" 2>/dev/null; fi
  rm -f "outputs/$name/best_model.pth" "outputs/$name/last_model.pth"
done; done; done
echo STREAM_DONE
