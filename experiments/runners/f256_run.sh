#!/bin/sh
# RobotCar на 256/256. Ждёт окончания текущей очереди (V4RL на 128/64).
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/f256_runs"; mkdir -p "$OUT"
UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
while [ -f "$SC/add.pid" ] && kill -0 "$(cat "$SC/add.pid")" 2>/dev/null; do sleep 60; done
echo "текущая очередь завершена $(date +%H:%M)"
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
for fold in 0 1 2 3 4; do
  for arm in notext add shufadd; do
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
echo F256_DONE
