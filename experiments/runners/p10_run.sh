#!/bin/sh
# Схема 10/25 (позитивы до 10 м, промежуток 10–25 м не участвует): Qwen слагаемое и без текста, 256/256, 5×5.
# Аргумент "after" — ждать окончания второго потока (буфер 35/75), иначе стартовать сразу.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/f256_runs"; UV="$HOME/.local/bin/uv"
cd "$HOME/TGSeqLoc" || exit 1
if [ "$1" = "after" ]; then
  while [ -f "$SC/radii.pid" ] && kill -0 "$(cat "$SC/radii.pid")" 2>/dev/null; do sleep 60; done
  echo "второй поток завершён $(date +%H:%M)"
fi
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
for fold in 0 1 2 3 4; do
  for arm in addp10 notextp10; do
    for seed in 42 43 44 45 46; do
      out="$OUT/${arm}_f${fold}_${seed}.json"; grep -q final_metrics "$out" 2>/dev/null && continue
      wait_gpu; echo "=== ${arm} фолд ${fold} сид ${seed} $(date +%H:%M:%S) ==="
      $UV run tgseqloc train -c "configs/f256_${arm}_f${fold}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
      rm -f "outputs/f256-${arm}-f${fold}-s${seed}/best_model.pth" "outputs/f256-${arm}-f${fold}-s${seed}/last_model.pth"
    done
  done
done
echo P10_DONE
