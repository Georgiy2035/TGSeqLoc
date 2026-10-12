#!/bin/sh
# Поток прогонов V4RL 256/256 с блокировками: каждый прогон берёт ровно один поток.
# $1 — порядок: "forward" (набор строк → узел → без текста, сиды по возрастанию) или "reverse".
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/v256_runs"; UV="$HOME/.local/bin/uv"
cd "$HOME/TGSeqLoc" || exit 1; mkdir -p "$OUT"
if [ "$1" = "reverse" ]; then ARMS="notext node set"; SEEDS="51 50 49 48 47 46 45 44 43 42"; else ARMS="set node notext"; SEEDS="42 43 44 45 46 47 48 49 50 51"; fi
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
for arm in $ARMS; do
  for seed in $SEEDS; do
    out="$OUT/${arm}_${seed}.json"
    grep -q final_metrics "$out" 2>/dev/null && continue
    mkdir "$OUT/.lock_${arm}_${seed}" 2>/dev/null || continue
    wait_gpu; echo "=== V4RL ${arm} сид ${seed} $(date +%H:%M) ==="
    $UV run tgseqloc train -c "configs/v256_${arm}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
    rm -f "outputs/v256-${arm}-s${seed}/best_model.pth" "outputs/v256-${arm}-s${seed}/last_model.pth"
  done
done
echo STREAM_DONE
