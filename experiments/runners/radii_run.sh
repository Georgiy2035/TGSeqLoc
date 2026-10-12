#!/bin/sh
# Второй поток: буфер 35 и 75 м для текстовой ветки (Qwen, слагаемое, 256/256), кросс-валидация 5×5.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/f256_runs"; UV="$HOME/.local/bin/uv"
cd "$HOME/TGSeqLoc" || exit 1
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
for fold in 0 1 2 3 4; do
  for r in 35 75; do
    for seed in 42 43 44 45 46; do
      out="$OUT/addbuf${r}_f${fold}_${seed}.json"; grep -q final_metrics "$out" 2>/dev/null && continue
      wait_gpu; echo "=== буфер ${r} фолд ${fold} сид ${seed} $(date +%H:%M:%S) ==="
      $UV run tgseqloc train -c "configs/f256_addbuf${r}_f${fold}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
      rm -f "outputs/f256-addbuf${r}-f${fold}-s${seed}/best_model.pth" "outputs/f256-addbuf${r}-f${fold}-s${seed}/last_model.pth"
    done
  done
done
echo RADII_DONE
