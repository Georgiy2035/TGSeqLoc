#!/bin/sh
# Размер GAT на RobotCar. Данные уже подготовлены (размер модели в отпечаток
# подготовки не входит), поэтому все сиды идут через train.
cd "$HOME/TGSeqLoc"
OUT="$HOME/tgseqloc_data/gsz_runs"; mkdir -p "$OUT"
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 1500 ] && return 0; sleep 60; done; }
for v in h256p64 h256p256; do
  for arm in notext real_paddle real_qwen; do
    for seed in 42 43 44 45 46 47 48 49 50 51; do
      out="$OUT/${v}_${arm}_${seed}.json"
      grep -q final_metrics "$out" 2>/dev/null && continue
      wait_gpu
      echo "=== ${v} ${arm} сид ${seed} $(date +%H:%M) ==="
      ~/.local/bin/uv run tgseqloc train -c "configs/gsz_${v}_${arm}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
      rm -f "outputs/gsz-${v}-${arm}-s${seed}/last_model.pth"
    done
  done
done
echo GSZ_DONE
