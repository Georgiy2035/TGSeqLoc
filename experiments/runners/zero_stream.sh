#!/bin/sh
# Поток прогонов с нулевым стартом; прогон берёт блокировку, так что два потока не пересекаются.
# Порядок: RobotCar набор строк → V4RL → RobotCar перемешанный контроль.
# У сида 42 лучший чекпойнт уезжает на внешний диск — для замера вклада текста в дескриптор.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/zero_runs"; UV="$HOME/.local/bin/uv"
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/zero
cd "$HOME/TGSeqLoc" || exit 1; mkdir -p "$OUT" "$KEEP"
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
jobs() {
  for f in 0 1 2 3 4; do for s in 42 43 44 45 46; do echo "set0_f${f}_${s} f256_set0_f${f}_s${s} f256-set0-f${f}-s${s}"; done; done
  for s in 42 43 44 45 46 47 48 49 50 51; do echo "v4rl_set0_${s} v256_set0_s${s} v256-set0-s${s}"; done
  for f in 0 1 2 3 4; do for s in 42 43 44 45 46; do echo "shufset0_f${f}_${s} f256_shufset0_f${f}_s${s} f256-shufset0-f${f}-s${s}"; done; done
}
keep() {
  mkdir -p "$KEEP/$2" && cp "outputs/$2/best_model.pth" "outputs/$2/final_metrics.json" "$KEEP/$2/" && echo "  чекпойнт сохранён: $2"
}
jobs | while read -r key cfg name; do
  out="$OUT/${key}.json"
  if grep -q final_metrics "$out" 2>/dev/null; then
    # Прогон сида 42, прошедший до сохранения чекпойнтов, повторяется отдельно и сверяется побитно.
    case "$key" in *_42) ;; *) continue ;; esac
    [ -f "$KEEP/$name/best_model.pth" ] && continue
    mkdir "$OUT/.lock_keep_${key}" 2>/dev/null || continue
    wait_gpu; echo "=== ${key} повтор ради чекпойнта $(date +%H:%M) ==="
    $UV run tgseqloc train -c "configs/${cfg}.yaml" > "$OUT/keep_${key}.json" 2>&1 || echo "  ОШИБКА"
    a=$(python3 "$SC/metrics_of.py" "$out"); b=$(python3 "$SC/metrics_of.py" "outputs/${name}/final_metrics.json")
    [ -n "$a" ] && [ "$a" = "$b" ] && { echo "  совпало побитно"; keep "$key" "$name"; } || echo "  РАСХОЖДЕНИЕ [$a] → [$b]"
  else
    mkdir "$OUT/.lock_${key}" 2>/dev/null || continue
    wait_gpu; echo "=== ${key} $(date +%H:%M) ==="
    $UV run tgseqloc train -c "configs/${cfg}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
    case "$key" in *_42) keep "$key" "$name" ;; esac
  fi
  rm -f "outputs/${name}/best_model.pth" "outputs/${name}/last_model.pth"
done
echo STREAM_DONE
