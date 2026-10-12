#!/bin/sh
# Проверка «сцена + набор строк». Стартует после прогона с буферной зоной,
# только если правка закоммичена и прежние прогоны воспроизводятся побитно.
SC="$HOME/tgseqloc_data/scripts"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
while [ -f "$SC/buf256.pid" ] && kill -0 "$(cat "$SC/buf256.pid")" 2>/dev/null; do sleep 60; done
[ -f "$SC/set_ready" ] || { echo "правка не закоммичена — не запускаю"; echo SET_FAIL; exit 1; }
echo "прогон с буфером завершён $(date +%H:%M)"
clean() { rm -f "outputs/$1/best_model.pth" "outputs/$1/last_model.pth"; }
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }

echo "=== воспроизведение после правки $(date +%H:%M) ==="
repro() {
  sed "s/^\(  experiment_name:\).*/\1 repro-set-$1/" "configs/$2.yaml" > "configs/repro_set_$1.yaml"
  wait_gpu
  $UV run tgseqloc train -c "configs/repro_set_$1.yaml" > "$SC/repro_set_$1.log" 2>&1
  new=$(python3 "$SC/metrics_of.py" "outputs/repro-set-$1/final_metrics.json"); ref=$(python3 "$SC/metrics_of.py" "$3")
  clean "repro-set-$1"
  if [ -n "$new" ] && [ "$new" = "$ref" ]; then echo "  $1: совпало побитно ($new)"; else echo "  $1: РАСХОЖДЕНИЕ [$ref] → [$new]"; return 1; fi
}
repro node_v4rl c10_real_s42 "$HOME/TGSeqLoc/outputs/c10-real-s42/final_metrics.json" || { echo SET_FAIL; exit 1; }
repro add_rc add_rc_p0_f0_s42 "$HOME/tgseqloc_data/add_runs/rc_p0_f0_42.json" || { echo SET_FAIL; exit 1; }

OUT="$HOME/tgseqloc_data/f256_runs"
for fold in 0 1 2 3 4; do
  for arm in set shufset; do
    for seed in 42 43 44 45 46; do
      out="$OUT/${arm}_f${fold}_${seed}.json"; grep -q final_metrics "$out" 2>/dev/null && continue
      wait_gpu; echo "=== RobotCar ${arm} фолд ${fold} сид ${seed} $(date +%H:%M) ==="
      $UV run tgseqloc train -c "configs/f256_${arm}_f${fold}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
      clean "f256-${arm}-f${fold}-s${seed}"
    done
  done
done
OUT="$HOME/tgseqloc_data/v256_runs"; mkdir -p "$OUT"
for arm in set node notext; do
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    out="$OUT/${arm}_${seed}.json"; grep -q final_metrics "$out" 2>/dev/null && continue
    wait_gpu; echo "=== V4RL ${arm} сид ${seed} $(date +%H:%M) ==="
    $UV run tgseqloc train -c "configs/v256_${arm}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
    clean "v256-${arm}-s${seed}"
  done
done
echo SET_DONE
