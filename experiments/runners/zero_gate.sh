#!/bin/sh
# Нулевой старт текстовой ветки. Сначала побитная проверка, что прежние прогоны
# набора строк воспроизводятся на новом коде, затем два потока zero_stream.sh.
SC="$HOME/tgseqloc_data/scripts"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 2000 ] && return 0; sleep 60; done; }
repro() {
  sed "s/^\(  experiment_name:\).*/\1 repro-zero-$1/" "configs/$2.yaml" > "configs/repro_zero_$1.yaml"
  wait_gpu
  $UV run tgseqloc train -c "configs/repro_zero_$1.yaml" > "$SC/repro_zero_$1.log" 2>&1
  new=$(python3 "$SC/metrics_of.py" "outputs/repro-zero-$1/final_metrics.json"); ref=$(python3 "$SC/metrics_of.py" "$3")
  rm -f "outputs/repro-zero-$1/best_model.pth" "outputs/repro-zero-$1/last_model.pth"
  if [ -n "$new" ] && [ "$new" = "$ref" ]; then echo "  $1: совпало побитно ($new)"; else echo "  $1: РАСХОЖДЕНИЕ [$ref] → [$new]"; return 1; fi
}
echo "=== воспроизведение $(date +%H:%M) ==="
repro set_rc f256_set_f0_s42 "$HOME/tgseqloc_data/f256_runs/set_f0_42.json" &
a=$!
repro set_v4rl v256_set_s42 "$HOME/tgseqloc_data/v256_runs/set_42.json"; rb=$?
wait $a; ra=$?
[ $ra -eq 0 ] && [ $rb -eq 0 ] || { echo ZERO_FAIL; exit 1; }
echo "=== очередь $(date +%H:%M) ==="
setsid nohup sh experiments/runners/zero_stream.sh forward > "$SC/zero_a.log" 2>&1 < /dev/null &
setsid nohup sh experiments/runners/zero_stream.sh forward > "$SC/zero_b.log" 2>&1 < /dev/null &
echo ZERO_STARTED
