#!/bin/sh
# GAT 1 слой, 256/256, RobotCar без текста, фолды 0 и 3, сид 42, lr 3e-5, batch 8, dropout 0.1, 40 эпох без ранней остановки.
# hard — hard mining с первой эпохи (контроль); rand — случайные негативы (A); warm — 10 эпох случайных, затем hard (Б).
# $1 = gate | stream
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/g1_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/g1; mkdir -p "$OUT" "$KEEP"
if [ "$1" = gate ]; then
  for f in 0 3; do
    b=configs/lrs_notext_f${f}_lr3.0e-5.yaml
    for v in hard rand warm; do
      sed -e "s/^\(  experiment_name:\).*/\1 g1-$v-f$f/" -e "s/^\(  epochs:\).*/\1 40/" -e "s/^\(  patience:\).*/\1 40/" \
          -e "s/^\(  proj_dim: 256\)$/\1\n  n_layers: 1/" $b > configs/g1_${v}_f$f.yaml
    done
    sed -i "s/^\(  learning_rate:.*\)$/\1\n  miner: random/" configs/g1_rand_f$f.yaml
    sed -i "s/^\(  learning_rate:.*\)$/\1\n  miner_warmup_epochs: 10/" configs/g1_warm_f$f.yaml
  done
  sed "s/^\(  experiment_name:\).*/\1 repro-g1-notext-f0/" configs/f256_notext_f0_s42.yaml > configs/repro_g1_notext_f0.yaml
  $UV run tgseqloc train -c configs/repro_g1_notext_f0.yaml > "$SC/repro_g1.log" 2>&1
  a=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/f256_runs/notext_f0_42.json"); b=$(python3 "$SC/metrics_of.py" outputs/repro-g1-notext-f0/final_metrics.json)
  rm -f outputs/repro-g1-notext-f0/*.pth
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение: совпало побитно ($b)"; touch "$OUT/ready"; else echo "воспроизведение: РАСХОЖДЕНИЕ [$a] → [$b]"; echo G1_FAIL; fi
  exit 0
fi
while [ ! -f "$OUT/ready" ]; do sleep 20; done
for cfg in g1_rand_f0 g1_warm_f0 g1_hard_f0 g1_rand_f3 g1_warm_f3 g1_hard_f3; do
  grep -q final_metrics "$OUT/$cfg.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$cfg" 2>/dev/null || continue
  n=$(sed -n "s/^  experiment_name: //p" "configs/$cfg.yaml")
  echo "=== $cfg $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/$cfg.yaml" > "$OUT/$cfg.json" 2>&1 || echo "  ОШИБКА"
  mkdir -p "$KEEP/$n" && cp "outputs/$n/best_model.pth" "outputs/$n/final_metrics.json" "$KEEP/$n/" 2>/dev/null
  rm -f "outputs/$n/best_model.pth" "outputs/$n/last_model.pth"
done
echo STREAM_DONE
