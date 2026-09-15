#!/bin/sh
# Что мешает полному обучению: по одному отличию от режима 64 троек. RobotCar без текста, фолды 0 и 3, сид 42,
# lr 3e-5, 15 эпох без ранней остановки. Сравнение — lrs_notext_f*_lr3.0e-5 (тот же режим без изменений).
# $1 = gate | stream
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/abl_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/abl; mkdir -p "$OUT" "$KEEP"
if [ "$1" = gate ]; then
  for f in 0 3; do
    b=configs/lrs_notext_f${f}_lr3.0e-5.yaml
    sed -e "s/^\(  experiment_name:\).*/\1 abl-drop0-f$f/" -e "s/^\(  proj_dim: 256\)$/\1\n  dropout: 0.0/" $b > configs/abl_drop0_f$f.yaml
    sed -e "s/^\(  experiment_name:\).*/\1 abl-rand-f$f/" -e "s/^\(  learning_rate:.*\)$/\1\n  miner: random/" $b > configs/abl_rand_f$f.yaml
    sed -e "s/^\(  experiment_name:\).*/\1 abl-b64-f$f/" -e "s/^\(  batch_size:\).*/\1 64/" $b > configs/abl_b64_f$f.yaml
  done
  sed "s/^\(  experiment_name:\).*/\1 repro-abl-notext-f0/" configs/f256_notext_f0_s42.yaml > configs/repro_abl_notext_f0.yaml
  $UV run tgseqloc train -c configs/repro_abl_notext_f0.yaml > "$SC/repro_abl.log" 2>&1
  a=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/f256_runs/notext_f0_42.json"); b=$(python3 "$SC/metrics_of.py" outputs/repro-abl-notext-f0/final_metrics.json)
  rm -f outputs/repro-abl-notext-f0/*.pth
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение: совпало побитно ($b)"; touch "$OUT/ready"; else echo "воспроизведение: РАСХОЖДЕНИЕ [$a] → [$b]"; echo ABL_FAIL; fi
  exit 0
fi
while [ ! -f "$OUT/ready" ]; do sleep 20; done
for cfg in abl_rand_f0 abl_rand_f3 abl_drop0_f0 abl_drop0_f3 abl_b64_f0 abl_b64_f3; do
  grep -q final_metrics "$OUT/$cfg.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$cfg" 2>/dev/null || continue
  n=$(sed -n "s/^  experiment_name: //p" "configs/$cfg.yaml")
  echo "=== $cfg $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/$cfg.yaml" > "$OUT/$cfg.json" 2>&1 || echo "  ОШИБКА"
  mkdir -p "$KEEP/$n" && cp "outputs/$n/best_model.pth" "outputs/$n/final_metrics.json" "$KEEP/$n/" 2>/dev/null
  rm -f "outputs/$n/best_model.pth" "outputs/$n/last_model.pth"
done
echo STREAM_DONE
