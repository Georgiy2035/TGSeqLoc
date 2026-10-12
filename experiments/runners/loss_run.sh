#!/bin/sh
# 1) BatchNorm перед проекцией, 40 эпох; 2) InfoNCE (τ=0.07) без нормализации и с ней, 15 эпох. Фолды 0 и 3, сид 42.
# $1 = gate | stream
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/loss_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/loss; mkdir -p "$OUT" "$KEEP"
if [ "$1" = gate ]; then
  for f in 0 3; do
    sed -e "s/^\(  experiment_name:\).*/\1 bn40-notext-f$f/" -e "s/^\(  epochs:\).*/\1 40/" -e "s/^\(  patience:\).*/\1 40/" configs/nrm_notext_f${f}_pooled.yaml > configs/bn40_notext_f$f.yaml
    sed -e "s/^\(  experiment_name:\).*/\1 nce-notext-f$f/" -e "s/^\(  epochs:\).*/\1 15/" -e "s/^\(  patience:\).*/\1 15/" -e "s/^\(  learning_rate:.*\)$/\1\n  loss: infonce\n  temperature: 0.07/" configs/f256_notext_f${f}_s42.yaml > configs/nce_notext_f$f.yaml
    sed -e "s/^\(  experiment_name:\).*/\1 ncebn-notext-f$f/" -e "s/^\(  learning_rate:.*\)$/\1\n  loss: infonce\n  temperature: 0.07/" configs/nrm_notext_f${f}_pooled.yaml > configs/ncebn_notext_f$f.yaml
  done
  sed "s/^\(  experiment_name:\).*/\1 repro-loss-notext-f0/" configs/f256_notext_f0_s42.yaml > configs/repro_loss_notext_f0.yaml
  $UV run tgseqloc train -c configs/repro_loss_notext_f0.yaml > "$SC/repro_loss.log" 2>&1
  a=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/f256_runs/notext_f0_42.json"); b=$(python3 "$SC/metrics_of.py" outputs/repro-loss-notext-f0/final_metrics.json)
  rm -f outputs/repro-loss-notext-f0/*.pth
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение: совпало побитно ($b)"; touch "$OUT/ready"; else echo "воспроизведение: РАСХОЖДЕНИЕ [$a] → [$b]"; echo LOSS_FAIL; fi
  exit 0
fi
while [ ! -f "$OUT/ready" ]; do sleep 20; done
for cfg in nce_notext_f0 nce_notext_f3 ncebn_notext_f0 ncebn_notext_f3 bn40_notext_f0 bn40_notext_f3; do
  grep -q final_metrics "$OUT/$cfg.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$cfg" 2>/dev/null || continue
  n=$(sed -n "s/^  experiment_name: //p" "configs/$cfg.yaml")
  echo "=== $cfg $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/$cfg.yaml" > "$OUT/$cfg.json" 2>&1 || echo "  ОШИБКА"
  mkdir -p "$KEEP/$n" && cp "outputs/$n/best_model.pth" "outputs/$n/final_metrics.json" "$KEEP/$n/" 2>/dev/null
  rm -f "outputs/$n/best_model.pth" "outputs/$n/last_model.pth"
done
echo STREAM_DONE
