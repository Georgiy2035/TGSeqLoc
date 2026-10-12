#!/bin/sh
# Нормализация дескриптора: сначала эпоха 0 и побитная проверка старого прогона, затем поток прогонов.
# $1 = gate | stream
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/nrm_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/nrm; mkdir -p "$OUT" "$KEEP"
if [ "$1" = gate ]; then
  $UV run python "$SC/epoch0.py" 2>&1 | grep "эпоха 0"
  sed "s/^\(  experiment_name:\).*/\1 repro-nrm-notext-f0/" configs/f256_notext_f0_s42.yaml > configs/repro_nrm_notext_f0.yaml
  $UV run tgseqloc train -c configs/repro_nrm_notext_f0.yaml > "$SC/repro_nrm.log" 2>&1
  a=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/f256_runs/notext_f0_42.json"); b=$(python3 "$SC/metrics_of.py" outputs/repro-nrm-notext-f0/final_metrics.json)
  rm -f outputs/repro-nrm-notext-f0/*.pth
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение: совпало побитно ($b)"; touch "$OUT/ready"; else echo "воспроизведение: РАСХОЖДЕНИЕ [$a] → [$b]"; echo NRM_FAIL; fi
  exit 0
fi
while [ ! -f "$OUT/ready" ]; do sleep 20; done
for cfg in nrm_notext_f0_output nrm_notext_f3_output nrm_notext_f0_pooled nrm_notext_f3_pooled nrm_notext_f0_output_lr1e-4 nrm_notext_f3_output_lr1e-4; do
  grep -q final_metrics "$OUT/$cfg.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$cfg" 2>/dev/null || continue
  n=$(sed -n "s/^  experiment_name: //p" "configs/$cfg.yaml")
  echo "=== $cfg $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/$cfg.yaml" > "$OUT/$cfg.json" 2>&1 || echo "  ОШИБКА"
  mkdir -p "$KEEP/$n" && cp "outputs/$n/best_model.pth" "outputs/$n/final_metrics.json" "$KEEP/$n/" 2>/dev/null
  rm -f "outputs/$n/best_model.pth" "outputs/$n/last_model.pth"
done
echo STREAM_DONE
