#!/bin/sh
# Скан lr на RobotCar без текста: фолды 0 и 3, сид 42, 15 эпох без ранней остановки (видна вся кривая).
OUT="$HOME/tgseqloc_data/lr_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/lrscan; mkdir -p "$OUT" "$KEEP"
for lr in 1.0e-3 3.0e-4 1.0e-4 3.0e-5; do for f in 0 3; do
  key="f${f}_lr$lr"; n="lrs-notext-f$f-lr$lr"
  grep -q final_metrics "$OUT/$key.json" 2>/dev/null && continue
  mkdir "$OUT/.lock_$key" 2>/dev/null || continue
  echo "=== $key $(date +%H:%M) ==="
  $UV run tgseqloc train -c "configs/lrs_notext_f${f}_lr$lr.yaml" > "$OUT/$key.json" 2>&1 || echo "  ОШИБКА"
  mkdir -p "$KEEP/$n" && cp "outputs/$n/best_model.pth" "outputs/$n/final_metrics.json" "$KEEP/$n/" 2>/dev/null
  rm -f "outputs/$n/best_model.pth" "outputs/$n/last_model.pth"
done; done
echo STREAM_DONE
