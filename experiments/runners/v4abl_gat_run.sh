#!/bin/sh
# A6/A7: глубина GAT и размер дескриптора, сид 42. Подготовка не нужна — данные общие с эталоном.
OUT="$HOME/tgseqloc_data/v4abl_gat_runs"; BIN="$HOME/TGSeqLoc/.venv/bin"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/v4abl_gat; mkdir -p "$OUT" "$KEEP"
"$BIN/python" experiments/generators/setup_v4abl_gat.py || { echo GAT_FAIL; exit 1; }
for a in l2 d64 d128 d512; do echo "$a"; done | xargs -P 4 -I{} sh -c '
  "'"$BIN"'/tgseqloc" train -c "configs/v4abl_gat_{}_s42.yaml" > "'"$OUT"'/v4abl_gat_{}_s42.json" 2>&1 \
    && echo "  {} готов $(date +%H:%M)" || echo "  ОШИБКА {}"
  name="v4abl-gat_{}-s42"
  mkdir -p "'"$KEEP"'/$name" && cp "outputs/$name/best_model.pth" "outputs/$name/final_metrics.json" "configs/v4abl_gat_{}_s42.yaml" "'"$KEEP"'/$name/" 2>/dev/null
  rm -f "outputs/$name/last_model.pth"'
echo GAT_DONE
