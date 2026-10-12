#!/bin/sh
# Перепрогон на наборе строк. $1 — список армов, $2 — сколько параллельно (по умолчанию 4).
OUT="$HOME/tgseqloc_data/v4abl_set_runs"; BIN="$HOME/TGSeqLoc/.venv/bin"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/v4abl_set; mkdir -p "$OUT" "$KEEP"
ARMS="${1:-ref shuf l2 d128 d512}"; PAR="${2:-4}"
"$BIN/python" experiments/generators/setup_v4abl_set.py || { echo SET_FAIL; exit 1; }
for a in $ARMS; do echo "$a"; done | xargs -P "$PAR" -I{} sh -c '
  "'"$BIN"'/tgseqloc" train -c "configs/v4abl_set_{}_s42.yaml" > "'"$OUT"'/v4abl_set_{}_s42.json" 2>&1 \
    && echo "  {} готов $(date +%H:%M)" || echo "  ОШИБКА {}"
  name="v4abl-set_{}-s42"
  mkdir -p "'"$KEEP"'/$name" && cp "outputs/$name/best_model.pth" "outputs/$name/final_metrics.json" "configs/v4abl_set_{}_s42.yaml" "'"$KEEP"'/$name/" 2>/dev/null
  rm -f "outputs/$name/last_model.pth"'
echo SET_DONE
