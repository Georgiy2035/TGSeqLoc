#!/bin/sh
OUT="$HOME/tgseqloc_data/v4abl_set_runs"; BIN="$HOME/TGSeqLoc/.venv/bin"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/v4abl_set; mkdir -p "$OUT" "$KEEP"
"$BIN/python" experiments/generators/setup_v4abl_set.py >/dev/null || exit 1
"$BIN/python" experiments/generators/setup_seeds.py || exit 1
for c in v4abl_set_d64_s42 v4abl_set_ref_s43 v4abl_set_ref_s44 v4abl_add_ref_s43 v4abl_add_ref_s44; do echo "$c"; done | xargs -P 5 -I{} sh -c '
  "'"$BIN"'/tgseqloc" train -c "configs/{}.yaml" > "'"$OUT"'/{}.json" 2>&1 && echo "  {} готов $(date +%H:%M)" || echo "  ОШИБКА {}"'
echo WAVE_B_DONE
