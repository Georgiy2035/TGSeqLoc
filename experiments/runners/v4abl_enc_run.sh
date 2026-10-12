#!/bin/sh
# A4: текстовый энкодер на V4RL, сид 42. $1 = gate | stream.
# gate: конфиги, маски динамики в новые корни, подготовка (энкодеры считают эмбеддинги строк), проверка разбиений.
# stream: 5 прогонов, по 3 одновременно; чекпойнты — на общий диск.
OUT="$HOME/tgseqloc_data/v4abl_enc_runs"; BIN="$HOME/TGSeqLoc/.venv/bin"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/v4abl_enc; mkdir -p "$OUT/logs" "$KEEP"
SEG="$HOME/tgseqloc_data/abl_v4_qwen/v4rl/stages/segmentation"
ENCODERS="cn384 e5small e5base bgem3 labse"
if [ "$1" = gate ]; then
  "$BIN/python" experiments/generators/setup_v4abl_enc.py || { echo ENC_FAIL; exit 1; }
  for e in $ENCODERS; do
    root=$(grep "^  prepared_root:" "configs/v4abl_enc_${e}_s42.yaml" | awk '{print $2}')
    mkdir -p "$root/v4rl/stages"
    [ -e "$root/v4rl/stages/segmentation" ] || ln -s "$SEG" "$root/v4rl/stages/segmentation"
    "$BIN/tgseqloc" prepare -c "configs/v4abl_enc_${e}_s42.yaml" > "$OUT/logs/prepare_$e.log" 2>&1 \
      || { echo "ОШИБКА подготовки $e"; tail -5 "$OUT/logs/prepare_$e.log"; echo ENC_FAIL; exit 1; }
    echo "  $e готов $(date +%H:%M)"
  done
  "$BIN/python" - <<'PY' || { echo ENC_FAIL; exit 1; }
import json
from pathlib import Path
D = Path.home() / "tgseqloc_data"
roots = ["abl_v4_qwen"] + [f"abl_v4_enc_{e}" for e in ("cn384", "e5small", "e5base", "bgem3", "labse")]
s = {r: json.load(open(D / r / "v4rl/mappings/temporal_split.json")) for r in roots}
a = s[roots[0]]
keys = ("train_query_indices", "validation_query_indices", "test_query_indices", "database_paths", "query_paths")
same = all(x[k] == a[k] for x in s.values() for k in keys)
dims = {r: json.load(open(D / r / "v4rl/manifest.json"))["text_embedding_dim"] for r in roots}
print("  разбиения совпадают:", same, "| размерности:", dims)
raise SystemExit(0 if same else 1)
PY
  echo ENC_GATE_DONE
elif [ "$1" = stream ]; then
  for e in $ENCODERS; do echo "$e"; done | xargs -P 3 -I{} sh -c '
    "'"$BIN"'/tgseqloc" train -c "configs/v4abl_enc_{}_s42.yaml" > "'"$OUT"'/v4abl_enc_{}_s42.json" 2>&1 \
      && echo "  {} готов $(date +%H:%M)" || echo "  ОШИБКА {}"
    name="v4abl-enc_{}-s42"
    mkdir -p "'"$KEEP"'/$name" && cp "outputs/$name/best_model.pth" "outputs/$name/final_metrics.json" "configs/v4abl_enc_{}_s42.yaml" "'"$KEEP"'/$name/" 2>/dev/null
    rm -f "outputs/$name/best_model.pth" "outputs/$name/last_model.pth"'
  echo ENC_DONE
fi
