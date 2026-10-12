#!/bin/sh
# Абляция OCR на V4RL, новое разбиение (валидация — два блока). $1 = gate | stream.
# gate: конфиги, маски в новые корни, подготовка, проверка, что разбиения совпадают.
# stream: побитный повтор v4final-add-s42 на текущем коде и 4 варианта, по 3 одновременно.
OUT="$HOME/tgseqloc_data/v4abl_runs"; BIN="$HOME/TGSeqLoc/.venv/bin"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/v4abl; mkdir -p "$OUT/logs" "$KEEP"
ARMS="add_qwen notext add_paddle add_easyocr"
if [ "$1" = gate ]; then
  "$BIN/python" experiments/generators/setup_v4abl.py || { echo ABL_FAIL; exit 1; }
  S="$HOME/tgseqloc_data/rv4_m2f_qwen/v4rl/stages/segmentation"
  for arm in $ARMS; do
    root=$(grep "^  prepared_root:" "configs/v4abl_${arm}_s42.yaml" | awk '{print $2}')
    mkdir -p "$root/v4rl/stages" && [ -d "$root/v4rl/stages/segmentation" ] || cp -r "$S" "$root/v4rl/stages/"
    "$BIN/tgseqloc" prepare -c "configs/v4abl_${arm}_s42.yaml" > "$OUT/logs/prepare_$arm.log" 2>&1 \
      || { echo "ОШИБКА подготовки $arm"; tail -5 "$OUT/logs/prepare_$arm.log"; echo ABL_FAIL; exit 1; }
    echo "  $arm готов $(date +%H:%M)"
  done
  "$BIN/python" - <<'PY' || { echo ABL_FAIL; exit 1; }
import json
from pathlib import Path
D = Path.home() / "tgseqloc_data"
roots = ("abl_v4_qwen", "abl_v4_notext", "abl_v4_paddle", "abl_v4_easyocr")
s = {r: json.load(open(D / r / "v4rl/mappings/temporal_split.json")) for r in roots}
a = s[roots[0]]
keys = ("train_query_indices", "validation_query_indices", "test_query_indices", "database_paths", "query_paths")
same = all(x[k] == a[k] for x in s.values() for k in keys)
print("  train %d, val %d (блоки %s), test %d; из обучения выброшено %d, из валидации %d; guard=%s" % (
    len(a["train_query_indices"]), len(a["validation_query_indices"]), a["validation_blocks"], len(a["test_query_indices"]),
    len(a["excluded_train_indices"]), len(a["excluded_validation_indices"]), a["split_guard"]))
print("  разбиения совпадают:", same)
raise SystemExit(0 if same and a["split_guard"] == "ground_truth_blocks" else 1)
PY
  echo ABL_GATE_DONE
elif [ "$1" = stream ]; then
  # побитный повтор эталона старого протокола на текущем коде
  sed -e "s/^  experiment_name: .*/  experiment_name: repro-v4final-add-s42/" configs/v4final_add_s42.yaml > configs/repro_now_v4final_add_s42.yaml
  { echo "repro_now_v4final_add_s42"; for arm in $ARMS; do echo "v4abl_${arm}_s42"; done; } | xargs -P 3 -I{} sh -c '
    name=$(grep "^  experiment_name:" "configs/{}.yaml" | awk "{print \$2}")
    "'"$BIN"'/tgseqloc" train -c "configs/{}.yaml" > "'"$OUT"'/{}.json" 2>&1 && echo "  {} готов $(date +%H:%M)" || echo "  ОШИБКА {}"
    mkdir -p "'"$KEEP"'/$name" && cp "outputs/$name/best_model.pth" "outputs/$name/final_metrics.json" "'"$KEEP"'/$name/" 2>/dev/null
    rm -f "outputs/$name/best_model.pth" "outputs/$name/last_model.pth"'
  echo ABL_DONE
fi
