#!/bin/sh
# V4RL в финальной конфигурации, сид 42: маски динамики → подготовка трёх вариантов → проверки → три прогона.
SC="$HOME/tgseqloc_data/scripts"; OUT="$HOME/tgseqloc_data/v4final_runs"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
KEEP=/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints/v4final; mkdir -p "$OUT" "$KEEP"
fail() { echo "$1"; echo V4_FAIL; exit 1; }
python3 experiments/generators/setup_v4rl_final.py || fail "генератор"
echo "=== маски динамики $(date +%H:%M) ==="
$UV run tgseqloc stage segmentation -c configs/v4final_add_s42.yaml > "$SC/v4final_seg.log" 2>&1 || { tail -20 "$SC/v4final_seg.log"; fail "сегментация"; }
tail -3 "$SC/v4final_seg.log"
S="$HOME/tgseqloc_data/rv4_m2f_qwen/v4rl/stages/segmentation"
for r in rv4_m2f_notext rv4_m2f_shuf_qwen; do mkdir -p "$HOME/tgseqloc_data/$r/v4rl/stages" && cp -r "$S" "$HOME/tgseqloc_data/$r/v4rl/stages/"; done
echo "  масок: $(find $S -name "*.json" | wc -l)"
echo "=== подготовка $(date +%H:%M) ==="
for arm in add notext shufadd; do
  $UV run tgseqloc prepare -c "configs/v4final_${arm}_s42.yaml" > "$SC/v4final_prep_$arm.log" 2>&1 || { tail -20 "$SC/v4final_prep_$arm.log"; fail "подготовка $arm"; }
  echo "  $arm готов $(date +%H:%M)"
done
python3 - <<'PY' || fail "проверка разбиений"
import json
from pathlib import Path
D = Path.home() / "tgseqloc_data"
s = {r: json.load(open(D / r / "v4rl/mappings/temporal_split.json")) for r in ("rv4_m2f_qwen", "rv4_m2f_notext", "rv4_m2f_shuf_qwen")}
a = s["rv4_m2f_qwen"]
for r, x in s.items():
    print("  %s: база %d, запросы %d, train %d, val %d (исключено %d), test %d (исключено %d), guard=%s" % (r, len(x["database_paths"]), len(x["query_paths"]),
          len(x["train_query_indices"]), len(x["validation_query_indices"]), len(x.get("excluded_validation_indices", [])),
          len(x["test_query_indices"]), len(x.get("excluded_test_indices", [])), x.get("split_guard")))
same = all(x[k] == a[k] for x in s.values() for k in ("train_query_indices", "validation_query_indices", "test_query_indices", "database_paths", "query_paths"))
print("  разбиения совпадают:", same)
raise SystemExit(0 if same and a.get("split_guard") == "ground_truth_overlap" else 1)
PY
for pair in "c10_real_s42 $HOME/TGSeqLoc/outputs/c10-real-s42/final_metrics.json" "f256_addp10_f0_s42 $HOME/tgseqloc_data/f256_runs/addp10_f0_42.json"; do
  set -- $pair
  sed "s/^\(  experiment_name:\).*/\1 repro-v4final-$1/" "configs/$1.yaml" > "configs/repro_v4final_$1.yaml"
  $UV run tgseqloc train -c "configs/repro_v4final_$1.yaml" > "$SC/repro_v4final_$1.log" 2>&1
  a=$(python3 "$SC/metrics_of.py" "$2"); b=$(python3 "$SC/metrics_of.py" "outputs/repro-v4final-$1/final_metrics.json")
  rm -f outputs/repro-v4final-$1/*.pth
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "воспроизведение $1: совпало побитно ($b)"; else fail "воспроизведение $1: РАСХОЖДЕНИЕ [$a] → [$b]"; fi
done
echo "=== прогоны $(date +%H:%M) ==="
for arm in add notext shufadd; do
  ( $UV run tgseqloc train -c "configs/v4final_${arm}_s42.yaml" > "$OUT/${arm}_s42.json" 2>&1 || echo "  ОШИБКА $arm"
    n="v4final-$arm-s42"; mkdir -p "$KEEP/$n" && cp "outputs/$n/best_model.pth" "outputs/$n/final_metrics.json" "$KEEP/$n/" 2>/dev/null
    rm -f "outputs/$n/best_model.pth" "outputs/$n/last_model.pth"; echo "  $arm готов $(date +%H:%M)" ) &
  sleep 20
done
wait
echo V4_DONE
