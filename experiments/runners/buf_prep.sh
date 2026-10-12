#!/bin/sh
# Подготовка к буферной зоне: координаты в разбиениях фолдов + проверка, что ничего не сдвинулось.
SC="$HOME/tgseqloc_data/scripts"; UV="$HOME/.local/bin/uv"; cd "$HOME/TGSeqLoc" || exit 1
for fold in 0 1 2 3 4; do
  for arm in notext real_qwen; do
    $UV run tgseqloc prepare -c "configs/cv_${arm}_f${fold}_s42.yaml" > "$SC/bufprep_${arm}_f${fold}.log" 2>&1 \
      || { echo "ОШИБКА подготовки $arm фолд $fold"; tail -3 "$SC/bufprep_${arm}_f${fold}.log"; echo PREP_FAIL; exit 1; }
  done
done
python3 - <<'PY' || { echo PREP_FAIL; exit 1; }
import json, sys
from pathlib import Path
D = Path.home() / "tgseqloc_data"; bad = 0
for arm, root in (("notext", "rcg_notext"), ("real_qwen", "rcgf_real_qwen")):
    for f in range(5):
        new = json.load(open(D / f"{root}/robotcar/mappings/split_fold{f}.json"))
        old = json.load(open(D / f"cv_runs/split_{arm}_f{f}.json"))
        same = all(new[k] == old[k] for k in ("train_query_indices", "validation_query_indices", "test_query_indices", "query_paths", "database_paths"))
        has = len(new.get("query_positions", {})) > 0 and len(new.get("database_positions", {})) > 0
        print("  %-15s фолд %d: части %s, координаты %s (запросов %d, базы %d)" % (arm, f, "те же" if same else "ИЗМЕНИЛИСЬ", "есть" if has else "НЕТ", len(new.get("query_positions", {})), len(new.get("database_positions", {}))))
        bad += not (same and has)
sys.exit(1 if bad else 0)
PY
sed "s/^\(  experiment_name:\).*/\1 repro-buf-cv-f0/" configs/add_rc_p0_f0_s42.yaml > configs/repro_buf_cv_f0.yaml
$UV run tgseqloc train -c configs/repro_buf_cv_f0.yaml > "$SC/repro_buf.log" 2>&1
new=$(python3 "$SC/metrics_of.py" outputs/repro-buf-cv-f0/final_metrics.json)
ref=$(python3 "$SC/metrics_of.py" "$HOME/tgseqloc_data/add_runs/rc_p0_f0_42.json")
rm -f outputs/repro-buf-cv-f0/best_model.pth outputs/repro-buf-cv-f0/last_model.pth
if [ -n "$new" ] && [ "$new" = "$ref" ]; then echo "  воспроизведение аддитивного прогона: совпало побитно ($new)"; echo PREP_OK; else echo "  РАСХОЖДЕНИЕ: было [$ref], стало [$new]"; echo PREP_FAIL; fi
