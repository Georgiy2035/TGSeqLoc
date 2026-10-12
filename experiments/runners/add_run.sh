#!/bin/sh
# Аддитивный текст. Сначала доказываем, что прежние результаты воспроизводятся
# побитно после правки кода; при любом расхождении дальше не идём.
SC="$HOME/tgseqloc_data/scripts"
OUT="$HOME/tgseqloc_data/add_runs"; mkdir -p "$OUT"
UV="$HOME/.local/bin/uv"
cd "$HOME/TGSeqLoc" || exit 1

wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 1500 ] && return 0; sleep 60; done; }
# Диск заполнен: чекпоинт не нужен — прогоны детерминированы, а метрики и выдача остаются.
clean() { rm -f "outputs/$1/best_model.pth" "outputs/$1/last_model.pth"; }

echo "=== 1. возвращаю разбиение v2 в корни, перезаписанные кросс-валидацией $(date +%H:%M) ==="
for arm in rcg_notext rcgf_real_qwen rcgf_shuffled_qwen; do
    $UV run tgseqloc prepare -c "configs/v2_${arm}_s42.yaml" > "$SC/restore_${arm}.log" 2>&1 \
        || { echo "  ОШИБКА восстановления $arm"; tail -3 "$SC/restore_${arm}.log"; echo REPRO_FAIL; exit 1; }
done
python3 - <<'PY' || { echo REPRO_FAIL; exit 1; }
import json, sys
from pathlib import Path
bad = 0
for arm in ("rcg_notext", "rcgf_real_qwen", "rcgf_shuffled_qwen"):
    s = json.load(open(Path.home() / f"tgseqloc_data/{arm}/robotcar/mappings/temporal_split.json"))
    ok = "split_fold" not in s and len(s["test_query_indices"]) == 532 and len(s["validation_query_indices"]) == 257
    print("  %-20s тест %d, валидация %d, фолд %s — %s" % (arm, len(s["test_query_indices"]),
          len(s["validation_query_indices"]), s.get("split_fold"), "ок" if ok else "НЕ ТО"))
    bad += not ok
sys.exit(1 if bad else 0)
PY

echo "=== 2. воспроизведение прежних результатов $(date +%H:%M) ==="
repro() {   # $1 имя, $2 исходный конфиг, $3 эталон
    sed "s/^\(  experiment_name:\).*/\1 repro-$1/" "configs/$2.yaml" > "configs/repro_$1.yaml"
    wait_gpu
    $UV run tgseqloc train -c "configs/repro_$1.yaml" > "$SC/repro_$1.log" 2>&1
    new=$(python3 "$SC/metrics_of.py" "outputs/repro-$1/final_metrics.json")
    ref=$(python3 "$SC/metrics_of.py" "$3")
    clean "repro-$1"
    if [ -n "$new" ] && [ "$new" = "$ref" ]; then
        echo "  $1: совпало побитно — R@1 R@5 R@10 = $new"
    else
        echo "  $1: РАСХОЖДЕНИЕ — было [$ref], стало [$new]"; tail -5 "$SC/repro_$1.log"; return 1
    fi
}
repro v2_qwen v2_rcgf_real_qwen_s42 "$HOME/tgseqloc_data/rcv2_runs/rcgf_real_qwen_42.json" || { echo REPRO_FAIL; exit 1; }
$UV run tgseqloc prepare -c configs/cv_real_qwen_f0_s42.yaml > "$SC/repro_prep_cv.log" 2>&1 || { echo REPRO_FAIL; exit 1; }
repro cv_f0 cv_real_qwen_f0_s42 "$HOME/tgseqloc_data/cv_runs/real_qwen_f0_42.json" || { echo REPRO_FAIL; exit 1; }
repro v4rl c10_real_s42 "$HOME/TGSeqLoc/outputs/c10-real-s42/final_metrics.json" || { echo REPRO_FAIL; exit 1; }
echo REPRO_OK

echo "=== 3. аддитивный текст, RobotCar, кросс-валидация $(date +%H:%M) ==="
for fold in 0 1 2 3 4; do
    $UV run tgseqloc prepare -c "configs/cv_real_qwen_f${fold}_s42.yaml" > "$SC/prep_add_f${fold}.log" 2>&1 \
        || { echo "  ОШИБКА подготовки фолда $fold"; continue; }
    for tag in p0 p05; do
        for seed in 42 43 44 45 46; do
            out="$OUT/rc_${tag}_f${fold}_${seed}.json"
            grep -q final_metrics "$out" 2>/dev/null && continue
            wait_gpu
            echo "=== RobotCar ${tag} фолд ${fold} сид ${seed} $(date +%H:%M) ==="
            $UV run tgseqloc train -c "configs/add_rc_${tag}_f${fold}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
            clean "add-rc-${tag}-f${fold}-s${seed}"
        done
    done
done

echo "=== 4. аддитивный текст, V4RL $(date +%H:%M) ==="
for tag in p0 p05; do
    for seed in 42 43 44 45 46 47 48 49 50 51; do
        out="$OUT/v4rl_${tag}_${seed}.json"
        grep -q final_metrics "$out" 2>/dev/null && continue
        wait_gpu
        echo "=== V4RL ${tag} сид ${seed} $(date +%H:%M) ==="
        $UV run tgseqloc train -c "configs/add_v4rl_${tag}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
        clean "add-v4rl-${tag}-s${seed}"
    done
done
echo ADD_DONE
