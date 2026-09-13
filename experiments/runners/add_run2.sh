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

echo "=== 1. возвращаю разбиения из сохранённых копий (внешний диск не читается, подготовка невозможна) $(date +%H:%M) ==="
python3 - <<'PY' || { echo REPRO_FAIL; exit 1; }
import json, shutil, sys
from pathlib import Path
D = Path.home() / "tgseqloc_data"
src = D / "rcgf_real_paddle/robotcar/mappings/temporal_split.json"
s = json.load(open(src))
if "split_fold" in s or len(s["test_query_indices"]) != 532 or len(s["validation_query_indices"]) != 257:
    print("  эталонное разбиение v2 не то"); sys.exit(1)
for arm in ("rcg_notext", "rcgf_real_qwen", "rcgf_shuffled_qwen"):
    dst = D / f"{arm}/robotcar/mappings/temporal_split.json"
    here = json.load(open(dst))
    if here["query_paths"] != s["query_paths"] or here["database_paths"] != s["database_paths"]:
        print("  пути кадров в", arm, "не совпадают с эталоном"); sys.exit(1)
    shutil.copyfile(src, dst)
    print("  %-20s разбиение v2 возвращено (тест 532, валидация 257)" % arm)
for fold in range(5):
    c = json.load(open(D / f"cv_runs/split_real_qwen_f{fold}.json"))
    if c.get("split_fold") != fold or c["query_paths"] != s["query_paths"]:
        print("  копия фолда", fold, "не та"); sys.exit(1)
    shutil.copyfile(D / f"cv_runs/split_real_qwen_f{fold}.json",
                    D / f"rcgf_real_qwen/robotcar/mappings/split_fold{fold}.json")
    print("  фолд %d: разбиение взято из копии кросс-валидации, тест %d" % (fold, len(c["test_query_indices"])))
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
repro cv_f0 cv_real_qwen_f0_s42 "$HOME/tgseqloc_data/cv_runs/real_qwen_f0_42.json" || { echo REPRO_FAIL; exit 1; }
repro v4rl c10_real_s42 "$HOME/TGSeqLoc/outputs/c10-real-s42/final_metrics.json" || { echo REPRO_FAIL; exit 1; }
echo REPRO_OK

echo "=== 3. аддитивный текст, RobotCar, кросс-валидация $(date +%H:%M) ==="
for fold in 0 1 2 3 4; do
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
