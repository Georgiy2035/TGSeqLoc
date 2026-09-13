#!/bin/sh
# Блочная кросс-валидация RobotCar: 5 фолдов × 3 ветки × 5 сидов.
#
# Разметка фолдов лежит в файле и одна для всех веток. Подготовка переписывает
# файл разбиения внутри корня ветки, поэтому фолды идут строго по очереди:
# сначала подготовка ветки под фолд, потом все её сиды, и только затем следующий
# фолд. Копию разбиения кладём рядом с журналами, иначе разбор задним числом
# будет нечем воспроизвести.
SC="$HOME/tgseqloc_data/scripts"
OUT="$HOME/tgseqloc_data/cv_runs"; mkdir -p "$OUT"
cd "$HOME/TGSeqLoc" || exit 1

wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 1500 ] && return 0; sleep 60; done; }

root_of() {
    case "$1" in
        notext) echo "$HOME/tgseqloc_data/rcg_notext/robotcar" ;;
        real_qwen) echo "$HOME/tgseqloc_data/rcgf_real_qwen/robotcar" ;;
        shuffled_qwen) echo "$HOME/tgseqloc_data/rcgf_shuffled_qwen/robotcar" ;;
    esac
}

for fold in 0 1 2 3 4; do
    for arm in notext real_qwen shuffled_qwen; do
        echo "=== подготовка ${arm} фолд ${fold} $(date +%H:%M) ==="
        ~/.local/bin/uv run tgseqloc prepare -c "configs/cv_${arm}_f${fold}_s42.yaml" \
            > "$SC/prep_cv_${arm}_f${fold}.log" 2>&1 || { echo "  ОШИБКА подготовки"; tail -3 "$SC/prep_cv_${arm}_f${fold}.log"; continue; }
        cp "$(root_of "$arm")/mappings/temporal_split.json" "$OUT/split_${arm}_f${fold}.json"
        for seed in 42 43 44 45 46; do
            out="$OUT/${arm}_f${fold}_${seed}.json"
            grep -q final_metrics "$out" 2>/dev/null && continue
            wait_gpu
            echo "=== ${arm} фолд ${fold} сид ${seed} $(date +%H:%M) ==="
            ~/.local/bin/uv run tgseqloc train -c "configs/cv_${arm}_f${fold}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
            rm -f "outputs/cv-${arm}-f${fold}-s${seed}/last_model.pth"
        done
    done
done
echo CV_DONE
