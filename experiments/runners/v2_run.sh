#!/bin/sh
# Перепрогон всех веток RobotCar на новом разбиении (валидация — свой географический
# сегмент, отделённый теми же 25 м, что и тест).
#
# Ждём две вещи: окончания прогона размеров GAT (он считает на старом разбиении,
# и пересборка split посреди него смешала бы два протокола) и отметки о том, что
# правка разбиения прошла тесты.
SC="$HOME/tgseqloc_data/scripts"
cd "$HOME/TGSeqLoc" || exit 1

while [ -f "$SC/gsz.pid" ] && kill -0 "$(cat "$SC/gsz.pid")" 2>/dev/null; do sleep 60; done
echo "прогон размеров GAT завершён $(date +%H:%M)"
i=0
while [ ! -f "$SC/split_v2_ready" ]; do
    i=$((i + 1)); [ "$i" -gt 60 ] && { echo "НЕТ ОТМЕТКИ о правке разбиения, выхожу"; exit 1; }
    sleep 60
done

wait_gpu() { while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 1500 ] && return 0; sleep 60; done; }

echo "=== пересборка разбиения во всех корнях $(date +%H:%M) ==="
for spec in rcgf:real_paddle rcgf:real_qwen rcgf:real_easyocr rcgf:shuffled_paddle rcgf:shuffled_qwen \
            rcg:notext rcg:real_paddle rcg:real_qwen rcg:real_easyocr rcg:shuffled_paddle; do
    prefix=${spec%%:*}; arm=${spec#*:}
    ~/.local/bin/uv run tgseqloc prepare -c "configs/v2_${prefix}_${arm}_s42.yaml" > "$SC/prep_${prefix}_${arm}.log" 2>&1 \
        || { echo "  ОШИБКА подготовки ${prefix} ${arm}"; tail -3 "$SC/prep_${prefix}_${arm}.log"; }
done
~/.local/bin/uv run python "$SC/split_stats.py" 2>&1 | grep -v Warning

OUT="$HOME/tgseqloc_data/rcv2_runs"; mkdir -p "$OUT"
for spec in rcgf:real_paddle rcg:notext rcgf:real_qwen rcgf:shuffled_paddle rcgf:shuffled_qwen rcgf:real_easyocr \
            rcg:real_paddle rcg:real_qwen rcg:real_easyocr rcg:shuffled_paddle; do
    prefix=${spec%%:*}; arm=${spec#*:}
    for seed in 42 43 44 45 46 47 48 49 50 51; do
        out="$OUT/${prefix}_${arm}_${seed}.json"
        grep -q final_metrics "$out" 2>/dev/null && continue
        wait_gpu
        echo "=== ${prefix} ${arm} сид ${seed} $(date +%H:%M) ==="
        ~/.local/bin/uv run tgseqloc train -c "configs/v2_${prefix}_${arm}_s${seed}.yaml" > "$out" 2>&1 || echo "  ОШИБКА"
        rm -f "outputs/v2-${prefix}-${arm}-s${seed}/last_model.pth"
    done
done
echo V2_DONE
