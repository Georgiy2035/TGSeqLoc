#!/bin/sh
# Разметка V4RL через EasyOCR, третья точка на оси качества распознавания.
#
# Запускается интерпретатором .venv-torch, а не `uv run`: uv пересобирает
# опциональную зависимость vllm из исходников и падает на отсутствующем
# CUDA_HOME, хотя самому EasyOCR vllm не нужен.
set -e
cd "$HOME/ocr_benchmark"
FRAMES=/mnt/external_usb_hdd/6YL/Datasets/v4rl
PY="$HOME/ocr_benchmark/.venv-torch/bin/python"

echo "старт $(date '+%F %H:%M')"
echo "=== проверка на 8 кадрах ==="
"$PY" run.py --model easyocr --images "$FRAMES/seq1_frames" \
    --experiment v4rl_smoke_easyocr --results-root results --limit 8
probe=$(ls results/v4rl_smoke_easyocr/*/easyocr.json 2>/dev/null | head -1)
if [ -z "$probe" ]; then
    echo "ПРОВЕРКА ПРОВАЛЕНА: json не создан"
    exit 1
fi
python3 -c "
import json
data = json.load(open('$probe'))
print('  предсказаний:', len(data.get('predictions', [])))
print('  пример:', [p.get('text') for p in data.get('predictions', [])][:5])
"

for seq in seq1 seq2; do
    echo "=== easyocr ${seq} старт $(date '+%H:%M') ==="
    "$PY" run.py --model easyocr --images "$FRAMES/${seq}_frames" \
        --experiment "v4rl_${seq}" --results-root results
    echo "=== easyocr ${seq} готово $(date '+%H:%M') ==="
done
echo EASYOCR_DONE
