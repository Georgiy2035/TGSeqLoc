#!/bin/sh
# Разметка V4RL двумя дополнительными распознавателями для OCR-ablation.
#
# Обе модели пишут в тот же каталог кадра, что и PaddleOCR, под своим именем:
# qwen3_vl_4b.json и easyocr.json рядом с paddleocr_v5.json. Ветка эксперимента
# выбирается именем файла, а не отдельным деревом результатов, поэтому графы,
# разбиение и ground truth остаются общими на все три ветки.
#
# Qwen идёт через vLLM: 1.18 кадр/с против 0.48 у Transformers. --enforce-eager
# обязателен, torch.compile и CUDA-графы ломают грундинг у Qwen3-VL.
# Докатка встроена в оба раннера, так что обрыв не стоит всего прогона.
set -e
cd "$HOME/ocr_benchmark"

VLLM_PY=/mnt/external_usb_hdd/6YL/projects/sitnikov_sv_ocr_benchmark/envs/vllm/bin/python
STORAGE="$HOME/ocr_benchmark/.model_storage"
FRAMES=/mnt/external_usb_hdd/6YL/Datasets/v4rl
export HF_HUB_OFFLINE=1
export OCR_BENCHMARK_TMPDIR=/mnt/external_usb_hdd/6YL/otmp

echo "старт $(date '+%F %H:%M')"

# Проверка на восьми кадрах: полный прогон стоит четыре часа, и убедиться,
# что движок вообще отдаёт предсказания, дешевле заранее.
echo "=== проверка на 8 кадрах ==="
"$VLLM_PY" run_vllm.py --model qwen3_vl_4b \
    --images "$FRAMES/seq1_frames" --experiment v4rl_smoke_qwen \
    --results-root results --model-dir "$STORAGE" \
    --concurrency 4 --enforce-eager --limit 8
probe=$(ls results/v4rl_smoke_qwen/*/qwen3_vl_4b.json 2>/dev/null | head -1)
if [ -z "$probe" ]; then
    echo "ПРОВЕРКА ПРОВАЛЕНА: json не создан"
    exit 1
fi
echo "проверка пройдена: $probe"
python3 -c "
import json, sys
data = json.load(open('$probe'))
print('  предсказаний:', len(data.get('predictions', [])))
print('  пример:', [p.get('text') for p in data.get('predictions', [])][:5])
"

for seq in seq1 seq2; do
    echo "=== qwen ${seq} старт $(date '+%H:%M') ==="
    "$VLLM_PY" run_vllm.py --model qwen3_vl_4b \
        --images "$FRAMES/${seq}_frames" --experiment "v4rl_${seq}" \
        --results-root results --model-dir "$STORAGE" \
        --concurrency 4 --enforce-eager
    echo "=== qwen ${seq} готово $(date '+%H:%M') ==="
done

for seq in seq1 seq2; do
    echo "=== easyocr ${seq} старт $(date '+%H:%M') ==="
    ~/.local/bin/uv run python run.py --model easyocr \
        --images "$FRAMES/${seq}_frames" --experiment "v4rl_${seq}" \
        --results-root results
    echo "=== easyocr ${seq} готово $(date '+%H:%M') ==="
done
echo OCR_CHAIN_DONE
