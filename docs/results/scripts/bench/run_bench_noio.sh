#!/bin/sh
# Перемер OCR и масок без чтения кадров с диска (кадры заранее в памяти). Те же 500 кадров, пакет 1.
B="$HOME/tgseqloc_data/bench"; cd "$B" || exit 1
mkdir -p noio && rm -f noio/stages.jsonl noio/monitor.csv
{ echo "=== старт $(date "+%F %T")"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; uptime; } > noio/env.txt
touch noio/monitor.flag; "$HOME/TGSeqLoc/.venv/bin/python" monitor.py noio/monitor.flag noio/monitor.csv & MON=$!
sleep 5
"$HOME/ocr_benchmark/.venv-paddle/bin/python" stage_ocr_mem.py paddleocr_v5 500 noio/stages.jsonl > noio/ocr_paddle.log 2>&1 || echo "ОШИБКА paddle"
sleep 5
"$HOME/dynseg_benchmark/.venv/bin/python" stage_yolo_mem.py noio/stages.jsonl > noio/yolo.log 2>&1 || echo "ОШИБКА yolo"
sleep 5
"$HOME/ocr_benchmark/.venv-torch/bin/python" stage_ocr_mem.py qwen3_vl_4b 200 noio/stages.jsonl > noio/ocr_qwen.log 2>&1 || echo "ОШИБКА qwen"
rm -f noio/monitor.flag; wait $MON
{ echo "=== конец $(date "+%F %T")"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; uptime; } >> noio/env.txt
"$HOME/TGSeqLoc/.venv/bin/python" report.py "$B/noio" > noio/report.md 2>&1
echo BENCH_DONE
