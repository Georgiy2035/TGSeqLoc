#!/bin/sh
# Замер инференса по этапам на 500 кадрах (125 на камеру), пакет 1. GPU проверяется на чужие процессы до и после.
B="$HOME/tgseqloc_data/bench"; cd "$B" || exit 1
rm -f stages.jsonl monitor.csv
{ echo "=== старт $(date "+%F %T")"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; uptime; } > env.txt
"$HOME/TGSeqLoc/.venv/bin/python" frames.py || exit 1
touch monitor.flag; "$HOME/TGSeqLoc/.venv/bin/python" monitor.py monitor.flag monitor.csv & MON=$!
sleep 5
"$HOME/ocr_benchmark/.venv-paddle/bin/python" stage_ocr.py paddleocr_v5 500 stages.jsonl > ocr_paddle.log 2>&1 || echo "ОШИБКА paddle"
sleep 5
"$HOME/dynseg_benchmark/.venv/bin/python" stage_yolo.py stages.jsonl > yolo.log 2>&1 || echo "ОШИБКА yolo"
sleep 5
cd "$HOME/TGSeqLoc" && "$HOME/.local/bin/uv" run python "$B/stage_tg.py" "$B/stages.jsonl" > "$B/tg.log" 2>&1 || echo "ОШИБКА tgseqloc"; cd "$B"
sleep 5
"$HOME/ocr_benchmark/.venv-torch/bin/python" stage_ocr.py qwen3_vl_4b 200 stages.jsonl > ocr_qwen.log 2>&1 || echo "ОШИБКА qwen"
rm -f monitor.flag; wait $MON
{ echo "=== конец $(date "+%F %T")"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; uptime; } >> env.txt
"$HOME/TGSeqLoc/.venv/bin/python" report.py "$B" > report.md
echo BENCH_DONE
