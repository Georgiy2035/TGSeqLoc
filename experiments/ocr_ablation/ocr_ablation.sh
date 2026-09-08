#!/bin/sh
# OCR-ablation: три распознавателя на одном во всём остальном пайплайне.
#
# Сцен-графы, разбиение, ground truth, текстовый энкодер, схема обучения и
# сиды общие; отличается только то, чей текст попал в узлы. Ветка без текста
# не перезапускается -- она от распознавателя не зависит, числа взяты из
# основного эксперимента.
#
# Конфигурация повторяет основной прогон: char_ngram 64d, GATv2 2x4,
# lr 3e-5, batch 8, до 8 эпох, patience 3, сиды 42-51.
cd "$HOME/TGSeqLoc"
mkdir -p /tmp/ocrabl

while pgrep -f '[e]asyocr_chain.sh' > /dev/null; do sleep 120; done
echo "разметка готова $(date '+%F %H:%M')"

SEEDS="42 43 44 45 46 47 48 49 50 51"

# Разметка могла не доехать: без проверки обучение стартовало бы на пустоте
# и падало бы покадрово, что читается хуже одной внятной строки.
require_sidecars() {
    for seq in seq1 seq2; do
        expected=$(ls -d "$HOME/ocr_benchmark/results/v4rl_${seq}"/*/ 2>/dev/null | wc -l)
        found=$(ls "$HOME/ocr_benchmark/results/v4rl_${seq}"/*/"$1" 2>/dev/null | wc -l)
        echo "  ${seq}: ${found} из ${expected} кадров размечено ($1)"
        [ "$found" -lt "$expected" ] && { echo "ПРОПУСК ветки $1: разметка неполна"; return 1; }
    done
    return 0
}

run_arm() {
    name=$1; sidecar=$2; root=$3
    echo "--- проверка разметки ${sidecar} ---"
    require_sidecars "$sidecar" || return 0
    for seed in $SEEDS; do
        cat > "configs/abl_${name}_s${seed}.yaml" <<YAML
dataset:
  adapter: v4rl
  root: /mnt/external_usb_hdd/6YL/Datasets/v4rl
  sequences: [seq1, seq2]
  ocr_root_template: /home/sitnikov_sv/ocr_benchmark/results/v4rl_{sequence}
  ocr_file_name: ${sidecar}
  scene_graph_root_template: /mnt/external_usb_hdd/6YL/Datasets/v4rl_llm/{sequence}_scenegraphs
  gt_path: /mnt/external_usb_hdd/6YL/Datasets/v4rl/gt_shop_street_1/gt_shop_street_1.txt
  prepared_root: ${root}
  reference_sequence: seq1
  query_sequence: seq2
  test_ratio: 0.2
  validation_ratio: 0.1
  interpolate_gt: true
  chunk_size: 200
preprocess:
  use_text_nodes: true
  encoder_batch_size: 256
  text_encoder:
    backend: char_ngram
    params:
      embedding_dim: 64
model:
  use_text_nodes: true
  text_emb_dim: 64
  hidden_dim: 128
  proj_dim: 64
training:
  epochs: 8
  patience: 3
  batch_size: 8
  learning_rate: 3.0e-5
runtime:
  device: auto
  seed: ${seed}
  deterministic: true
output:
  experiment_name: abl-${name}-s${seed}
YAML
        echo "=== ${name} сид ${seed} $(date '+%H:%M') ==="
        if [ "$seed" = "42" ]; then
            ~/.local/bin/uv run tgseqloc run -c "configs/abl_${name}_s${seed}.yaml" \
                > "/tmp/ocrabl/${name}_${seed}.json" 2>&1 || echo '  ОШИБКА'
        else
            ~/.local/bin/uv run tgseqloc train -c "configs/abl_${name}_s${seed}.yaml" \
                > "/tmp/ocrabl/${name}_${seed}.json" 2>&1 || echo '  ОШИБКА'
        fi
    done
}

run_arm qwen qwen3_vl_4b.json "$HOME/tgseqloc_data/prep_qwen"
run_arm easyocr easyocr.json "$HOME/tgseqloc_data/prep_easyocr"

echo ABLATION_DONE
