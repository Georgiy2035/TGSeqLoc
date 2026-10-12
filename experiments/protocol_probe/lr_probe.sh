#!/bin/sh
# Свой learning rate для веток, на которых держится вывод о пороге.
#
# lr 3e-5 подобран сканом на ветке PaddleOCR с char_ngram-64 и достался всем
# остальным по наследству. У энкодеров оптимум оказался зависящим от
# представления (E5 выигрывает 3.7 пункта на 1e-5), поэтому утверждение
# «EasyOCR неотличим от отсутствия текста» нельзя делать, не проверив, что обе
# ветки обучались на разумном для них шаге. Ветка без текста не тюнилась вовсе.
#
# epochs 15 / patience 5 -- общие для скана, чтобы медленный шаг успел дойти до
# своего пика; отбор эпохи по-прежнему по валидации.
cd "$HOME/TGSeqLoc"
mkdir -p /tmp/lrprobe

SEEDS="42 43 44"

# имя | prepared_root | use_text_nodes | ocr_file_name
run_arm() {
    name=$1; root=$2; text=$3; sidecar=$4
    for lr in 1.0e-5 3.0e-5 1.0e-4; do
        tag=$(echo "$lr" | tr -d '.')
        for seed in $SEEDS; do
            cat > "configs/probe_${name}_${tag}_s${seed}.yaml" <<YAML
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
  use_text_nodes: ${text}
  text_encoder:
    backend: char_ngram
    params:
      embedding_dim: 64
model:
  use_text_nodes: ${text}
  text_emb_dim: 64
  hidden_dim: 128
  proj_dim: 64
training:
  epochs: 15
  patience: 5
  batch_size: 8
  learning_rate: ${lr}
runtime:
  device: auto
  seed: ${seed}
  deterministic: true
output:
  experiment_name: probe-${name}-${tag}-s${seed}
YAML
            echo "=== ${name} ${lr} сид ${seed} $(date +%H:%M) ==="
            ~/.local/bin/uv run tgseqloc train -c "configs/probe_${name}_${tag}_s${seed}.yaml" \
                > "/tmp/lrprobe/${name}_${tag}_${seed}.json" 2>&1 || echo "  ОШИБКА"
        done
    done
}

run_arm notext  "$HOME/tgseqloc_data/prepared_notext" false paddleocr_v5.json
run_arm easyocr "$HOME/tgseqloc_data/prep_easyocr"    true  easyocr.json

# Потолок в 8 эпох связывал ветку Qwen: три сида из десяти пикуют на 7-8.
# Направление безопасное -- бюджет мог только занизить победителя, -- но
# проверить дешевле, чем оговаривать.
for seed in 42 43 44 45 46 47 48 49 50 51; do
    sed -e 's/^  epochs: 8$/  epochs: 15/' -e 's/^  patience: 3$/  patience: 5/' \
        -e "s/experiment_name: abl-qwen-s${seed}/experiment_name: long-qwen-s${seed}/" \
        "configs/abl_qwen_s${seed}.yaml" > "configs/long_qwen_s${seed}.yaml"
    echo "=== qwen15 сид ${seed} $(date +%H:%M) ==="
    ~/.local/bin/uv run tgseqloc train -c "configs/long_qwen_s${seed}.yaml" \
        > "/tmp/lrprobe/qwen15_${seed}.json" 2>&1 || echo "  ОШИБКА"
done

echo PROBE_DONE
