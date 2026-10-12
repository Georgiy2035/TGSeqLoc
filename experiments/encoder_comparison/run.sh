#!/bin/sh
# Сравнение семейств текстовых энкодеров, разведочный заход: 3 сида на строку.
#
# Всё, кроме энкодера, зафиксировано: те же кадры, те же сцен-графы, тот же OCR,
# тот же split, та же схема обучения. Отличается только представление строк.
#
# char_ngram взят на 384, а не на 64, чтобы размерность совпадала с E5-small и
# разница между строками была разницей представлений, а не ёмкости.
set -e
cd "$HOME/TGSeqLoc"
mkdir -p /tmp/encexp

SEEDS="42 43 44"

# имя | prepared_root | backend | params-блок | text_emb_dim
run_row() {
    name=$1; root=$2; backend=$3; params=$4; dim=$5
    for seed in $SEEDS; do
        cat > "configs/enc_${name}_s${seed}.yaml" <<YAML
dataset:
  adapter: v4rl
  root: /mnt/external_usb_hdd/6YL/Datasets/v4rl
  sequences: [seq1, seq2]
  ocr_root_template: /home/sitnikov_sv/ocr_benchmark/results/v4rl_{sequence}
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
    backend: ${backend}
    params:
${params}
model:
  use_text_nodes: true
  text_emb_dim: ${dim}
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
  experiment_name: enc-${name}-s${seed}
YAML
        echo "=== ${name} сид ${seed} ==="
        # Первый сид строит графы (prepare), остальные переиспользуют.
        if [ "$seed" = "42" ]; then
            ~/.local/bin/uv run tgseqloc run -c "configs/enc_${name}_s${seed}.yaml" \
                > "/tmp/encexp/${name}_${seed}.json" 2>&1 || echo "  ОШИБКА"
        else
            ~/.local/bin/uv run tgseqloc train -c "configs/enc_${name}_s${seed}.yaml" \
                > "/tmp/encexp/${name}_${seed}.json" 2>&1 || echo "  ОШИБКА"
        fi
    done
}

run_row charngram384 "$HOME/tgseqloc_data/prep_cn384" char_ngram \
"      embedding_dim: 384" 384

run_row e5small "$HOME/tgseqloc_data/prep_e5s" multilingual_e5 \
"      model_name: intfloat/multilingual-e5-small
      revision: null" 384

run_row bgem3 "$HOME/tgseqloc_data/prep_bgem3" multilingual_e5 \
"      model_name: BAAI/bge-m3
      revision: null
      prefix: ''" 1024

echo ENC_DONE
