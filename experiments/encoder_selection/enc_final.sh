#!/bin/sh
# Выбор текстового энкодера: скан lr на каждую строку, затем финал на её
# собственном оптимуме.
#
# Шаг выбирается ПО ВАЛИДАЦИИ, тест в отборе не участвует. Общий lr был бы
# подгонкой под ту строку, на которой его когда-то подобрали, — на этом уже
# сгорело прошлое сравнение энкодеров.
#
# Пулинг у каждой модели свой, как её обучали: E5 и RuModernBERT — mean,
# BGE-M3 и LaBSE — CLS. Единый рецепт сравнивал бы пулинги, а не представления.
cd "$HOME/TGSeqLoc"
mkdir -p /tmp/encsel

emit() {  # имя, бэкенд, params-блок, dim, prepared_root, lr, сид
    cat > "configs/es_$1_$6_s$7.yaml" <<YAML
dataset:
  adapter: v4rl
  root: /mnt/external_usb_hdd/6YL/Datasets/v4rl
  sequences: [seq1, seq2]
  ocr_root_template: /home/sitnikov_sv/ocr_benchmark/results/v4rl_{sequence}
  scene_graph_root_template: /mnt/external_usb_hdd/6YL/Datasets/v4rl_llm/{sequence}_scenegraphs
  gt_path: /mnt/external_usb_hdd/6YL/Datasets/v4rl/gt_shop_street_1/gt_shop_street_1.txt
  prepared_root: $5
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
    backend: $2
    params:
$3
model:
  use_text_nodes: true
  text_emb_dim: $4
  hidden_dim: 128
  proj_dim: 64
training:
  epochs: 8
  patience: 3
  batch_size: 8
  learning_rate: $6
runtime:
  device: auto
  seed: $7
  deterministic: true
output:
  experiment_name: es-$1-$6-s$7
YAML
}

scan() {
    name=$1; backend=$2; params=$3; dim=$4; root=$5
    for lr in 1.0e-5 3.0e-5 1.0e-4; do
        for seed in 42 43 44; do
            emit "$name" "$backend" "$params" "$dim" "$root" "$lr" "$seed"
            echo "=== скан $name $lr сид $seed $(date +%H:%M) ==="
            ~/.local/bin/uv run tgseqloc run -c "configs/es_${name}_${lr}_s${seed}.yaml" \
                > "/tmp/encsel/scan_${name}_${lr}_${seed}.json" 2>&1 || echo "  ОШИБКА"
        done
    done
}

D=$HOME/tgseqloc_data
scan charngram char_ngram "      embedding_dim: 64" 64 "$D/es_charngram"
scan e5small multilingual_e5 "      model_name: intfloat/multilingual-e5-small
      prefix: 'passage: '
      pooling: mean" 384 "$D/es_e5small"
scan e5base multilingual_e5 "      model_name: intfloat/multilingual-e5-base
      prefix: 'passage: '
      pooling: mean" 768 "$D/es_e5base"
scan bgem3 multilingual_e5 "      model_name: BAAI/bge-m3
      prefix: ''
      pooling: cls" 1024 "$D/es_bgem3"
scan labse multilingual_e5 "      model_name: sentence-transformers/LaBSE
      prefix: ''
      pooling: cls" 768 "$D/es_labse"
scan rumodernbert multilingual_e5 "      model_name: deepvk/RuModernBERT-base
      prefix: ''
      pooling: mean" 768 "$D/es_rumodernbert"

echo ENC_SCAN_DONE
