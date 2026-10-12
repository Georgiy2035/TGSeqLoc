"""A4 — текстовый энкодер на V4RL, эталонная конфигурация, сид 42 (docs/ablations_v4rl.md).

Меняется только энкодер строк и согласованный с ним model.text_emb_dim; всё остальное — эталон
(v4abl_add_qwen_s42: Mask2Former, Qwen3-VL, фильтры, слагаемое, GAT 1×4 256/256, блочная валидация).
Пулинг и префикс — те, с которыми модель обучали: E5 — среднее и 'passage: ', BGE-M3 и LaBSE — CLS.
Эталон char n-grams 64d уже посчитан (v4abl-add_qwen-s42), здесь не повторяется.
"""
from pathlib import Path

H = Path.home()
C = H / "TGSeqLoc/configs"
DATA = H / "tgseqloc_data"
SOURCE = C / "v4abl_add_qwen_s42.yaml"
ENCODERS = {
    "cn384": (["    backend: char_ngram", "    params:", "      embedding_dim: 384"], 384),
    "e5small": (["    backend: multilingual_e5", "    params:", "      model_name: intfloat/multilingual-e5-small",
                 "      prefix: 'passage: '", "      pooling: mean"], 384),
    "e5base": (["    backend: multilingual_e5", "    params:", "      model_name: intfloat/multilingual-e5-base",
                "      prefix: 'passage: '", "      pooling: mean"], 768),
    "bgem3": (["    backend: multilingual_e5", "    params:", "      model_name: BAAI/bge-m3",
               "      prefix: ''", "      pooling: cls"], 1024),
    "labse": (["    backend: multilingual_e5", "    params:", "      model_name: sentence-transformers/LaBSE",
               "      prefix: ''", "      pooling: cls"], 768),
}

n = 0
for name, (block, dim) in ENCODERS.items():
    root = DATA / f"abl_v4_enc_{name}"
    out, skipping = [], False
    for line in SOURCE.read_text().splitlines():
        if line.startswith("  text_encoder:"):
            out += ["  text_encoder:"] + block
            skipping = True
            continue
        if skipping:
            if line.startswith("    ") or line.startswith("      "):
                continue
            skipping = False
        if line.startswith("  prepared_root:"):
            line = f"  prepared_root: {root}"
        elif line.startswith("  text_emb_dim:"):
            line = f"  text_emb_dim: {dim}"
        elif line.startswith("  experiment_name:"):
            line = f"  experiment_name: v4abl-enc_{name}-s42"
        out.append(line)
    body = "\n".join(out) + "\n"
    for need in (f"text_emb_dim: {dim}", str(root), "split_guard: ground_truth_blocks", "validation_centres: [0.3, 0.6]",
                 "ocr_file_name: qwen3_vl_4b.json", "text_fusion: additive", "miner_warmup_epochs: 10", "n_layers: 1"):
        assert need in body, (name, need)
    assert body.count("text_encoder:") == 1 and body.count("backend:") == 3  # ocr-источник, сегментация, динамика + энкодер
    (C / f"v4abl_enc_{name}_s42.yaml").write_text(body)
    n += 1
print("конфигов:", n)
