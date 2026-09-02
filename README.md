# TGSeqLoc

TGSeqLoc — экспериментальный конвейер распознавания мест по графам сцены, дополненным текстовыми узлами. Для каждого кадра он объединяет внешний scene graph и заранее рассчитанный PaddleOCR, кодирует строки замороженным multilingual E5, обучает GAT-дескриптор триплетной функцией потерь и оценивает поиск места через FAISS и Recall@K.

Поддержанный сквозной сценарий сейчас один: датасет `v4rl`, OCR source `precomputed_paddleocr`, scene graph source `external_json`, фильтр `confidence`, text encoder backend `multilingual_e5`, fusion `text_nodes`, graph encoder `gat`, miner `hard_negative`, retriever `faiss_cosine` и metric `recall_at_k`. `PipelineRunner` разрешает эти имена через реестр и внедряет выбранные реализации в preparation и trainer. Полные контракты форматов и стадий находятся в [docs/pipeline_contracts.md](docs/pipeline_contracts.md).

## Установка через uv

Требуется Python 3.13 и установленный [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run tgseqloc components
uv run tgseqloc validate --config configs/v4rl_gat_text_nodes.yaml --no-check-inputs
```

Первый запуск подготовки скачивает модель Hugging Face, если её ещё нет в локальном кэше. Для CUDA используется `runtime.device: auto` или `cuda`; доступность CUDA всё равно проверяется во время запуска.

## Команды

```bash
# Проверить конфигурацию, реестр и наличие всех исходных файлов
uv run tgseqloc validate -c configs/v4rl_gat_text_nodes.yaml

# Проверить только конфигурацию и выбранные реализации
uv run tgseqloc validate -c configs/v4rl_gat_text_nodes.yaml --no-check-inputs

# Построить/переиспользовать подготовленные графы
uv run tgseqloc prepare -c configs/v4rl_gat_text_nodes.yaml

# Обучить модель; по завершении автоматически оценить лучший checkpoint
uv run tgseqloc train -c configs/v4rl_gat_text_nodes.yaml

# Оценить best, last или явный путь
uv run tgseqloc evaluate -c configs/v4rl_gat_text_nodes.yaml --checkpoint best
uv run tgseqloc evaluate -c configs/v4rl_gat_text_nodes.yaml --checkpoint outputs/v4rl-gat-text-nodes/last_model.pth

# Выполнить prepare → train → финальную оценку
uv run tgseqloc run -c configs/v4rl_gat_text_nodes.yaml
```

## Размещение V4RL и внешних признаков

Имена кадров обязаны иметь вид `{index:06d}_{timestamp:019d}.png|jpg|jpeg`. Индексы внутри каждой последовательности должны начинаться с нуля, идти без пропусков, а timestamps — не убывать. При `chunk_size: 200` индекс `237` относится к chunk `1`, локальный индекс scene graph равен `37`.

```text
data/
├── V4RL/
│   ├── seq1_frames/
│   │   └── 000000_0000000000000000000.png
│   ├── seq2_frames/
│   │   └── 000000_0000000000000000000.png
│   └── gt/
│       └── gt_shop_street_1.txt
├── ocr/
│   ├── seq1/chunk_0/000000_0000000000000000000/paddleocr_v5.json
│   └── seq2/chunk_0/000000_0000000000000000000/paddleocr_v5.json
└── scene_graphs/
    ├── seq1/seq1_c000/000000.json
    └── seq2/seq2_c000/000000.json
```

Входной GT — построчный текстовый файл `.txt`: комментарии начинаются с `#`; строка `query_timestamp,reference_start_timestamp,reference_end_timestamp` задаёт положительный интервал; одиночный timestamp служит ключевым кадром для режима интерполяции. После подготовки нормализованное отображение сохраняется уже как настоящий JSON в `data/prepared/v4rl/mappings/gt_shop_street_1.json`.

Пути OCR и scene graph задаются шаблонами с `{sequence}` и `{chunk}`. Scene graph использует node-link JSON (`nodes`, `links` или `edges`), нормализованные `bbox_2d.xyxy`/`bbox` и ссылки `source`/`target`. PaddleOCR JSON должен содержать положительные размеры изображения и список `predictions`, `results` или `ocr`.

## Политики кэша

Подготовленные графы лежат в `dataset.prepared_root/v4rl`. Кэш использует два уровня идентичности:

- `global_preprocess_fingerprint` описывает общие для всех графов schema/config, словари классов и рёбер, выбранные registry backends и фактическую identity E5. Identity реализации включает module/qualname, SHA-256 доступного Python source и явный детерминированный `cache_identity`, а encoder — также model/revision/resolved commit и размерность;
- `preprocess_fingerprint` каждого кадра объединяет global fingerprint только с identity его image, OCR и scene graph: абсолютным путём, size, `mtime_ns` и SHA-256.

Поэтому изменение одного OCR пересобирает только соответствующий `.pt`. Изменение общей настройки/backend/vocabulary пересобирает все затронутые глобальной identity графы. GT и temporal split не входят в per-frame fingerprint и не вызывают ненужную пересборку графов.

Stateful OCR/scene-graph/filter/fusion/text-encoder factory callables и создаваемые text encoders обязаны объявлять стабильный `cache_identity`. Это не позволяет скрытому состоянию переиспользовать неверный кэш.

- `reuse_if_compatible` — переиспользовать каждый совместимый `.pt`, несовместимый, повреждённый или отсутствующий построить заново.
- `require_existing` — ничего не строить; завершиться ошибкой при первом отсутствующем, повреждённом или несовместимом `.pt`.
- `rebuild` — пересобрать все графы независимо от текущего кэша.

После подготовки `dataset_fingerprint`/`checkpoint_identity` связывает global fingerprint, все ожидаемые per-frame fingerprints, упорядоченные SHA-256 подготовленных графов, identity GT, нормализованный mapping и полный temporal split. Поэтому изменение graph bytes, GT, положительных интервалов, ролей sequence или split инвалидирует checkpoint, но изменение одной разметки не требует заново вычислять неизменные frame graphs.

`manifest.json` содержит точный `graph_records`: для каждого ожидаемого графа — `path`, `sequence`, `frame_index`, `frame_stem`, SHA-256 файла, `expected_fingerprint` и `compatible`. При следующем `prepare` SHA из предыдущего manifest сверяется с файлом: tampering делает только этот кадр несовместимым и вызывает его пересборку. Если manifest отсутствует или повреждён, существующие `.pt` не считаются доверенными и перестраиваются. Лишние `.pt`, не входящие в текущий набор кадров, перечисляются в `stale_graph_paths` и не считаются ожидаемыми outputs. Перед созданием модели `Trainer` повторно сверяет все пути database/query из split с `graph_records`, наличие файлов, их SHA-256 и встроенный per-frame fingerprint.

Temporal split явно сохраняет role-aware `database_paths` для `reference_sequence` и `query_paths` для `query_sequence`, а также оба имени ролей. Checkpoint связан с dataset fingerprint и точными аргументами модели. Запись `.pt`, manifest, mapping, split и checkpoints выполняется через временный файл с атомарной заменой.

## Обучение и retrieval

- `training.recall_values` управляет validation и выбором `training.early_stopping_metric`; `retrieval.recall_values` — финальным test Recall@K.
- `retrieval.top_k` активен для финального поиска. Фактическая глубина равна `max(retrieval.top_k, max(retrieval.recall_values))`, поэтому все запрошенные Recall@K вычислимы, а `test_retrievals.json` сохраняет кандидатов этой глубины.
- `training.patience: 0` не останавливает обучение после первого улучшения. Остановка происходит на первом epoch без строгого улучшения; каждый улучшившийся epoch сбрасывает счётчик.

## Как добавить реализацию

1. Реализуйте подходящий контракт из `tgseqloc.registry`: `source`, `filter`, `encoder`, `fusion`, `graph_encoder`, `dataset`, `miner`, `retriever`, `reranker` или `metric`.
2. Зарегистрируйте фабрику через `registry.add(...)`/декоратор `register(...)`; имена нормализуются в lower-case с заменой `-` на `_`, дубликаты запрещены без `replace=True`.
3. Подключите встроенную регистрацию в `register_builtin_components()` и добавьте имя в допустимые значения строгой конфигурации.
4. Выберите имя в соответствующем config-поле. `PipelineRunner` разрешит dataset adapter, оба source, filter, text encoder factory, fusion builder, graph encoder, miner, retriever и metric через registry и внедрит фабрику в нужную стадию. Сохраните ожидаемую сигнатуру и контракт артефактов либо увеличьте `SCHEMA_VERSION` и включите новые влияющие параметры в fingerprint.
5. Проверьте имя через `uv run tgseqloc components`, затем `validate`, `prepare`, однопроходное обучение и `evaluate`.

Неизвестные YAML-ключи и неподдержанные имена отклоняются сразу. Подробная памятка для расширения каждого компонента приведена в [docs/pipeline_contracts.md](docs/pipeline_contracts.md).
