# Контракты конвейера TGSeqLoc

Документ описывает фактически реализованный registry-driven путь данных. `PipelineRunner` разрешает через реестр dataset adapter, OCR source, scene graph source, text filter, text encoder factory, fusion builder, graph encoder, miner, retriever и metric, после чего внедряет их в preparation и `Trainer`.

Обозначения размеров:

- `N_o` — число валидных объектных узлов кадра;
- `N_t` — число OCR-строк после фильтрации;
- `N = N_o + N_t` — все узлы;
- `E_s` — число валидных направленных рёбер внешнего scene graph;
- `E_t` — число добавленных направленных text-object рёбер;
- `E = E_s + E_t`;
- `D_t` — размерность Frozen E5, для `multilingual-e5-small` обычно 384;
- `B` — число графов в PyG batch;
- `D_p = model.proj_dim` — размер place descriptor.

## 1. Строгая конфигурация и реестр

- **Имя.** Корневой тип `AppConfig`; секции `dataset`, `sources`, `preprocess`, `cache`, `model`, `training`, `retrieval`, `runtime`, `output`. Registry kinds: `source`, `filter`, `encoder`, `fusion`, `graph_encoder`, `dataset`, `miner`, `retriever`, `reranker`, `metric`. Встроенные имена: `precomputed_paddleocr`, `external_json`, `confidence`, `multilingual_e5`, `text_nodes`, `gat`, `v4rl`, `hard_negative`, `faiss_cosine`, `recall_at_k`.
- **Обязательный вход.** Для `load_config` — YAML mapping; формально все поля имеют defaults, но реальный запуск требует существующие пути V4RL/OCR/SG/GT. YAML из `configs/v4rl_gat_text_nodes.yaml` перечисляет каждое поле.
- **Опциональный вход.** `revision`, `resume_from`, `reranker` допускают `null`; остальные пропущенные поля получают dataclass-default. Поля выбора реализаций включают `preprocess.text_encoder_backend`, `training.miner` и `retrieval.metric`.
- **Выход.** Валидированный `AppConfig`; `component_map()` возвращает отсортированные доступные имена по kind.
- **Проверки и ошибки.** Неизвестный YAML-ключ, неверный тип, неподдержанное имя, пустые/повторяющиеся sequences, одинаковые роли, неверные ratio/range/размеры и некорректный early-stopping key дают `ConfigError`. Registry отклоняет неизвестный kind/name и повторную регистрацию (`KeyError`), пустое имя (`ValueError`).
- **Кэш.** Сам config не кэшируется; resolved config сохраняется в checkpoint, preprocessing-поля входят в fingerprint.
- **Следующий потребитель.** `PipelineRunner`, затем preparation или `Trainer`.

Имена реестра нормализуются: trim, lower-case, `-` → `_`. При создании runner все выбранные имена проверяются через `registry.get`. На `prepare` dataset adapter получает внедрённые `text_encoder_factory`, `ocr_parser`, `scene_graph_parser`, `text_filter` и `fusion_builder`; при создании trainer runner строит graph encoder через `registry.create` и передаёт miner, retriever factory и metric factory.

## 2. Обнаружение V4RL и precomputed OCR

- **Имя.** Config/registry: `dataset.adapter: v4rl`, `sources.ocr: precomputed_paddleocr`; parser `parse_paddleocr`.
- **Обязательный вход.** Кадры `{sequence}_frames/{index:06d}_{timestamp:019d}.png|jpg|jpeg`; OCR по `ocr_root_template`, конкретно `{frame_stem}/paddleocr_v5.json`. JSON требует `image_width`/`width > 0`, `image_height`/`height > 0` и строки в `predictions`, `results` или `ocr`.
- **Опциональный вход.** У строки: `confidence` или `score`, `text` или `rec_text`, `bbox` или `box`; bbox может быть `[x,y,w,h]` либо списком точек.
- **Выход.** `boxes: list[list[float]]` с нормализованными `[x1,y1,x2,y2]` в `[0,1]` и `texts: list[str]`, одинаковой длины.
- **Проверки и ошибки.** Нет кадров, отсутствующий sidecar, индексы не от нуля/с пропусками, немонотонные timestamps дают исключение. Неположительный размер изображения даёт `ValueError`; malformed/no-op строки и вырожденные bbox пропускаются.
- **Кэш.** Абсолютный путь, size, `mtime_ns` и SHA-256 OCR и изображения входят в identity только своего кадра. Она объединяется с global preprocess fingerprint в per-frame fingerprint; изменение одного OCR инвалидирует один graph, а не весь датасет. Отдельного OCR-кэша нет.
- **Следующий потребитель.** Внедрённый confidence filter применяется OCR parser через `prediction_filter`, затем выбранные E5 backend и text-node fusion.

Пути вычисляются как `chunk, local = divmod(index, dataset.chunk_size)`. `local` нужен scene graph; OCR использует полный `frame_stem`.

## 3. Внешний scene graph

- **Имя.** Config/registry: `sources.scene_graph: external_json`; parser `parse_scene_graph`.
- **Обязательный вход.** JSON по `scene_graph_root_template/{sequence}_c{chunk:03d}/{local:06d}.json`; `nodes` и `links` либо `edges`. Узлу нужен `id` и нормализованный bbox из `data.bbox_2d.xyxy`, `data.bbox` или полей верхнего уровня.
- **Опциональный вход.** `class_name`/`label` и edge `label`/`type`; пустое значение становится `unknown`.
- **Выход.** Объектные узлы с `class_idx`, `center`, `wh`, `xyxy`; directed edges `(source_index,target_index,label)`; число отброшенных dangling edges. Словари `node_class_to_idx` и `edge_label_to_idx` строятся по всем SG, `unknown` всегда имеет индекс 0.
- **Проверки и ошибки.** Невалидный/вырожденный bbox удаляет узел; ребро на удалённый или неизвестный `id` удаляется и учитывается в metadata. Ошибки JSON/IO распространяются.
- **Кэш.** Identity SG входит в fingerprint своего кадра. Общие словари классов/labels входят в global fingerprint: изменение SG без изменения vocabulary инвалидирует только этот кадр, но изменение состава vocabulary меняет global identity.
- **Следующий потребитель.** Text-node fusion.

## 4. Confidence filter

- **Имя.** Config/registry: `preprocess.text_filter: confidence`; порог `preprocess.ocr_confidence_threshold`.
- **Обязательный вход.** OCR sample с численно приводимым `confidence` или `score`; фактический parser считает отсутствующее значение равным `0.0`.
- **Опциональный вход.** Порог в `[0,1]`, default `0.0`; no-op тексты (`""`, `none`, `null`, `n/a`, `no text` и варианты) исключаются независимо от confidence.
- **Выход.** Registry-функция возвращает `bool`; `PipelineRunner` внедряет её в OCR parser как `prediction_filter`, результатом остаются согласованные boxes/texts.
- **Проверки и ошибки.** Конфигурация отклоняет порог вне `[0,1]`; неприводимый confidence вызывает `ValueError`.
- **Кэш.** Порог и отсортированный набор no-op строк входят в preprocessing fingerprint.
- **Следующий потребитель.** Frozen E5 получает только оставшиеся строки.

## 5. Замороженный multilingual E5

- **Имя.** Config/registry `preprocess.text_encoder_backend: multilingual_e5`; config хранит Hugging Face model id отдельно в `preprocess.text_encoder` (default `intfloat/multilingual-e5-small`) и revision в `preprocess.revision`. `PipelineRunner` разрешает backend factory через registry и внедряет её в dataset adapter.
- **Обязательный вход.** Последовательность строк и доступная HF tokenizer/model revision; строки получают prefix `passage: `.
- **Опциональный вход.** `revision`, `runtime.device`, `encoder_batch_size`; в тестах/интеграциях можно передать объект `TextEncoder` с `embedding_dim` и `encode`.
- **Выход.** CPU `torch.float32 [N_t,D_t]`, mean pooling по attention mask, затем L2-нормализация строк.
- **Проверки и ошибки.** Отсутствие `transformers` даёт `RuntimeError`; `validate_embeddings` требует точную форму `[len(texts), embedding_dim]` и конечные значения. Для пустого списка — `[0,D_t]`. Явно внедрённый encoder без детерминированного `cache_identity` отклоняется до кодирования.
- **Кэш.** Отдельный кэш embeddings не пишется. Implementation/source identity, явный `cache_identity`, model id, revision/resolved commit и `D_t` входят в global fingerprint; embeddings хранятся внутри frame `.pt`. Hugging Face может использовать собственный внешний кэш.
- **Следующий потребитель.** `build_fused_graph`.

## 6. Fusion текстовых узлов

- **Имя.** Config/registry: `preprocess.fusion: text_nodes`; strategy `overlap_nearest`, `overlap`, `nearest` или `fully_connected`.
- **Обязательный вход.** Object nodes, scene edges, одинаковое число boxes/texts, E5 matrix `[N_t,D_t]`, `edge_label_to_idx`.
- **Опциональный вход.** `connection_k` (фактически значим для `nearest`) и frame metadata.
- **Выход.** Один `torch_geometric.data.Data`: object nodes идут первыми, text nodes — после них. Для каждой выбранной связи создаются оба направления text→object и object→text; label у них `unknown`, `is_text_edge=True`.
- **Проверки и ошибки.** Несовпадение boxes/texts или embeddings даёт `ValueError`; неизвестная strategy — `ValueError`. При отсутствии object nodes text nodes не получают связей.
- **Кэш.** Strategy и `connection_k` входят в fingerprint; готовый fused graph атомарно сохраняется в `.pt`.
- **Следующий потребитель.** `sanitize_graph`, затем prepared V4RL dataset и GAT.

Стратегии: `overlap_nearest` выбирает один лучший объект по contain, IoU, дистанции и индексу; `overlap` — все содержащие/пересекающиеся; `nearest` — ближайшие `max(1,k)`; `fully_connected` — все объекты.

## 7. Каноническая схема графа

Schema version — `3`. После `sanitize_graph` все tensors находятся на CPU, числовые поля очищены от NaN/Inf, рёбра с индексом вне `[0,N)` удалены.

- `x`: `torch.float32 [N,4]`, строго `[cx, cy, width, height]`.
- `node_class`: `torch.int64 [N]`; text nodes имеют class index `0`.
- `is_text`: `torch.bool [N]`.
- `text_emb`: `torch.float32 [N,D_t]`; object rows нулевые, text rows содержат E5.
- `edge_index`: `torch.int64 [2,E]`, directed COO.
- `edge_attr`: `torch.float32 [E,10]`.
- `edge_label`: `torch.int64 [E]`.
- `edge_u_class`: `torch.int64 [E]`, класс source.
- `edge_v_class`: `torch.int64 [E]`, класс target.
- `is_text_edge`: `torch.bool [E]`.
- `text_strings`: Python `list[str]` длины `N_t`.

Десять edge-признаков в точном порядке: `distance`, `dx`, `dy`, `sin(angle)`, `cos(angle)`, `IoU`, `intersection/min_area`, `log1p(target_area/source_area)`, `log1p(target_width/source_width)`, `log1p(target_height/source_height)`.

Frame metadata:

- `sequence: str`;
- `frame_index: int`;
- `timestamp: int`;
- `frame_stem: str`;
- `image_path`, `ocr_path`, `scene_graph_path`: строки путей;
- `text_encoder_name: str`, `text_encoder_revision: str|null`;
- `preprocess_fingerprint: str` — per-frame fingerprint;
- `global_preprocess_fingerprint: str`;
- `source_identity: dict` — identity image/OCR/SG данного кадра;
- `backend_identities: dict` — настроенные registry names и фактические callable/encoder identities;
- `dropped_scene_edge_count: int`;
- `graph_rotated: bool` (сейчас всегда `false`);
- `schema_version: int` (сейчас `3`).

Контракт стадии:

- **Имя.** `sanitize_graph`, constants `NODE_FEATURE_DIM=4`, `EDGE_FEATURE_DIM=10`, `SCHEMA_VERSION=3`.
- **Обязательный вход.** `x [N,4]`, согласованный `edge_index [2,E]`; preparation передаёт все поля выше.
- **Опциональный вход.** Отсутствующие edge/node auxiliary fields дополняются нулями; неизвестная metadata переносится.
- **Выход.** Shape-consistent PyG `Data`.
- **Проверки и ошибки.** Неверный rank/фиксированная ширина `x` или `edge_index` вызывает `ValueError`; auxiliary tensors pad/truncate; invalid edges и соответствующие edge rows удаляются.
- **Кэш.** `graph_is_compatible` проверяет fingerprint, формы, индексы и finite values перед reuse.
- **Следующий потребитель.** `load_graph`, `EdgeAttrNormalizer`, GAT.

## 8. Fingerprint и политика артефактов

- **Имя.** `build_preprocess_fingerprint`, `frame_fingerprint`, `fingerprint_payload`, `source_file_identity`, config `cache.policy`.
- **Обязательный вход.** Global fingerprint включает schema version, dataset/sequences/chunk size, E5 id/revision/resolved commit/dimension, registry backend/callable identities, fusion strategy/k, OCR threshold/no-op set и node/edge vocabularies. Для Python callable identity содержит module/qualname, SHA-256 исходного текста из `inspect.getsource`, если он доступен, и явно заданный детерминированный `cache_identity`. Per-frame fingerprint объединяет global fingerprint с identity image/OCR/SG только конкретного кадра.
- **Опциональный вход.** Политика default `reuse_if_compatible`.
- **Выход.** Каждый graph хранит `preprocess_fingerprint` своего кадра и `global_preprocess_fingerprint`. Dataset fingerprint вычисляется отдельно из global fingerprint, отображений `expected graph path → per-frame fingerprint` и `expected graph path → SHA-256`, identity GT, нормализованного mapping и полного temporal split.
- **Проверки и ошибки.** Stateful OCR source, scene-graph source, filter, fusion или text-encoder factory (включая closure) без `cache_identity`, а также любой text encoder без `cache_identity`, отклоняются. Совместимость frame cache требует читаемый `.pt`, точный per-frame fingerprint, tensor shapes, finite values и валидные indices. Corrupt output считается несовместимым. `require_existing` превращает несовместимость в `RuntimeError`.
- **Кэш.** `reuse_if_compatible` пропускает совместимые кадры; изменение одного OCR/image/SG перестраивает только его graph. Preparation сверяет `expected_fingerprint` и SHA-256 из предыдущего manifest с текущим `.pt`: tampering инвалидирует и перестраивает этот кадр. Если manifest отсутствует, повреждён или не содержит записи, существующий `.pt` не считается доверенным и перестраивается; при `require_existing` это ошибка. Изменение global config/backend/source code/`cache_identity`/vocabulary меняет global fingerprint и инвалидирует все соответствующие graphs. GT/split меняют dataset/checkpoint fingerprint, но не per-frame cache. `require_existing` запрещает build; `rebuild` всегда пишет заново.
- **Следующий потребитель.** `PipelineRunner._prepared_metadata` требует `manifest.output_complete == true`; `Trainer` использует dataset fingerprint как checkpoint identity и выполняет preflight graph validation.

Точные integrity-поля manifest:

- `global_preprocess_fingerprint` — общая preprocessing identity;
- `dataset_fingerprint`, `checkpoint_identity`, `fingerprint` и историческое поле `preprocess_fingerprint` — один dataset-level digest;
- `expected_graph_paths` — упорядоченные относительные paths ожидаемых кадров;
- `graph_records` (тот же список также доступен как `expected_graphs`) — по одной записи на ожидаемый graph;
- каждая запись `graph_records` содержит ровно фактические поля `path`, `sequence`, `frame_index`, `frame_stem`, `sha256`, `expected_fingerprint`, `compatible`; `sha256` равен `null`, если файл отсутствует;
- `stale_graph_paths` — отсортированные относительные paths всех `.pt` под prepared root, которых нет в ожидаемом наборе; `stale_graph_count` — их число;
- `expected_graph_count`, `output_file_count`, `validated_output_count`, `output_complete` считают только ожидаемые graphs. Stale files не делают manifest complete и не попадают в split.

Dataset/checkpoint fingerprint включает global fingerprint, path-keyed набор ожидаемых per-frame fingerprints, path-keyed SHA-256 всех подготовленных графов в детерминированном порядке, GT как `source_identities.ground_truth`, а также получившиеся mapping и temporal split. Это связывает checkpoint не только с разметкой, но и с точными bytes каждого graph и гарантирует, что изменение разметки или ролей invalidates старый checkpoint без повторного E5/fusion для неизменившихся кадров.

## 9. V4RL dataset, GT и temporal split

- **Имя.** Config/registry `dataset.adapter: v4rl`; функции `discover_v4rl_records`, `build_gt_mapping`, `build_positive_intervals`, `build_temporal_split`.
- **Обязательный вход.** Две разные роли из `dataset.sequences`: `reference_sequence` (database) и `query_sequence`; входной GT `data/V4RL/gt/gt_shop_street_1.txt`. Каждая значимая строка содержит либо один timestamp, либо `query_ts,start_reference_ts,end_reference_ts`.
- **Опциональный вход.** Комментарии `#`, `gt_tolerance_ns`, `interpolate_gt`, `test_ratio`, `validation_ratio`.
- **Выход.** Входной `.txt` преобразуется в настоящий JSON `mappings/gt_shop_street_1.json` с привязкой timestamps к frame indices. `mappings/temporal_split.json` содержит `positives`, `train_query_indices`, `validation_query_indices`, `test_query_indices`, `test_start`, явные `database_paths`, `query_paths`, `reference_sequence`, `query_sequence`. Positive interval включает обе границы.
- **Проверки и ошибки.** Строка иной длины, query timestamp дальше tolerance, некорректные ratios, пустой обязательный split дают ошибку. Reference boundary вне временного диапазона помечается unavailable и не создаёт positives. Preparation разрешает пустые split (`require_nonempty=False`), но Trainer требует database/query graphs и непустой train.
- **Кэш.** Mapping и split пересчитываются при каждом `prepare`, даже если все frame `.pt` переиспользованы. GT identity, mapping и полный split входят в dataset/checkpoint fingerprint, но не в global или per-frame graph fingerprint.
- **Следующий потребитель.** Trainer формирует lazy graph datasets, triplets и evaluation query subsets.

Temporal protocol: `test_start = int(query_count * (1-test_ratio))`; test — размеченные query с индексом не меньше boundary; validation берётся с конца размеченной pre-test части; остаток идёт в train.

Пути split role-aware и не зависят от имён `seq1`/`seq2`: `database_paths` перечисляет каждый graph `reference_sequence` как `{reference_sequence}/{frame_stem}.pt`, `query_paths` аналогично перечисляет `query_sequence`. Trainer разрешает относительные paths от prepared root.

## 10. Нормализатор рёбер

- **Имя.** `EdgeAttrNormalizer`; отдельного поля в строгом `AppConfig` нет. Trainer пытается читать legacy `normalization.log_indices`, но такой YAML-ключ строгий loader отклонит, поэтому штатный AppConfig использует пустой список.
- **Обязательный вход.** Все database graphs плюс train query graphs; `edge_attr [E,10]`.
- **Опциональный вход.** `log_indices`, `eps=1e-6`, известный `feature_dim=10`.
- **Выход.** Per-column sample mean/std и transformed `float` tensor той же формы `[E,10]`; state сохраняется в checkpoint.
- **Проверки и ошибки.** Не-2D, неверная ширина, out-of-range log index или значение `<=-1` перед `log1p` дают `ValueError`; transform до finalize — `RuntimeError`. Пустые edge tensors игнорируются; без наблюдений при известной dimension получаются mean 0/std 1.
- **Кэш.** Отдельного файла нет. Fit выполняется перед новым training run; при resume/evaluate state загружается из checkpoint.
- **Следующий потребитель.** `load_graph` перед PyG batching и GAT.

## 11. GAT: входы и выходы

- **Имя.** Config/registry `model.graph_encoder: gat`; класс `GATGraphEncoder`. `PipelineRunner` проверяет dimensions по manifest, добавляет размеры vocabulary, создаёт модель через `registry.create` и внедряет её в `Trainer`.
- **Обязательный вход.** PyG graph/batch: `x float [ΣN,4]`, `edge_index long [2,ΣE]`; при настроенных class embeddings нужен `node_class`; при text nodes — `is_text [ΣN]`, `text_emb [ΣN,D_t]`.
- **Опциональный вход.** `edge_attr [ΣE,10]`, `edge_label [ΣE]`, `is_text_edge [ΣE]`, batch assignment `[ΣN]`. В Trainer размеры vocab берутся из manifest.
- **Выход.** L2-normalized `torch.Tensor [B,D_p]`. При `return_attn=True` также возвращаются attention weights последнего слоя.
- **Проверки и ошибки.** Пустой граф, неверные shapes, class/label index вне embedding table, отсутствующие text fields дают `ValueError`.
- **Кэш.** Параметры модели живут в checkpoints; exact `model.init_args` проверяется при resume/evaluate.
- **Следующий потребитель.** Hard miner, triplet loss или FAISS retrieval.

Обработка: geometry и categorical edge label проецируются в hidden space; при `use_edge_geometry=true` и доступном label смешиваются обучаемым gate. Для text edges categorical component зануляется. Следуют `n_layers` GATv2Conv, mean/max global pooling, projection и L2 normalization.

## 12. Hard-negative mining

- **Имя.** Config/registry `training.miner: hard_negative`; функция `mine_hard_negatives`. `PipelineRunner` разрешает miner через registry и внедряет callable в `Trainer`, который вызывает его каждый epoch.
- **Обязательный вход.** Database descriptors `float32 [N_db,D_p]`, descriptors выбранных train queries `[N_q,D_p]`, соответствующие query indices, positive mapping, `negatives_per_query`.
- **Опциональный вход.** `hard_search_depth`, seed.
- **Выход.** `dict[query_index,list[database_index]]` точной запрошенной длины, если существует хотя бы один negative; иначе пустой список.
- **Проверки и ошибки.** Матрицы должны быть 2-D с одинаковой dimension; query indices должны совпадать по длине; отрицательное число negatives запрещено.
- **Кэш.** Не сохраняется; пересчитывается в начале каждого epoch по текущей модели. Недостающие кандидаты добираются deterministic random и при необходимости повторяются.
- **Следующий потребитель.** `TripletGraphDataset` выбирает один случайный positive и фиксированное число mined negatives на query.

FAISS здесь использует нормализованные копии и `IndexFlatIP`, то есть exact cosine similarity; positives исключаются.

## 13. Triplet trainer и checkpoint

- **Имя.** `Trainer`; training config section.
- **Обязательный вход.** Complete manifest, temporal split с непустым train, prepared graphs, хотя бы одна query с positive и доступным negative.
- **Опциональный вход.** `training.resume_from`, validation subset, injected model/device.
- **Выход.** На epoch: mean triplet loss и validation Recall@K (либо `-loss`, если validation пуст); `best_model.pth`, `last_model.pth`; после training — final test metrics/retrievals и summary dict.
- **Проверки и ошибки.** Сразу после чтения split, до выбора device и создания модели, Trainer выполняет preflight по `manifest.graph_records`: каждый database/query path должен иметь уникальную запись, файл должен существовать, SHA-256 обязан присутствовать и совпадать, а встроенный graph `preprocess_fingerprint` — равняться `expected_fingerprint`. Таким образом, tampering после preparation блокирует train/evaluate даже до model work. Для legacy manifest без ключа `graph_records` проверка пропускается; текущий preparation всегда записывает ключ. Затем проверяются CUDA, graph shapes, model inputs, early-stopping metric, checkpoint dataset fingerprint/model args и обязательные resume fields. Gradient norm ограничивается `max_grad_norm`.
- **Кэш.** Hard negatives не кэшируются. Last checkpoint перезаписывается атомарно каждый epoch, best — только при строгом улучшении. Resume продолжает с `epoch+1` и восстанавливает optimizer/normalizer.
- **Следующий потребитель.** Final evaluation либо последующий `train --resume-from` через config.

Одна training group имеет порядок `[query, positive, negative_1, …]`. После GAT output reshaped в `[-1,2+M,D_p]`; `TripletMarginLoss(margin, p=2)` усредняется по всем `M` negatives.

Early stopping учитывает только epochs без строгого улучшения target metric (или `-loss`, если validation отсутствует). При `patience=0` первый улучшившийся epoch сохраняется нормально; остановка происходит на первом последующем неулучшившемся epoch. Любое улучшение сбрасывает `no_improvement` в ноль.

### Контракт checkpoint

Training checkpoint format version `1` содержит:

- `format_version: 1`;
- `model_init_args: dict` — точные constructor args;
- `model_state_dict`;
- `optimizer_state_dict`;
- `epoch: int`;
- `best_metric`, `metric: float`;
- `metrics: dict`;
- `resolved_config: dict`;
- `manifest: dict`;
- `manifest_fingerprint: str`;
- `normalizer_state` с count/mean/M2/std/log_indices/eps/feature_dim.

Для resume обязательны model state, normalizer state, optimizer state и epoch. Для evaluation достаточно model state и normalizer state, но в обоих случаях `manifest_fingerprint` checkpoint должен совпасть с текущим dataset fingerprint, включающим GT/split, а `model_init_args` — точно равняться аргументам текущей модели. `best` и `last` разрешаются в `<output.root>/<experiment_name>/best_model.pth|last_model.pth`; явный путь используется как передан.

## 14. FAISS retrieval и Recall@K

- **Имя.** Config/registry `retrieval.retriever: faiss_cosine` и `retrieval.metric: recall_at_k`; классы `Retriever`, `RecallAtK`. Обе фабрики разрешаются runner и внедряются в `Trainer`.
- **Обязательный вход.** Database descriptors `[N_db,D_p]`, query descriptors `[N_q,D_p]`, positives и query indices.
- **Опциональный вход.** IDs в `Retriever.fit`, набор K и `retrieval.top_k`. `retrieval.reranker` сейчас обязан быть `null`, потому что реализаций нет.
- **Выход.** Scores и neighbor indices имеют форму `[N_q,search_k]`; метрики — `{"R@K": percentage}`. Для final test `search_k=max(retrieval.top_k,max(retrieval.recall_values))`, и именно столько indices на query пишется в `test_retrievals.json`. Пустой query set даёт нули.
- **Проверки и ошибки.** Fit до search обязателен, matrices 2-D и dimensions должны совпадать, `K>0`, IDs должны соответствовать database.
- **Кэш.** FAISS `IndexFlatIP` строится в памяти для каждого вызова evaluation и не сохраняется. `final_metrics.json` и `test_retrievals.json` пишутся атомарно.
- **Следующий потребитель.** Early stopping использует validation metrics с `training.recall_values`; итоговый test-отчёт использует `retrieval.recall_values`.

Descriptors L2-normalized моделью и повторно нормализуются FAISS на копиях. Recall@K считается hit-rate по queries: hit есть, если хотя бы один из первых K database indices входит в positive interval.

`training.recall_values` задаёт набор K для validation и выбора `training.early_stopping_metric`; validation ищет до максимального из этих K. `retrieval.recall_values` независимо задаёт K для финальной test evaluation. Активный `retrieval.top_k` задаёт желаемую глубину сохраняемой test-выдачи, но поиск автоматически расширяется до максимального Recall@K, если `top_k` меньше.

## 15. Stage runner и CLI

- **Имя.** `PipelineRunner`; CLI `tgseqloc`.
- **Обязательный вход.** `--config/-c` для `validate`, `prepare`, `train`, `evaluate`, `run`.
- **Опциональный вход.** `validate --no-check-inputs`; `evaluate --checkpoint best|last|PATH`. `reranker` должен оставаться `null`: registry не содержит ни одной реализации этого kind.
- **Выход.** JSON в stdout. `components` печатает component map; `validate` — выбранный dataset, registry и prepared root; `prepare` — manifest; `train`/`run` — training summary; `evaluate` — final metrics.
- **Проверки и ошибки.** CLI преобразует `KeyError`, `OSError`, `RuntimeError`, `TypeError`, `ValueError` в Click error. `train/evaluate` требуют complete manifest и split; `prepare` сначала валидирует все source inputs.
- **Кэш.** `prepare` следует `cache.policy`; `train` пишет checkpoints; `evaluate` пишет final JSON. `run` выполняет `prepare → train`; final test evaluation уже вызывается внутри `Trainer.train`, отдельного третьего вызова runner нет.
- **Следующий потребитель.** Пользователь, CI или orchestration wrapper.

## Проход одного кадра

Пусть `seq2` содержит `000237_1234567890123456789.png`, `chunk_size=200`.

1. Discovery вычисляет `chunk=1`, `local=37` и проверяет:
   - `data/ocr/seq2/chunk_1/000237_1234567890123456789/paddleocr_v5.json`;
   - `data/scene_graphs/seq2/seq2_c001/000037.json`.
2. Scene graph parser оставляет `N_o` валидных bbox, переводит class/edge labels в глобальные vocabulary indices и удаляет dangling edges.
3. OCR parser нормализует bbox, применяет confidence threshold и no-op filter. Допустим, остаются `N_t=3` строки.
4. Frozen E5 создаёт `float32 [3,384]`.
5. Fusion добавляет три text nodes после object nodes. При `overlap_nearest` каждый text node выбирает один object и создаёт два directed edges, поэтому `E_t=6`.
6. Graph получает tensors `[N_o+3,4]`, `[N_o+3,384]`, `[2,E_s+6]`, `[E_s+6,10]` и metadata кадра.
7. Graph получает общий global fingerprint и per-frame fingerprint от global identity плюс identity только этого image/OCR/SG, затем атомарно сохраняется как `data/prepared/v4rl/seq2/000237_1234567890123456789.pt`. Изменение OCR этого кадра перестроит только данный `.pt`.
8. GT связывает query timestamp с query index 237 и reference interval. Temporal split назначает размеченный query в train/validation/test, сохраняет явные role-aware database/query paths; GT и split входят в dataset/checkpoint fingerprint.
9. Manifest фиксирует SHA-256 файла и ожидаемый per-frame fingerprint в `graph_records`. Перед моделью Trainer сверяет запись, файл, SHA и embedded fingerprint.
10. Trainer нормализует десять edge columns, PyG добавляет batch vector, GAT выдаёт `[1,D_p]`; descriptor участвует в mining/triplet loss либо ищет reference descriptor через exact cosine FAISS.

## Дерево артефактов на диске

```text
data/prepared/
└── v4rl/
    ├── manifest.json
    ├── seq1/
    │   └── {frame_stem}.pt
    ├── seq2/
    │   └── {frame_stem}.pt
    └── mappings/
        ├── gt_shop_street_1.json
        └── temporal_split.json

outputs/
└── v4rl-gat-text-nodes/
    ├── best_model.pth
    ├── last_model.pth
    ├── final_metrics.json
    └── test_retrievals.json
```

Manifest хранит global и dataset/checkpoint fingerprints, `graph_records` с SHA-256 и ожидаемым per-frame fingerprint каждого graph, `stale_graph_paths`, expected/write/skip/validated counts, source/backend identities, E5 identity/dimension, vocabularies и их размеры, graph dimensions, fusion settings, dropped-edge counts и относительные paths mapping/split. Split содержит явные role-aware database/query paths и sequence roles.

## Как коллегам добавлять реализации

1. Выберите registry kind и реализуйте соответствующий фактический интерфейс. Protocols дают общий ориентир, но сверяйтесь с инъекционной сигнатурой стадии: built-in source factories являются file parsers, dataset — preparation adapter, encoder — factory, fusion/miner — callables, retriever/metric/graph encoder — factories.
2. Зарегистрируйте стабильное уникальное имя в `register_builtin_components`; проверьте `tgseqloc components`.
3. Добавьте имя в допустимый tuple строгого config и, если нужны параметры, в dataclass секцию. Иначе loader отклонит YAML.
4. Выберите реализацию в соответствующем config-поле. Существующие точки подключения runner автоматически разрешают и внедряют dataset adapter, оба source parser, filter, text encoder factory, fusion builder, graph encoder, miner, retriever и metric.
5. Определите required/optional inputs, точные output shapes/dtypes, validation и следующего потребителя. Не меняйте существующую graph schema молча.
6. Если меняется подготовленный graph, увеличьте `SCHEMA_VERSION` при несовместимости и включите все влияющие параметры, model revisions и source identities в fingerprint/manifest.
7. Если меняется модель, сохраните полные constructor args в `model_init_args`; checkpoint compatibility должна отклонять старые несовместимые веса.
8. Обеспечьте атомарную запись артефактов и явную cache policy. Python source hash включается автоматически, но stateful callable и внедрённый encoder должны предоставить стабильный `cache_identity`, описывающий всё влияющее скрытое состояние. Для внешнего nondeterminism фиксируйте revision/version/seed.
9. Выполните `validate --no-check-inputs`, затем validation с inputs, `prepare`, повторный `prepare` для проверки reuse, однопроходный `train` и `evaluate`.
