"""Configuration-driven orchestration for preparation, training, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tgseqloc.components import component_map, register_builtin_components
from tgseqloc.config import AppConfig
from tgseqloc.registry import registry
from tgseqloc.training import Trainer


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


@dataclass(slots=True)
class PipelineRunner:
    """Execute TGSeqLoc stages using one validated experiment configuration."""

    config: AppConfig

    def __post_init__(self) -> None:
        register_builtin_components()
        self.config.validate()
        selected = (
            ("dataset", self.config.dataset.adapter),
            ("source", self.config.sources.ocr),
            ("source", self.config.sources.scene_graph),
            ("filter", self.config.preprocess.text_filter),
            ("encoder", self.config.preprocess.text_encoder_backend),
            ("fusion", self.config.preprocess.fusion),
            ("graph_encoder", self.config.model.graph_encoder),
            ("miner", self.config.training.miner),
            ("retriever", self.config.retrieval.retriever),
            ("metric", self.config.retrieval.metric),
        )
        if self.config.retrieval.reranker is not None:
            selected += (("reranker", self.config.retrieval.reranker),)
        for kind, name in selected:
            registry.get(kind, name)

    @property
    def prepared_data_root(self) -> Path:
        return self.config.dataset.prepared_root / self.config.dataset.adapter

    @property
    def manifest_path(self) -> Path:
        return self.prepared_data_root / "manifest.json"

    @property
    def split_path(self) -> Path:
        return self.prepared_data_root / "mappings" / "temporal_split.json"

    def validate(self, *, check_inputs: bool = True) -> dict[str, Any]:
        """Validate configuration, registered names, and precomputed inputs."""

        result: dict[str, Any] = {
            "valid": True,
            "dataset": self.config.dataset.adapter,
            "components": component_map(),
            "prepared_data_root": str(self.prepared_data_root),
        }
        if check_inputs:
            # Input discovery belongs to the dataset adapter: the orchestrator
            # must not know how any particular dataset is laid out on disk.
            adapter = registry.get("dataset", self.config.dataset.adapter)
            discover = getattr(adapter, "discover_inputs", None)
            if discover is None:
                raise RuntimeError(
                    f"dataset adapter {self.config.dataset.adapter!r} does not expose "
                    "discover_inputs; run with check_inputs disabled or implement it"
                )
            result["source_frame_count"] = discover(self.config)
        return result

    def prepare(self, **kwargs: Any) -> dict[str, Any]:
        """Create or reuse prepared text-graph artifacts."""

        self.validate(check_inputs=True)
        adapter = registry.get("dataset", self.config.dataset.adapter)
        return adapter(
            self.config,
            text_encoder_factory=registry.get(
                "encoder", self.config.preprocess.text_encoder_backend
            ),
            ocr_parser=registry.get("source", self.config.sources.ocr),
            scene_graph_parser=registry.get(
                "source", self.config.sources.scene_graph
            ),
            text_filter=registry.get("filter", self.config.preprocess.text_filter),
            fusion_builder=registry.get("fusion", self.config.preprocess.fusion),
            **kwargs,
        )

    def _prepared_metadata(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepared manifest is absent: {self.manifest_path}; run prepare first"
            )
        if not self.split_path.is_file():
            raise FileNotFoundError(
                f"Prepared temporal split is absent: {self.split_path}; run prepare first"
            )
        manifest = _load_json(self.manifest_path)
        if not manifest.get("output_complete"):
            raise RuntimeError(
                f"Prepared dataset is incomplete according to {self.manifest_path}"
            )
        return manifest, _load_json(self.split_path)

    def trainer(self) -> Trainer:
        """Build the shared trainer against validated prepared artifacts."""

        manifest, split = self._prepared_metadata()
        model_args = asdict(self.config.model)
        model_args.pop("graph_encoder")
        expected_dimensions = {
            "in_dim": int(manifest.get("node_feature_dim", 4)),
            "edge_cont_dim": int(manifest["edge_attr_dim"]),
            "text_emb_dim": int(manifest["text_embedding_dim"]),
        }
        for key, expected in expected_dimensions.items():
            if int(model_args[key]) != expected:
                raise ValueError(
                    f"model.{key}={model_args[key]} is incompatible with "
                    f"prepared manifest value {expected}"
                )
        if bool(model_args["use_text_nodes"]) != bool(
            manifest.get("uses_text_nodes", True)
        ):
            raise ValueError(
                "model.use_text_nodes is incompatible with the prepared manifest"
            )
        model_args.update(
            num_node_classes=max(1, int(manifest["num_obj_classes"])),
            num_edge_classes=max(1, int(manifest["num_edge_classes"])),
        )
        model = registry.create(
            "graph_encoder", self.config.model.graph_encoder, **model_args
        )
        return Trainer(
            self.config,
            self.prepared_data_root,
            manifest=manifest,
            split=split,
            model=model,
            miner=registry.get("miner", self.config.training.miner),
            retriever_factory=registry.get(
                "retriever", self.config.retrieval.retriever
            ),
            metric_factory=registry.get("metric", self.config.retrieval.metric),
        )

    def train(self) -> dict[str, Any]:
        """Train the selected model and evaluate its best checkpoint."""

        trainer = self.trainer()
        resume = self.config.training.resume_from
        return trainer.train(resume_from=resume)

    def evaluate(self, checkpoint: str | Path = "best") -> dict[str, Any]:
        """Evaluate a checkpoint on the configured test protocol."""

        trainer = self.trainer()
        metrics, _ = trainer.evaluate_test(checkpoint, write_results=True)
        return metrics

    def run(self, **prepare_kwargs: Any) -> dict[str, Any]:
        """Run prepare, train, and final evaluation as one reproducible job."""

        manifest = self.prepare(**prepare_kwargs)
        training = self.train()
        return {"manifest": manifest, "training": training}

