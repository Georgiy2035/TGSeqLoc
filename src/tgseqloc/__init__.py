"""TGSeqLoc: text-graph place recognition building blocks.

Names resolve lazily. Importing the package used to pull in the component
registry and, through it, torch, which made a stage that needs neither — OCR
and segmentation produce plain data — unrunnable in an environment built for
that model alone. Public names are unchanged; they are just materialized on
first use.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

_EXPORTS = {
    "ArtifactError": "artifacts",
    "ArtifactManifest": "artifacts",
    "ArtifactPolicy": "artifacts",
    "ArtifactRecord": "artifacts",
    "ArtifactStore": "artifacts",
    "IncompatibleArtifactError": "artifacts",
    "canonical_json": "artifacts",
    "file_sha256": "artifacts",
    "fingerprint": "artifacts",
    "AppConfig": "config",
    "CacheConfig": "config",
    "ComponentConfig": "config",
    "ConfigError": "config",
    "DatasetConfig": "config",
    "ModelConfig": "config",
    "OutputConfig": "config",
    "PreprocessConfig": "config",
    "RetrievalConfig": "config",
    "RuntimeConfig": "config",
    "SourceConfig": "config",
    "SourcesConfig": "config",
    "TrainingConfig": "config",
    "dump_config": "config",
    "load_config": "config",
    "component_map": "components",
    "register_builtin_components": "components",
    "PipelineRunner": "pipeline",
    "Dataset": "registry",
    "Encoder": "registry",
    "Filter": "registry",
    "Fusion": "registry",
    "GraphEncoder": "registry",
    "Metric": "registry",
    "Miner": "registry",
    "OCRModel": "registry",
    "Registry": "registry",
    "Reranker": "registry",
    "Retriever": "registry",
    "Segmenter": "registry",
    "Source": "registry",
    "TextDynamics": "registry",
    "register": "registry",
    "registry": "registry",
}

__all__ = sorted([*_EXPORTS, "__version__"])


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return __all__
