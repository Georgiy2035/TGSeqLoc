"""Model stages that run once and leave their output on disk.

OCR and segmentation are the expensive part of preparation and they do not
depend on each other, so running them as one pass forces both to be redone
whenever either changes -- and forces both to live in one environment, which
PaddleOCR and torch do not do gracefully.

Each stage here writes one JSON file per frame through :class:`ArtifactStore`,
so a stage is resumable, is skipped entirely when its inputs and settings have
not changed, and can be run from a different machine or environment than the
stage that consumes it.

The dynamics stage is deliberately not cached: it only intersects two
artifacts, costs nothing next to a model, and keeping it live means the
threshold can be changed without invalidating anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tgseqloc.artifacts import ArtifactStore
from tgseqloc.data.formats import FrameMasks, FrameText, TextDetection

STAGES = ("ocr", "segmentation")


@dataclass(frozen=True, slots=True)
class StageResult:
    """What one stage run did, for reporting and for tests."""

    stage: str
    total: int
    written: int
    reused: int
    failed: int

    @property
    def complete(self) -> bool:
        return self.failed == 0 and self.written + self.reused == self.total


def frame_text_to_json(value: FrameText) -> dict[str, Any]:
    return {
        "image_size": list(value.image_size),
        "model_identity": dict(value.model_identity),
        "detections": [
            {
                "box": list(detection.box),
                "text": detection.text,
                "confidence": detection.confidence,
                **(
                    {"dynamic_score": detection.dynamic_score}
                    if detection.dynamic_score is not None
                    else {}
                ),
            }
            for detection in value.detections
        ],
    }


def frame_text_from_json(payload: Mapping[str, Any]) -> FrameText:
    return FrameText(
        detections=tuple(
            TextDetection(
                box=tuple(row["box"]),
                text=row["text"],
                confidence=row["confidence"],
                dynamic_score=row.get("dynamic_score"),
            )
            for row in payload.get("detections", ())
        ),
        image_size=tuple(payload["image_size"]),
        model_identity=payload.get("model_identity", {}),
    )


def frame_masks_to_json(value: FrameMasks) -> dict[str, Any]:
    return {
        "image_size": list(value.image_size),
        "model_identity": dict(value.model_identity),
        "dynamic_rle": value.dynamic_rle,
    }


def frame_masks_from_json(payload: Mapping[str, Any]) -> FrameMasks:
    return FrameMasks(
        dynamic_rle=payload.get("dynamic_rle"),
        image_size=tuple(payload["image_size"]),
        model_identity=payload.get("model_identity", {}),
    )


CODECS: dict[str, tuple[Callable[[Any], dict], Callable[[Mapping], Any]]] = {
    "ocr": (frame_text_to_json, frame_text_from_json),
    "segmentation": (frame_masks_to_json, frame_masks_from_json),
}


def stage_root(prepared_root: Path, dataset: str, stage: str) -> Path:
    return Path(prepared_root) / dataset / "stages" / stage


def artifact_name(sequence: str, frame_stem: str) -> str:
    return f"{sequence}/{frame_stem}.json"


def run_stage(
    stage: str,
    model: Any,
    frames: Sequence[tuple[str, str, Path]],
    root: Path,
    *,
    source_identity: Callable[[Path], Any] | None = None,
    rebuild: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> StageResult:
    """Run ``model`` over ``frames``, skipping what is already current.

    ``frames`` carries ``(sequence, frame_stem, image_path)``. A frame is
    rebuilt when its image or the model's identity changed; everything else is
    reused, which is what makes an interrupted run cheap to repeat.
    """

    if stage not in CODECS:
        raise ValueError(f"unknown stage {stage!r}; available: {', '.join(STAGES)}")
    encode, _ = CODECS[stage]
    store = ArtifactStore(root)
    identity = dict(getattr(model, "cache_identity", {}) or {})
    loaded = False

    written = reused = failed = 0
    for index, (sequence, stem, image_path) in enumerate(frames):
        name = artifact_name(sequence, stem)
        inputs = {
            "stage": stage,
            "model": identity,
            "source": source_identity(image_path) if source_identity else str(image_path),
        }
        if not rebuild and store.is_compatible(name, inputs, verify_content=False):
            reused += 1
        else:
            if not loaded:
                # Deferred so that a fully cached run never loads the model.
                model.load()
                loaded = True
            try:
                value = model.predict_one(image_path)
            except Exception as error:  # noqa: BLE001 - one frame must not end the run
                failed += 1
                _write_json(
                    store.path(name),
                    {"error": f"{type(error).__name__}: {error}"},
                )
                continue
            _write_json(store.path(name), encode(value))
            store.record(name, inputs, write_manifest=False)
            written += 1
        if on_progress is not None:
            on_progress(index + 1, len(frames))
    store.write_manifest()
    return StageResult(stage, len(frames), written, reused, failed)


def build_stage_model(stage: str, config: Any) -> Any:
    """Create the backend a stage is configured to use, weights resolved.

    Weights are named logically in the configuration, so the manifest is what
    turns that name into a path here rather than at every call site.
    """

    from tgseqloc.registry import registry

    kinds = {"ocr": ("ocr", "ocr"), "segmentation": ("segmenter", "segmentation")}
    if stage not in kinds:
        raise ValueError(f"unknown stage {stage!r}; available: {', '.join(STAGES)}")
    kind, section = kinds[stage]
    component = getattr(config, section)
    if not component.enabled:
        raise RuntimeError(
            f"{section}.backend is not set; enable it to run the {stage} stage"
        )
    factory = registry.get(kind, str(component.backend))
    params = dict(component.params)
    if component.weights:
        from tgseqloc.diagnostics import default_manifest_path
        from tgseqloc.weights import load_manifest, resolve

        specs = load_manifest(default_manifest_path())
        if component.weights not in specs:
            raise RuntimeError(
                f"{section}.weights={component.weights!r} is not in the weight manifest"
            )
        params["weights_path"] = resolve(specs[component.weights])
    device = config.runtime.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    return factory(device=device, **params)


def run_configured_stage(
    stage: str,
    config: Any,
    *,
    rebuild: bool = False,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> StageResult:
    """Run one stage over the frames the configured dataset adapter discovers."""

    from tgseqloc.preparation.v4rl import source_file_identity
    from tgseqloc.registry import registry

    adapter = registry.get("dataset", config.dataset.adapter)
    discover = getattr(adapter, "discover_frames", None)
    if discover is None:
        raise RuntimeError(
            f"dataset adapter {config.dataset.adapter!r} does not expose discover_frames"
        )
    frames = list(discover(config))
    if limit:
        frames = frames[:limit]
    root = stage_root(config.dataset.prepared_root, config.dataset.adapter, stage)
    return run_stage(
        stage,
        build_stage_model(stage, config),
        frames,
        root,
        source_identity=source_file_identity,
        rebuild=rebuild,
        on_progress=on_progress,
    )


def read_stage(stage: str, root: Path, sequence: str, frame_stem: str) -> Any:
    """Load one cached stage artifact, or raise if it is absent or failed."""

    if stage not in CODECS:
        raise ValueError(f"unknown stage {stage!r}; available: {', '.join(STAGES)}")
    _, decode = CODECS[stage]
    path = Path(root) / artifact_name(sequence, frame_stem)
    if not path.is_file():
        raise FileNotFoundError(
            f"{stage} artifact is absent: {path}; run `tgseqloc stage {stage}` first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "error" in payload:
        raise RuntimeError(f"{stage} failed for {frame_stem}: {payload['error']}")
    return decode(payload)


def stage_status(root: Path, frames: Iterable[tuple[str, str, Path]]) -> dict[str, int]:
    """Count how many frames already have an artifact, without reading them."""

    present = missing = 0
    for sequence, stem, _ in frames:
        if (Path(root) / artifact_name(sequence, stem)).is_file():
            present += 1
        else:
            missing += 1
    return {"present": present, "missing": missing}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write atomically: an interrupted run must not leave a half file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
