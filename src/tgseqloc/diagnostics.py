"""Pre-flight checks: everything that can be wrong before a run starts.

The point of this module is that it does not stop at the first problem. A fresh
clone is typically missing several things at once, and fixing them one failed
run at a time is exactly the experience this avoids. Every check therefore
captures its own failure and the report lists all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tgseqloc.config import AppConfig
from tgseqloc.registry import registry
from tgseqloc.weights import WeightError, WeightSpec, WeightStatus, check, load_manifest

OK = "ok"
WARN = "warn"
ERROR = "error"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One diagnosis: what was checked, how it went, and how to fix it."""

    name: str
    status: str
    detail: str
    fix: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == ERROR


@dataclass(slots=True)
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, fix: str | None = None) -> None:
        self.results.append(CheckResult(name, status, detail, fix))

    def run(self, name: str, action: Callable[[], tuple[str, str, str | None]]) -> None:
        """Run one check, turning any exception into a failed result.

        A check that raises must not prevent the remaining checks from running,
        which is the whole reason this helper exists.
        """

        try:
            status, detail, fix = action()
        except Exception as error:  # noqa: BLE001 - a check must never abort the report
            self.add(name, ERROR, f"{type(error).__name__}: {error}", None)
        else:
            self.add(name, status, detail, fix)

    @property
    def ok(self) -> bool:
        return not any(result.failed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": sum(1 for r in self.results if r.status == ERROR),
            "warnings": sum(1 for r in self.results if r.status == WARN),
            "checks": [
                {
                    "name": r.name,
                    "status": r.status,
                    "detail": r.detail,
                    **({"fix": r.fix} if r.fix else {}),
                }
                for r in self.results
            ],
        }


def default_manifest_path(root: Path | None = None) -> Path:
    """Locate weights/manifest.yaml relative to the installed package."""

    base = root or Path(__file__).resolve().parents[2]
    return base / "weights" / "manifest.yaml"


def required_weights(config: AppConfig) -> dict[str, str]:
    """Map each required logical weight name to the section that asked for it.

    Two sources: an explicit ``weights`` field on a swappable stage, and a
    backend that declares what it needs through ``weight_requirements()``.
    """

    required: dict[str, str] = {}
    for section, component in (
        ("segmentation", config.segmentation),
        ("text_dynamics", config.text_dynamics),
    ):
        if component.enabled and component.weights:
            required[component.weights] = section
    for kind, name, section in (
        ("encoder", config.preprocess.text_encoder_backend, "preprocess.text_encoder_backend"),
        ("segmenter", config.segmentation.backend, "segmentation.backend"),
        ("text_dynamics", config.text_dynamics.backend, "text_dynamics.backend"),
    ):
        if not name:
            continue
        try:
            factory = registry.get(kind, name)
        except KeyError:
            continue
        declare = getattr(factory, "weight_requirements", None)
        if declare is None:
            continue
        for logical in declare():
            required.setdefault(str(logical), section)
    return required


def diagnose(
    config: AppConfig,
    *,
    manifest_path: Path | None = None,
    check_inputs: bool = True,
    verify_hashes: bool = False,
) -> Report:
    """Check configuration, components, dataset inputs, weights and device."""

    report = Report()

    def check_components() -> tuple[str, str, str | None]:
        from tgseqloc.components import register_builtin_components

        register_builtin_components()
        selected = [
            ("dataset", config.dataset.adapter),
            ("source", config.sources.ocr),
            ("source", config.sources.scene_graph),
            ("filter", config.preprocess.text_filter),
            ("encoder", config.preprocess.text_encoder_backend),
            ("fusion", config.preprocess.fusion),
            ("graph_encoder", config.model.graph_encoder),
            ("miner", config.training.miner),
            ("retriever", config.retrieval.retriever),
            ("metric", config.retrieval.metric),
        ]
        for section, component in (
            ("segmenter", config.segmentation),
            ("text_dynamics", config.text_dynamics),
        ):
            if component.enabled:
                selected.append((section, str(component.backend)))
        missing = [
            f"{kind}:{name}"
            for kind, name in selected
            if not registry.contains(kind, name)
        ]
        if missing:
            return ERROR, f"not registered: {', '.join(missing)}", "tgseqloc components"
        return OK, f"{len(selected)} components resolve", None

    report.run("components", check_components)

    def check_dataset() -> tuple[str, str, str | None]:
        if not check_inputs:
            return WARN, "skipped (--no-check-inputs)", None
        adapter = registry.get("dataset", config.dataset.adapter)
        discover = getattr(adapter, "discover_inputs", None)
        if discover is None:
            return (
                WARN,
                f"adapter {config.dataset.adapter!r} cannot check inputs",
                None,
            )
        return OK, f"{discover(config)} frames with OCR and scene graph", None

    report.run("dataset inputs", check_dataset)

    specs: dict[str, WeightSpec] = {}
    manifest = manifest_path or default_manifest_path()

    def check_manifest() -> tuple[str, str, str | None]:
        specs.update(load_manifest(manifest))
        return OK, f"{len(specs)} entries in {manifest}", None

    report.run("weight manifest", check_manifest)

    for logical, section in sorted(required_weights(config).items()):
        def check_weight(
            logical: str = logical, section: str = section
        ) -> tuple[str, str, str | None]:
            spec = specs.get(logical)
            if spec is None:
                return (
                    ERROR,
                    f"{section} requires {logical!r}, which the manifest does not define",
                    f"add {logical} to {manifest}",
                )
            status: WeightStatus = check(spec, verify_hash=verify_hashes)
            # A weight the library fetches itself only delays the first run.
            level = OK if status.ok else (WARN if status.auto_fetch else ERROR)
            return level, f"{section}: {status.detail}", status.fix

        report.run(f"weights: {logical}", check_weight)

    def check_encoder_model() -> tuple[str, str, str | None]:
        """The encoder still names its Hugging Face model directly.

        Until that field moves to the shared backend/weights shape, the check
        looks the model up in the cache by id so a missing download is still
        reported here rather than at the first encode.
        """

        from huggingface_hub import snapshot_download

        model = config.preprocess.text_encoder
        try:
            snapshot_download(
                repo_id=model,
                revision=config.preprocess.revision,
                local_files_only=True,
            )
        except Exception:  # noqa: BLE001 - hub raises several unrelated types
            return (
                WARN,
                f"{model} is not in the local cache; it will be downloaded on first use",
                "tgseqloc weights sync",
            )
        return OK, f"{model} is cached", None

    report.run("text encoder model", check_encoder_model)

    def check_device() -> tuple[str, str, str | None]:
        """Verify the device by running on it, not by asking whether it exists.

        ``torch.cuda.is_available()`` answers yes for a GPU whose architecture
        the installed build has no kernels for; the failure then surfaces deep
        inside the first real operation. A one-element matmul settles it here.
        """

        import torch

        requested = config.runtime.device
        wants_cuda = requested == "auto" or requested.startswith("cuda")
        if not wants_cuda:
            return OK, f"{requested} is usable", None
        if not torch.cuda.is_available():
            if requested == "auto":
                return OK, "auto resolves to cpu; no CUDA device", None
            return ERROR, f"{requested} requested but CUDA is unavailable", "runtime.device: cpu"

        target = "cuda" if requested == "auto" else requested
        name = torch.cuda.get_device_name(0)
        try:
            probe = torch.zeros((2, 2), device=target)
            torch.mm(probe, probe).cpu()
        except RuntimeError as error:
            detail = (
                f"{name} is present but unusable with torch {torch.__version__}: "
                f"{str(error).splitlines()[0]}"
            )
            supported = ", ".join(torch.cuda.get_arch_list()) or "none"
            if requested == "auto":
                return WARN, f"{detail}; auto will fall back to cpu. Built for: {supported}", (
                    "install a torch build for this GPU architecture"
                )
            return ERROR, f"{detail}. Built for: {supported}", (
                "install a torch build for this GPU architecture, or set runtime.device: cpu"
            )
        return OK, f"{target} works ({name})", None

    report.run("device", check_device)

    def check_prepared() -> tuple[str, str, str | None]:
        root = config.dataset.prepared_root / config.dataset.adapter
        manifest_file = root / "manifest.json"
        if not manifest_file.is_file():
            return WARN, f"no prepared artifacts yet at {root}", "tgseqloc prepare"
        import json

        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not payload.get("output_complete"):
            return ERROR, f"prepared artifacts are incomplete: {manifest_file}", "tgseqloc prepare"
        return OK, f"prepared artifacts complete in {root}", None

    report.run("prepared artifacts", check_prepared)
    return report


def format_report(report: Report) -> str:
    """Render the report as aligned lines, failures carrying their fix."""

    symbols = {OK: "OK  ", WARN: "WARN", ERROR: "FAIL"}
    width = max((len(r.name) for r in report.results), default=0)
    lines = [
        f"[{symbols.get(r.status, r.status)}] {r.name.ljust(width)}  {r.detail}"
        + (f"\n{' ' * (width + 9)}fix: {r.fix}" if r.fix and r.status != OK else "")
        for r in report.results
    ]
    errors = sum(1 for r in report.results if r.status == ERROR)
    warnings = sum(1 for r in report.results if r.status == WARN)
    lines.append(
        "\nвсё в порядке" if report.ok and not warnings
        else f"\nошибок: {errors}, предупреждений: {warnings}"
    )
    return "\n".join(lines)
