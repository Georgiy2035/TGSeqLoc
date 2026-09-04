"""Model weights: a manifest of logical names, their state, and fetching them.

Configuration refers to weights by logical name rather than by path, so an
experiment stays runnable on a machine where the files live somewhere else.
The manifest maps those names to a source (a URL or a Hugging Face repo) and to
the checksum a correct download has.

Presence is checked without loading anything, which is what lets ``doctor``
report every missing file before the first model touches the GPU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

MANIFEST_VERSION = 1
CHUNK = 1 << 20


class WeightError(RuntimeError):
    """Raised when a manifest is malformed or a fetch cannot be completed."""


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """One named weight: where it comes from and how to recognize it."""

    name: str
    kind: str
    path: Path | None = None
    url: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    repo_id: str | None = None
    revision: str | None = None
    note: str = ""

    @property
    def is_file(self) -> bool:
        return self.kind == "file"


@dataclass(frozen=True, slots=True)
class WeightStatus:
    """Outcome of checking one weight, including how to fix it."""

    name: str
    ok: bool
    detail: str
    fix: str | None = None


def _as_path(root: Path, value: Any, name: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def load_manifest(
    manifest_path: str | Path, *, root: str | Path | None = None
) -> dict[str, WeightSpec]:
    """Parse the weight manifest, resolving relative paths against ``root``."""

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise WeightError(f"weight manifest is absent: {manifest_path}")
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise WeightError(f"cannot parse {manifest_path}: {error}") from error
    if not isinstance(raw, Mapping):
        raise WeightError(f"{manifest_path} must contain a mapping")
    version = int(raw.get("version", 0))
    if version != MANIFEST_VERSION:
        raise WeightError(
            f"{manifest_path} has version {version}, expected {MANIFEST_VERSION}"
        )
    base = Path(root) if root is not None else manifest_path.parent.parent
    entries = raw.get("weights") or {}
    if not isinstance(entries, Mapping):
        raise WeightError(f"{manifest_path}: 'weights' must be a mapping")

    specs: dict[str, WeightSpec] = {}
    for name, payload in entries.items():
        if not isinstance(payload, Mapping):
            raise WeightError(f"{manifest_path}: entry {name!r} must be a mapping")
        kind = str(payload.get("kind", "")).strip()
        if kind not in ("file", "huggingface"):
            raise WeightError(
                f"{manifest_path}: entry {name!r} has unsupported kind {kind!r}; "
                "use 'file' or 'huggingface'"
            )
        if kind == "file":
            if not payload.get("path"):
                raise WeightError(f"{manifest_path}: file entry {name!r} needs a path")
        elif not payload.get("repo_id"):
            raise WeightError(
                f"{manifest_path}: huggingface entry {name!r} needs a repo_id"
            )
        specs[str(name)] = WeightSpec(
            name=str(name),
            kind=kind,
            path=_as_path(base, payload["path"], str(name)) if kind == "file" else None,
            url=payload.get("url"),
            sha256=(str(payload["sha256"]).lower() if payload.get("sha256") else None),
            size_bytes=(int(payload["size_bytes"]) if payload.get("size_bytes") else None),
            repo_id=payload.get("repo_id"),
            revision=payload.get("revision"),
            note=str(payload.get("note", "")),
        )
    return specs


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def check(spec: WeightSpec, *, verify_hash: bool = False) -> WeightStatus:
    """Report whether a weight is usable, without loading any model.

    Hashing is opt-in: it reads the whole file, which is wasteful when the
    caller only wants to know what is missing.
    """

    if spec.is_file:
        return _check_file(spec, verify_hash=verify_hash)
    return _check_huggingface(spec)


def _check_file(spec: WeightSpec, *, verify_hash: bool) -> WeightStatus:
    assert spec.path is not None
    fix = f"tgseqloc weights sync --name {spec.name}"
    if not spec.path.is_file():
        return WeightStatus(spec.name, False, f"missing: {spec.path}", fix)
    if spec.size_bytes is not None:
        actual = spec.path.stat().st_size
        if actual != spec.size_bytes:
            return WeightStatus(
                spec.name,
                False,
                f"size {actual} != expected {spec.size_bytes}: {spec.path}",
                fix,
            )
    if verify_hash and spec.sha256:
        actual = file_sha256(spec.path)
        if actual != spec.sha256:
            return WeightStatus(
                spec.name,
                False,
                f"sha256 {actual[:12]}... != expected {spec.sha256[:12]}...",
                fix,
            )
        return WeightStatus(spec.name, True, f"present, sha256 verified: {spec.path}")
    return WeightStatus(spec.name, True, f"present: {spec.path}")


def _check_huggingface(spec: WeightSpec) -> WeightStatus:
    fix = f"tgseqloc weights sync --name {spec.name}"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return WeightStatus(
            spec.name, False, "huggingface-hub is not installed", "uv sync"
        )
    try:
        location = snapshot_download(
            repo_id=str(spec.repo_id),
            revision=spec.revision,
            local_files_only=True,
        )
    except Exception as error:  # noqa: BLE001 - hub raises several unrelated types
        return WeightStatus(
            spec.name,
            False,
            f"not in the local cache: {spec.repo_id}"
            + (f"@{spec.revision}" if spec.revision else "")
            + f" ({type(error).__name__})",
            fix,
        )
    return WeightStatus(spec.name, True, f"cached: {location}")


def fetch(spec: WeightSpec) -> Path:
    """Download one weight, verifying it if the manifest says what to expect."""

    if not spec.is_file:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(repo_id=str(spec.repo_id), revision=spec.revision)
        )

    assert spec.path is not None
    if not spec.url:
        raise WeightError(
            f"{spec.name} is missing and the manifest has no url to fetch it from; "
            f"place the file at {spec.path} manually"
        )
    import urllib.request

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = spec.path.with_suffix(spec.path.suffix + ".part")
    with urllib.request.urlopen(spec.url) as response, temporary.open("wb") as handle:
        while block := response.read(CHUNK):
            handle.write(block)
    if spec.sha256:
        actual = file_sha256(temporary)
        if actual != spec.sha256:
            temporary.unlink(missing_ok=True)
            raise WeightError(
                f"{spec.name} downloaded from {spec.url} has sha256 {actual}, "
                f"expected {spec.sha256}"
            )
    temporary.replace(spec.path)
    return spec.path


def resolve(spec: WeightSpec) -> Path:
    """Return the local path of a present weight, or explain why there is none."""

    status = check(spec)
    if not status.ok:
        raise WeightError(f"{spec.name}: {status.detail}")
    if spec.is_file:
        assert spec.path is not None
        return spec.path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(spec.repo_id), revision=spec.revision, local_files_only=True
        )
    )
