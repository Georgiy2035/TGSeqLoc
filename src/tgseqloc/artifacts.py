"""Content-addressed metadata and compatibility policies for artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Mapping


class ArtifactError(RuntimeError):
    """Base error for artifact persistence and compatibility failures."""


class IncompatibleArtifactError(ArtifactError):
    """Raised when a required compatible artifact does not exist."""


class ArtifactPolicy(StrEnum):
    """How an existing artifact should be handled."""

    REUSE_IF_COMPATIBLE = "reuse_if_compatible"
    REQUIRE_EXISTING = "require_existing"
    REBUILD = "rebuild"


def canonical_json(value: Any) -> str:
    """Serialize supported values deterministically for hashing."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any) -> str:
    """Return the SHA-256 digest of a value's canonical JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file and return its SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Manifest metadata for one file."""

    compatibility: str
    sha256: str
    size: int


@dataclass(slots=True)
class ArtifactManifest:
    """Versioned collection of artifact records."""

    version: int = 1
    files: dict[str, ArtifactRecord] = field(default_factory=dict)


class ArtifactStore:
    """Filesystem artifact store with an atomic JSON manifest.

    Compatibility is based on caller-provided inputs (configuration, source
    fingerprints, model versions, and so on), while file hashes detect
    accidental mutation of otherwise compatible output.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        manifest_name: str = "manifest.json",
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / manifest_name
        self._manifest: ArtifactManifest | None = None

    @property
    def manifest(self) -> ArtifactManifest:
        """Lazily read the current manifest."""

        if self._manifest is None:
            self._manifest = self.read_manifest()
        return self._manifest

    def path(self, relative_path: str | Path) -> Path:
        """Resolve a safe path below the store root."""

        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError(f"artifact path must stay within store: {relative}")
        return self.root / relative

    def read_manifest(self) -> ArtifactManifest:
        """Read and validate the manifest, or return an empty one."""

        if not self.manifest_path.exists():
            return ArtifactManifest()
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != 1:
                raise ArtifactError("unsupported or malformed artifact manifest")
            raw_files = raw.get("files", {})
            if not isinstance(raw_files, dict):
                raise ArtifactError("manifest files must be an object")
            records: dict[str, ArtifactRecord] = {}
            for name, record in raw_files.items():
                if not isinstance(name, str) or not isinstance(record, dict):
                    raise ArtifactError("malformed artifact manifest record")
                records[name] = ArtifactRecord(
                    compatibility=str(record["compatibility"]),
                    sha256=str(record["sha256"]),
                    size=int(record["size"]),
                )
            return ArtifactManifest(version=1, files=records)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"cannot read manifest {self.manifest_path}: {exc}") from exc

    def write_manifest(self) -> None:
        """Atomically persist the in-memory manifest."""

        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.manifest.version,
            "files": {
                name: asdict(record)
                for name, record in sorted(self.manifest.files.items())
            },
        }
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    def record(
        self,
        relative_path: str | Path,
        compatibility_inputs: Any,
        *,
        write_manifest: bool = True,
    ) -> ArtifactRecord:
        """Fingerprint an existing file and add it to the manifest."""

        path = self.path(relative_path)
        if not path.is_file():
            raise ArtifactError(f"artifact file does not exist: {path}")
        name = Path(relative_path).as_posix()
        record = ArtifactRecord(
            compatibility=fingerprint(compatibility_inputs),
            sha256=file_sha256(path),
            size=path.stat().st_size,
        )
        self.manifest.files[name] = record
        if write_manifest:
            self.write_manifest()
        return record

    def compatibility_reason(
        self,
        relative_path: str | Path,
        compatibility_inputs: Any,
        *,
        verify_content: bool = True,
    ) -> str | None:
        """Return ``None`` when compatible, otherwise a helpful reason."""

        path = self.path(relative_path)
        name = Path(relative_path).as_posix()
        record = self.manifest.files.get(name)
        if record is None:
            return "no manifest record"
        if not path.is_file():
            return "artifact file is missing"
        if record.compatibility != fingerprint(compatibility_inputs):
            return "compatibility fingerprint differs"
        if path.stat().st_size != record.size:
            return "artifact size differs"
        if verify_content and file_sha256(path) != record.sha256:
            return "artifact content hash differs"
        return None

    def is_compatible(
        self,
        relative_path: str | Path,
        compatibility_inputs: Any,
        *,
        verify_content: bool = True,
    ) -> bool:
        """Return whether a file and its manifest record are compatible."""

        return (
            self.compatibility_reason(
                relative_path,
                compatibility_inputs,
                verify_content=verify_content,
            )
            is None
        )

    def require_compatible(
        self,
        relative_path: str | Path,
        compatibility_inputs: Any,
        *,
        verify_content: bool = True,
    ) -> Path:
        """Return the artifact path or raise with its incompatibility reason."""

        reason = self.compatibility_reason(
            relative_path,
            compatibility_inputs,
            verify_content=verify_content,
        )
        if reason is not None:
            raise IncompatibleArtifactError(
                f"artifact {Path(relative_path).as_posix()!r} is unavailable: {reason}"
            )
        return self.path(relative_path)

    def should_build(
        self,
        relative_path: str | Path,
        compatibility_inputs: Any,
        policy: ArtifactPolicy | str = ArtifactPolicy.REUSE_IF_COMPATIBLE,
    ) -> bool:
        """Apply a policy and return whether the caller should build the file."""

        try:
            selected = ArtifactPolicy(policy)
        except ValueError as exc:
            choices = ", ".join(item.value for item in ArtifactPolicy)
            raise ArtifactError(f"unknown artifact policy {policy!r}; available: {choices}") from exc
        if selected is ArtifactPolicy.REBUILD:
            return True
        if selected is ArtifactPolicy.REQUIRE_EXISTING:
            self.require_compatible(relative_path, compatibility_inputs)
            return False
        return not self.is_compatible(relative_path, compatibility_inputs)


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")
