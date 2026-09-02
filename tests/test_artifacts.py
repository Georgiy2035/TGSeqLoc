from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tgseqloc.artifacts import (
    ArtifactError,
    ArtifactPolicy,
    ArtifactStore,
    IncompatibleArtifactError,
    canonical_json,
    fingerprint,
)


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ArtifactStore(self.root)
        self.path = self.store.path("nested/value.bin")
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"original")
        self.inputs = {"model": "v1", "options": {3, 1}}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_canonical_fingerprint_is_stable(self) -> None:
        first = {"b": [2, 1], "a": Path("x/y"), "set": {"z", "a"}}
        second = {"set": {"a", "z"}, "a": Path("x/y"), "b": [2, 1]}
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(fingerprint(first), fingerprint(second))
        with self.assertRaisesRegex(TypeError, "string keys"):
            canonical_json({1: "invalid"})

    def test_record_round_trip_and_cache_hit(self) -> None:
        record = self.store.record("nested/value.bin", self.inputs)
        reloaded = ArtifactStore(self.root)
        self.assertEqual(reloaded.manifest.files["nested/value.bin"], record)
        self.assertTrue(reloaded.is_compatible("nested/value.bin", self.inputs))
        self.assertFalse(
            reloaded.should_build(
                "nested/value.bin",
                self.inputs,
                ArtifactPolicy.REUSE_IF_COMPATIBLE,
            )
        )
        self.assertEqual(
            reloaded.require_compatible("nested/value.bin", self.inputs), self.path
        )

    def test_invalidates_on_inputs_size_content_and_missing_file(self) -> None:
        self.store.record("nested/value.bin", self.inputs)
        self.assertEqual(
            self.store.compatibility_reason("nested/value.bin", {"model": "v2"}),
            "compatibility fingerprint differs",
        )
        self.path.write_bytes(b"different-length")
        self.assertEqual(
            self.store.compatibility_reason("nested/value.bin", self.inputs),
            "artifact size differs",
        )
        self.path.write_bytes(b"mutated!")
        self.assertEqual(
            self.store.compatibility_reason("nested/value.bin", self.inputs),
            "artifact content hash differs",
        )
        self.path.unlink()
        self.assertEqual(
            self.store.compatibility_reason("nested/value.bin", self.inputs),
            "artifact file is missing",
        )

    def test_policies_and_safe_paths(self) -> None:
        self.store.record("nested/value.bin", self.inputs)
        self.assertTrue(
            self.store.should_build("nested/value.bin", self.inputs, "rebuild")
        )
        self.assertFalse(
            self.store.should_build("nested/value.bin", self.inputs, "require_existing")
        )
        with self.assertRaisesRegex(IncompatibleArtifactError, "fingerprint differs"):
            self.store.should_build(
                "nested/value.bin", {"model": "changed"}, "require_existing"
            )
        with self.assertRaisesRegex(ArtifactError, "unknown artifact policy"):
            self.store.should_build("nested/value.bin", self.inputs, "invalid")
        with self.assertRaisesRegex(ArtifactError, "stay within"):
            self.store.path("../escape")
        with self.assertRaisesRegex(ArtifactError, "stay within"):
            self.store.path(self.root / "absolute")


if __name__ == "__main__":
    unittest.main()
