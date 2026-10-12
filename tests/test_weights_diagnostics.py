"""Weight manifest handling and the pre-flight report."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_app_config
from tgseqloc.config import ComponentConfig
from tgseqloc.diagnostics import ERROR, OK, WARN, diagnose, format_report, required_weights
from tgseqloc.weights import WeightError, check, fetch, load_manifest

MANIFEST = """
version: 1
weights:
  local_thing:
    kind: file
    path: weights/thing.bin
    sha256: {sha}
    size_bytes: {size}
  remote_model:
    kind: huggingface
    repo_id: some/model
"""


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def write(self, text: str) -> Path:
        path = self.root / "weights" / "manifest.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_relative_paths_resolve_against_the_repository_root(self) -> None:
        manifest = self.write(MANIFEST.format(sha="ab" * 32, size=4))
        specs = load_manifest(manifest, root=self.root)
        self.assertEqual(specs["local_thing"].path, self.root / "weights" / "thing.bin")
        self.assertEqual(specs["remote_model"].repo_id, "some/model")

    def test_malformed_manifests_say_what_is_wrong(self) -> None:
        cases = (
            ("version: 99\nweights: {}\n", "version 99"),
            ("version: 1\nweights:\n  a:\n    kind: magic\n", "unsupported kind"),
            ("version: 1\nweights:\n  a:\n    kind: file\n", "needs a path"),
            ("version: 1\nweights:\n  a:\n    kind: huggingface\n", "needs a repo_id"),
        )
        for text, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(WeightError, expected):
                    load_manifest(self.write(text), root=self.root)

    def test_absent_manifest_is_reported_not_ignored(self) -> None:
        with self.assertRaisesRegex(WeightError, "manifest is absent"):
            load_manifest(self.root / "nope.yaml")

    def test_check_distinguishes_missing_wrong_size_and_present(self) -> None:
        target = self.root / "weights" / "thing.bin"
        manifest = self.write(MANIFEST.format(sha="ab" * 32, size=4))
        spec = load_manifest(manifest, root=self.root)["local_thing"]

        status = check(spec)
        self.assertFalse(status.ok)
        self.assertIn("missing", status.detail)
        self.assertIn("weights sync", status.fix or "")

        target.write_bytes(b"toolong")
        self.assertIn("size 7 != expected 4", check(spec).detail)

        target.write_bytes(b"data")
        self.assertTrue(check(spec).ok)

    def test_hashing_is_opt_in_and_catches_corruption(self) -> None:
        """Presence alone is cheap; hashing reads the file, so it is a choice."""

        manifest = self.write(MANIFEST.format(sha="ab" * 32, size=4))
        spec = load_manifest(manifest, root=self.root)["local_thing"]
        (self.root / "weights" / "thing.bin").write_bytes(b"data")

        self.assertTrue(check(spec, verify_hash=False).ok)
        corrupted = check(spec, verify_hash=True)
        self.assertFalse(corrupted.ok)
        self.assertIn("sha256", corrupted.detail)

    def test_fetch_refuses_a_download_that_fails_its_checksum(self) -> None:
        source = self.root / "source.bin"
        source.write_bytes(b"data")
        manifest = self.write(
            "version: 1\n"
            "weights:\n"
            "  local_thing:\n"
            "    kind: file\n"
            "    path: weights/thing.bin\n"
            f"    url: {source.as_uri()}\n"
            f"    sha256: {'ab' * 32}\n"
        )
        spec = load_manifest(manifest, root=self.root)["local_thing"]
        with self.assertRaisesRegex(WeightError, "expected"):
            fetch(spec)
        # A failed download must not leave a half-written file behind.
        self.assertFalse((self.root / "weights" / "thing.bin").exists())

    def test_missing_file_without_url_says_to_place_it_manually(self) -> None:
        manifest = self.write(
            "version: 1\nweights:\n  local_thing:\n    kind: file\n    path: w/thing.bin\n"
        )
        spec = load_manifest(manifest, root=self.root)["local_thing"]
        with self.assertRaisesRegex(WeightError, "no url"):
            fetch(spec)


class RequiredWeightsTests(unittest.TestCase):
    def test_enabled_stage_weights_are_collected_with_their_section(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config, _ = make_app_config(Path(temporary.name), frame_count=4)

        self.assertEqual(required_weights(config), {})
        config.segmentation = ComponentConfig(backend="yolo11x_seg", weights="yolo11x_seg")
        self.assertEqual(required_weights(config), {"yolo11x_seg": "segmentation"})

    def test_disabled_stage_requires_nothing(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config, _ = make_app_config(Path(temporary.name), frame_count=4)
        config.segmentation = ComponentConfig(backend=None, weights=None)
        self.assertEqual(required_weights(config), {})


class DiagnoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config, _ = make_app_config(self.root, frame_count=4)

    def test_report_covers_every_check_and_never_raises(self) -> None:
        report = diagnose(
            self.config,
            manifest_path=self.root / "absent.yaml",
            check_inputs=True,
        )
        names = [result.name for result in report.results]
        for expected in ("components", "dataset inputs", "weight manifest", "device"):
            self.assertIn(expected, names)

    def test_one_broken_check_does_not_hide_the_others(self) -> None:
        """A fresh clone is missing several things; all of them must show up."""

        self.config.dataset.root = self.root / "gone"
        report = diagnose(
            self.config, manifest_path=self.root / "absent.yaml", check_inputs=True
        )
        failed = {result.name for result in report.results if result.failed}
        self.assertIn("dataset inputs", failed)
        self.assertIn("weight manifest", failed)
        self.assertFalse(report.ok)
        # The checks after the failures still ran.
        self.assertIn("device", {result.name for result in report.results})

    def test_report_is_ok_when_only_warnings_are_present(self) -> None:
        report = diagnose(self.config, manifest_path=self.root / "absent.yaml")
        report.results = [r for r in report.results if r.status != ERROR]
        self.assertTrue(report.ok)

    def test_format_shows_fixes_for_problems_only(self) -> None:
        report = diagnose(
            self.config, manifest_path=self.root / "absent.yaml", check_inputs=False
        )
        text = format_report(report)
        self.assertIn("[WARN]", text)
        self.assertIn("dataset inputs", text)

    def test_json_shape_counts_errors_and_warnings(self) -> None:
        payload = diagnose(
            self.config, manifest_path=self.root / "absent.yaml", check_inputs=False
        ).to_dict()
        self.assertIn("ok", payload)
        self.assertEqual(
            payload["errors"],
            sum(1 for c in payload["checks"] if c["status"] == ERROR),
        )


if __name__ == "__main__":
    unittest.main()
