"""Readable views of pipeline output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_app_config
from tgseqloc.config import ComponentConfig
from tgseqloc.inspection import (
    format_summary,
    frame_detail,
    graph_overview,
    inspect,
    stage_overview,
)
from tgseqloc.pipeline import PipelineRunner


class EmptyProjectTests(unittest.TestCase):
    """Before anything runs, the report must say so rather than fail."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config, _ = make_app_config(Path(self.temporary.name), frame_count=4)

    def test_absent_stages_report_zero_artifacts(self) -> None:
        overview = stage_overview(self.config)
        self.assertEqual(overview["ocr"]["artifacts"], 0)
        self.assertEqual(overview["segmentation"]["artifacts"], 0)

    def test_unprepared_dataset_says_what_to_run(self) -> None:
        self.assertIn("prepare", graph_overview(self.config)["hint"])

    def test_summary_renders_without_any_artifacts(self) -> None:
        text = format_summary(inspect(self.config))
        self.assertIn("СТАДИИ", text)
        self.assertIn("ГРАФЫ", text)

    def test_disabled_dynamics_is_reported_as_such(self) -> None:
        self.assertFalse(inspect(self.config)["dynamics"]["enabled"])


class PreparedProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config, _ = make_app_config(Path(self.temporary.name), frame_count=6)
        self.config.preprocess.text_encoder = ComponentConfig(
            backend="char_ngram", params={"embedding_dim": 16}
        )
        self.config.model.text_emb_dim = 16
        PipelineRunner(self.config).prepare()

    def test_graph_overview_counts_what_was_built(self) -> None:
        overview = graph_overview(self.config)
        self.assertTrue(overview["prepared"])
        # make_app_config строит frame_count кадров на каждую из двух
        # последовательностей.
        self.assertEqual(overview["graphs"], 12)
        self.assertGreater(overview["text_nodes"], 0)
        self.assertEqual(overview["text_embedding_dim"], 16)

    def test_frequent_strings_are_reported(self) -> None:
        top = graph_overview(self.config)["top_texts"]
        self.assertTrue(top)
        self.assertTrue(any("shop" in text for text, _ in top))

    def test_frame_detail_covers_stages_and_graph(self) -> None:
        stem = next(
            (self.config.dataset.prepared_root / "v4rl" / "seq1").glob("*.pt")
        ).stem
        detail = frame_detail(self.config, "seq1", stem)
        # No stage was run here, so those must be reported as unavailable
        # rather than silently missing.
        self.assertIn("unavailable", detail["ocr"])
        self.assertIn("texts", detail["graph"])
        self.assertGreater(detail["graph"]["nodes"], 0)

    def test_frame_detail_of_an_unknown_frame_points_at_prepare(self) -> None:
        detail = frame_detail(self.config, "seq1", "999999_0000000000000000000")
        self.assertIn("unavailable", detail["graph"])

    def test_summary_mentions_the_graphs(self) -> None:
        text = format_summary(inspect(self.config))
        self.assertIn("графов 12", text)


if __name__ == "__main__":
    unittest.main()
