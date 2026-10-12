"""Поведение командной строки, прежде всего на путях ошибок."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
class CliErrorPathTests(unittest.TestCase):
    """The error path must report the error, not fail inside the handler."""

    def test_a_broken_config_produces_a_message(self) -> None:
        import tempfile

        from typer.testing import CliRunner

        from tgseqloc.cli import app

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            # 1e-4 is a string in YAML, not a float: exactly the mistake that
            # exposed the handler itself raising AttributeError.
            path.write_text("training:\n  learning_rate: 1e-4\n", encoding="utf-8")
            result = CliRunner().invoke(app, ["validate", "-c", str(path)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertNotIsInstance(result.exception, AttributeError)


if __name__ == "__main__":
    unittest.main()
