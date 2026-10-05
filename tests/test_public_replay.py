"""Synthetic and packaging tests; no clinical records are loaded."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class ReplaySafetyTests(unittest.TestCase):
    def test_help_does_not_require_data(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/replay.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verify", result.stdout)

    def test_workdir_cannot_be_package_or_ancestor(self):
        import replay
        for path in [ROOT, ROOT / "private", ROOT.parent]:
            with self.assertRaises(ValueError):
                replay.validate_workdir(path, ROOT)

    def test_configuration_is_resolved_relative_to_config(self):
        import replay
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / "runtime.json"
            path.write_text(json.dumps({"raw_data": {"inspire_archive": "raw/inspire.zip"}}))
            config = replay.read_runtime(path)
            self.assertEqual(config["raw_data"]["inspire_archive"],
                             str((folder / "raw/inspire.zip").resolve()))

    def test_help_subcommands_are_side_effect_free(self):
        for command in ["init", "verify", "etl", "analysis", "supplemental",
                        "finalize", "raw-audit", "ties", "retained-correction"]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/replay.py"), command, "--help"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, (command, result.stderr))


if __name__ == "__main__":
    unittest.main()
