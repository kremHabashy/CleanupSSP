"""Step 1: ensure the canonical config template parses and exposes expected sections."""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

REQUIRED_TOP_LEVEL = (
    "project",
    "wandb",
    "paths",
    "experiment",
    "ssp",
    "data",
    "trainer",
    "eval",
    "schedule",
)


class TestConfigTemplate(unittest.TestCase):
    def test_template_exists_and_loads(self):
        self.assertTrue(
            CONFIG_PATH.is_file(),
            f"Missing template: {CONFIG_PATH}",
        )
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(raw, dict)
        for key in REQUIRED_TOP_LEVEL:
            self.assertIn(key, raw, f"config.yaml must define top-level key '{key}'")

    def test_trainer_has_core_keys(self):
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        tr = raw["trainer"]
        for k in ("sampling_modes", "device", "batch_size", "epochs", "lr"):
            self.assertIn(k, tr)


if __name__ == "__main__":
    unittest.main()
