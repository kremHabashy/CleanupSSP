"""Step 2: config loader and path resolution."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


from src.utils import (
    get_project_root,
    load_config,
    resolve_config_paths,
    validate_config,
)


class TestStep2Utils(unittest.TestCase):
    def test_get_project_root(self):
        root = get_project_root()
        self.assertTrue((root / "configs" / "config.yaml").is_file())

    def test_load_default_config(self):
        cfg = load_config()
        self.assertEqual(cfg["experiment"]["name"], "default_run")

    def test_resolve_paths_absolute(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            resolved = resolve_config_paths(cfg, t)
            dr = resolved["paths"]["data_root"]
            self.assertTrue(dr.is_absolute())
            self.assertEqual(dr, (t / "data").resolve())

    def test_validate_rejects_incomplete(self):
        with self.assertRaises(ValueError):
            validate_config({"project": {}})


if __name__ == "__main__":
    unittest.main()
