"""Step 2: dataset ensure from config."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path


from src.data_gen import ensure_training_data
from src.utils import load_config


class TestStep2DataGen(unittest.TestCase):
    def test_ensure_creates_and_reuses(self):
        cfg = load_config()
        cfg = copy.deepcopy(cfg)
        cfg["data"]["train_samples"] = 4
        cfg["data"]["test_samples"] = 2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg["paths"]["data_root"] = str(root / "data")
            cfg["paths"]["checkpoint_dir"] = str(root / "trained_models")
            cfg["paths"]["figures_dir"] = str(root / "figures")

            out1 = ensure_training_data(cfg, project_root=root)
            self.assertTrue(out1["dataset"]["created"])
            tid = out1["dataset"]["dataset_id"]

            out2 = ensure_training_data(cfg, project_root=root)
            self.assertFalse(out2["dataset"]["created"])
            self.assertEqual(Path(out1["dataset"]["train_dir"]).name, "train")
            self.assertEqual(Path(out1["dataset"]["test_dir"]).name, "test")
            # New layout: targets live directly under the geometry group folder (no dataset_{hash} leaf).
            self.assertFalse(Path(out1["dataset"]["dataset_dir"]).name.startswith("dataset_"))


if __name__ == "__main__":
    unittest.main()
