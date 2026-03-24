"""Step 3: trainer config builder."""
from __future__ import annotations

import unittest

from src.train import build_trainer_config
from src.utils import load_config


class TestStep3Train(unittest.TestCase):
    def test_build_trainer_config_maps_paths(self):
        cfg = load_config()
        tc = build_trainer_config(cfg, "/tmp/train", "/tmp/test")
        self.assertEqual(tc["data_dir"], "/tmp/train")
        self.assertEqual(tc["test_dir"], "/tmp/test")
        self.assertIn("train_feedforward", tc)
        self.assertEqual(tc["sampling_modes"], ["geo_det"])


if __name__ == "__main__":
    unittest.main()
