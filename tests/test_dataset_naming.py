"""Dataset directory naming under data_root."""
from __future__ import annotations

import unittest

import numpy as np

from cleanup_ssps.dataset_registry import dataset_group_dirname


class TestDatasetNaming(unittest.TestCase):
    def test_group_contains_dim_ls_bounds(self):
        b = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        name = dataset_group_dirname(97, 0.2, b, bundle_type="hexagonal")
        self.assertIn("dim97", name)
        self.assertIn("ls", name)
        self.assertIn("bounds", name)
        self.assertTrue(name.startswith("hex_"))


if __name__ == "__main__":
    unittest.main()
