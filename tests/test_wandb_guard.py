"""wandb_utils: safe logging when no run is active."""
from __future__ import annotations

import unittest
from unittest import mock

import utils.wandb_utils as wu


class TestWandbGuard(unittest.TestCase):
    def test_log_metrics_skips_without_run(self):
        with mock.patch.object(wu.wandb, "run", None):
            with mock.patch.object(wu.wandb, "log") as mock_log:
                wu.log_metrics({"a": 1})
                mock_log.assert_not_called()

    def test_log_metrics_forwards_with_run(self):
        fake_run = object()
        with mock.patch.object(wu.wandb, "run", fake_run):
            with mock.patch.object(wu.wandb, "log") as mock_log:
                wu.log_metrics({"b": 2})
                mock_log.assert_called_once_with({"b": 2})


if __name__ == "__main__":
    unittest.main()
