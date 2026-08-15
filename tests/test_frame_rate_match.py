import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "FrameRateMatch.py"
SPEC = importlib.util.spec_from_file_location("ck_frame_rate_match_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrameRateMatchTest(unittest.TestCase):
    def test_same_fps_keeps_every_frame(self):
        self.assertEqual(MODULE.calculate_frame_indices(5, 24.0, 24.0), [0, 1, 2, 3, 4])

    def test_downsampling_uses_nearest_timestamps(self):
        indices = MODULE.calculate_frame_indices(30, 30.0, 24.0)
        self.assertEqual(len(indices), 24)
        self.assertEqual(indices[:8], [0, 1, 3, 4, 5, 6, 8, 9])
        self.assertEqual(indices[-1], 29)
        self.assertEqual(len(indices), len(set(indices)))

    def test_fractional_broadcast_rates_do_not_drift(self):
        indices = MODULE.calculate_frame_indices(300, 29.97, 23.976)
        self.assertEqual(len(indices), 240)
        self.assertTrue(all(a < b for a, b in zip(indices, indices[1:])))
        self.assertEqual(indices[-1], 299)

    def test_upsampling_duplicates_frames_without_interpolation(self):
        indices = MODULE.calculate_frame_indices(4, 24.0, 48.0)
        self.assertEqual(indices, [0, 1, 1, 2, 2, 3, 3, 3])
        self.assertEqual(MODULE._compact_indices(indices), "0, 1×2, 2×2, 3×3")

    def test_empty_batch_stays_empty(self):
        images = torch.empty((0, 2, 2, 3))
        output, info = MODULE.MatchBatchFrameRate().match_frame_rate(images, 30.0, 24.0)
        self.assertEqual(tuple(output.shape), (0, 2, 2, 3))
        self.assertIn("输出: 0 帧", info)

    def test_node_selects_expected_batch_items(self):
        images = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1)
        output, info = MODULE.MatchBatchFrameRate().match_frame_rate(images, 5.0, 2.0)
        self.assertEqual(output.flatten().tolist(), [0.0, 3.0])
        self.assertIn("丢弃: 3 帧", info)

    def test_invalid_fps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "FPS"):
            MODULE.calculate_frame_indices(5, 0, 24)


if __name__ == "__main__":
    unittest.main()
