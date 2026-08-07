import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "minimax_h3_latent.py"
SPEC = importlib.util.spec_from_file_location("ck_minimax_h3_latent_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeH3VAE:
    def encode(self, image):
        batch, height, width, _ = image.shape
        return torch.zeros((batch, 24, 1, height // 16, width // 16), dtype=image.dtype)


class MiniMaxH3LatentHelpersTest(unittest.TestCase):
    def test_frame_latent_roundtrip(self):
        for frames, latent_t in ((5, 2), (22, 7), (124, 37), (362, 107)):
            self.assertEqual(MODULE.frames_to_video_latent_t(frames), latent_t)
            self.assertEqual(MODULE.video_latent_t_to_frames(latent_t), frames)

    def test_time_convert_default_shape(self):
        result = MODULE.convert_h3_time("video_frames", 124, 24.0, "exact")
        self.assertEqual(result["video_frames"], 124)
        self.assertEqual(result["video_latent_t"], 37)
        self.assertEqual(result["audio_latent_t"], 207)
        self.assertAlmostEqual(result["duration_seconds"], 124 / 24)
        self.assertTrue(result["source_exact"])

    def test_time_convert_alignment(self):
        self.assertEqual(MODULE.convert_h3_time("video_frames", 125, 24.0, "up")["video_frames"], 141)
        self.assertEqual(MODULE.convert_h3_time("video_frames", 125, 24.0, "down")["video_frames"], 124)
        self.assertEqual(MODULE.convert_h3_time("video_latent_t", 38, 24.0, "nearest")["video_latent_t"], 37)
        audio_result = MODULE.convert_h3_time("audio_latent_t", 207, 24.0, "exact")
        self.assertEqual(audio_result["video_frames"], 124)
        self.assertEqual(audio_result["video_latent_t"], 37)
        self.assertTrue(audio_result["source_exact"])
        with self.assertRaises(ValueError):
            MODULE.convert_h3_time("video_frames", 125, 24.0, "exact")

    def test_separate_and_combine_preserve_streams_and_masks(self):
        video = torch.randn((1, 24, 37, 4, 6))
        audio = torch.randn((1, 32, 2, 207))
        video_mask = torch.ones_like(video)
        audio_mask = torch.zeros_like(audio)
        av = {
            "samples": MODULE.comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": MODULE.comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "custom": "kept",
        }

        video_latent, audio_latent = MODULE.CKMiniMaxH3SeparateAVLatent.execute(av).result
        self.assertIs(video_latent["samples"], video)
        self.assertIs(audio_latent["samples"], audio)
        self.assertIs(video_latent["noise_mask"], video_mask)
        self.assertIs(audio_latent["noise_mask"], audio_mask)

        combined = MODULE.CKMiniMaxH3CombineAVLatent.execute(video_latent, audio_latent).result[0]
        streams = combined["samples"].unbind()
        masks = combined["noise_mask"].unbind()
        self.assertIs(streams[0], video)
        self.assertIs(streams[1], audio)
        self.assertIs(masks[0], video_mask)
        self.assertIs(masks[1], audio_mask)
        self.assertEqual(combined["custom"], "kept")

    def test_replace_video_slice_keeps_audio(self):
        video = torch.zeros((1, 24, 7, 4, 6))
        audio = torch.randn((1, 32, 2, 40))
        replacement = torch.ones((1, 24, 2, 4, 6))
        target = {"samples": MODULE.comfy.nested_tensor.NestedTensor((video, audio))}
        replacement_latent = {"samples": replacement}

        output, start, length, end = MODULE.CKMiniMaxH3ReplaceVideoLatentByIndex.execute(
            target, replacement_latent, 3, "error", "freeze_replaced"
        ).result
        output_video, output_audio = output["samples"].unbind()
        video_mask, audio_mask = output["noise_mask"].unbind()

        self.assertEqual((start, length, end), (3, 2, 5))
        self.assertTrue(torch.all(output_video[:, :, :3] == 0))
        self.assertTrue(torch.all(output_video[:, :, 3:5] == 1))
        self.assertTrue(torch.all(output_video[:, :, 5:] == 0))
        self.assertIs(output_audio, audio)
        self.assertTrue(torch.all(video_mask[:, :, 3:5] == 0))
        self.assertTrue(torch.all(audio_mask == 1))

    def test_replace_trim(self):
        target = {"samples": torch.zeros((1, 24, 7, 4, 6))}
        replacement = {"samples": torch.ones((1, 24, 3, 4, 6))}
        output, _, length, end = MODULE.CKMiniMaxH3ReplaceVideoLatentByIndex.execute(
            target, replacement, 6, "trim", "preserve_target"
        ).result
        self.assertEqual(length, 1)
        self.assertEqual(end, 7)
        self.assertTrue(torch.all(output["samples"][:, :, 6:] == 1))

    def test_replace_without_replacement_mask_does_not_create_mask(self):
        target = {"samples": torch.zeros((1, 24, 7, 4, 6))}
        replacement = {"samples": torch.ones((1, 24, 1, 4, 6))}
        output = MODULE.CKMiniMaxH3ReplaceVideoLatentByIndex.execute(
            target, replacement, 0, "error", "use_replacement_if_present"
        ).result[0]
        self.assertNotIn("noise_mask", output)

    def test_image_encode_selects_and_aligns_canvas(self):
        image = torch.zeros((2, 770, 1346, 3))
        output, width, height = MODULE.CKMiniMaxH3ImageVAEEncode.execute(
            image, FakeH3VAE(), 1, "center_crop_to_32"
        ).result
        self.assertEqual((width, height), (1344, 768))
        self.assertEqual(tuple(output["samples"].shape), (1, 24, 1, 48, 84))

    def test_latent_info(self):
        video = torch.zeros((1, 24, 37, 48, 84))
        audio = torch.zeros((1, 32, 2, 207))
        latent = {"samples": MODULE.comfy.nested_tensor.NestedTensor((video, audio))}
        result = MODULE.CKMiniMaxH3LatentInfo.execute(latent, 24.0).result
        self.assertTrue(result[1])
        self.assertTrue(result[2])
        self.assertEqual(result[3:9], (1, 1344, 768, 124, 37, 207))

    def test_image_latent_info_is_valid_condition_latent(self):
        latent = {"samples": torch.zeros((1, 24, 1, 48, 84))}
        result = MODULE.CKMiniMaxH3LatentInfo.execute(latent, 24.0).result
        self.assertTrue(result[2])
        self.assertEqual(result[6], 1)
        self.assertIn("图片 VAE 条件", result[0])


if __name__ == "__main__":
    unittest.main()
