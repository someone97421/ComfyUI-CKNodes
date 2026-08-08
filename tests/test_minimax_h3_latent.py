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


class FakeH3AudioVAE:
    audio_sample_rate = 32000
    downscale_ratio = 800

    def encode(self, waveform):
        batch, samples, channels = waveform.shape
        self.last_input_shape = tuple(waveform.shape)
        latent_t = (samples + 799) // 800
        return torch.zeros((batch, 32, channels, latent_t), dtype=waveform.dtype)


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

    def test_combine_can_force_audio_to_video_length(self):
        video = {"samples": torch.zeros((1, 24, 7, 4, 6))}
        short_audio = torch.ones((1, 32, 2, 10))
        short_mask = torch.zeros_like(short_audio)
        audio = {"samples": short_audio, "noise_mask": short_mask}

        output, audio_t = MODULE.CKMiniMaxH3CombineAVLatent.execute(
            video, audio, "audio_to_video", 24.0, "zeros"
        ).result
        _, output_audio = output["samples"].unbind()
        _, output_audio_mask = output["noise_mask"].unbind()
        self.assertEqual(audio_t, 37)
        self.assertEqual(tuple(output_audio.shape), (1, 32, 2, 37))
        self.assertTrue(torch.all(output_audio[..., :10] == 1))
        self.assertTrue(torch.all(output_audio[..., 10:] == 0))
        self.assertTrue(torch.all(output_audio_mask[..., :10] == 0))
        self.assertTrue(torch.all(output_audio_mask[..., 10:] == 1))

        long_audio = {"samples": torch.ones((1, 32, 2, 50))}
        cropped, cropped_t = MODULE.CKMiniMaxH3CombineAVLatent.execute(
            video, long_audio, "audio_to_video", 24.0, "repeat_last"
        ).result
        self.assertEqual(cropped_t, 37)
        self.assertEqual(cropped["samples"].unbind()[1].shape[-1], 37)

    def test_audio_vae_encode_normalizes_to_stereo(self):
        vae = FakeH3AudioVAE()
        audio = {"waveform": torch.zeros((2, 1, 3201)), "sample_rate": 32000}
        output, audio_t, duration, sample_rate = MODULE.CKMiniMaxH3AudioVAEEncode.execute(
            audio, vae, 1
        ).result
        self.assertEqual(vae.last_input_shape, (1, 4000, 2))
        self.assertEqual(tuple(output["samples"].shape), (1, 32, 2, 5))
        self.assertEqual((audio_t, sample_rate), (5, 32000))
        self.assertAlmostEqual(duration, 0.125)

    def test_empty_video_and_audio_latents(self):
        video, frames, video_t, width, height = MODULE.CKMiniMaxH3EmptyVideoLatent.execute(
            1000, 570, "video_frames", 125, 24.0, "up", "nearest"
        ).result
        self.assertEqual((frames, video_t, width, height), (141, 42, 992, 576))
        self.assertEqual(tuple(video["samples"].shape), (1, 24, 42, 36, 62))

        audio, audio_t, duration = MODULE.CKMiniMaxH3EmptyAudioLatent.execute(
            "video_latent_t", 37, 24.0, "exact"
        ).result
        self.assertEqual(audio_t, 207)
        self.assertEqual(tuple(audio["samples"].shape), (1, 32, 2, 207))
        self.assertAlmostEqual(duration, 207 / 40)

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

    def test_spatial_alignment(self):
        self.assertEqual(MODULE.align_spatial_pixels(1000, "nearest"), 992)
        self.assertEqual(MODULE.align_spatial_pixels(1000, "down"), 992)
        self.assertEqual(MODULE.align_spatial_pixels(1000, "up"), 1024)
        self.assertEqual(MODULE.align_spatial_pixels(1, "nearest"), 32)
        self.assertEqual(MODULE.align_spatial_pixels(1024, "exact"), 1024)
        with self.assertRaises(ValueError):
            MODULE.align_spatial_pixels(1000, "exact")

    def test_latent_resize_target_resolution(self):
        video = torch.zeros((1, 24, 7, 4, 6))
        latent = {"samples": video, "custom": "kept"}
        output, width, height, scale_x, scale_y = MODULE.CKMiniMaxH3LatentResize.execute(
            latent, "target_resolution", 1000, 570, 1.0, "nearest", "bicubic", "disabled"
        ).result

        self.assertEqual((width, height), (992, 576))
        self.assertEqual(tuple(output["samples"].shape), (1, 24, 7, 36, 62))
        self.assertEqual(output["custom"], "kept")
        self.assertAlmostEqual(scale_x, 992 / 96)
        self.assertAlmostEqual(scale_y, 576 / 64)

    def test_latent_resize_by_scale_preserves_audio_and_masks(self):
        video = torch.zeros((1, 24, 7, 4, 6))
        audio = torch.randn((1, 32, 2, 40))
        video_mask = torch.zeros((1, 1, 7, 4, 6))
        video_mask[..., 2:, 3:] = 1
        audio_mask = torch.rand_like(audio)
        latent = {
            "samples": MODULE.comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": MODULE.comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
        }

        output, width, height, scale_x, scale_y = MODULE.CKMiniMaxH3LatentResize.execute(
            latent, "scale_by", 1, 1, 1.5, "nearest", "nearest-exact", "disabled"
        ).result
        output_video, output_audio = output["samples"].unbind()
        output_video_mask, output_audio_mask = output["noise_mask"].unbind()

        self.assertEqual((width, height), (160, 96))
        self.assertEqual(tuple(output_video.shape), (1, 24, 7, 6, 10))
        self.assertEqual(tuple(output_video_mask.shape), (1, 1, 7, 6, 10))
        self.assertIs(output_audio, audio)
        self.assertIs(output_audio_mask, audio_mask)
        self.assertAlmostEqual(scale_x, 160 / 96)
        self.assertAlmostEqual(scale_y, 96 / 64)

    def test_latent_resize_rejects_audio_only(self):
        audio = {"samples": torch.zeros((1, 32, 2, 40))}
        with self.assertRaises(ValueError):
            MODULE.CKMiniMaxH3LatentResize.execute(
                audio, "scale_by", 1, 1, 2.0, "nearest", "bicubic", "disabled"
            )

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
