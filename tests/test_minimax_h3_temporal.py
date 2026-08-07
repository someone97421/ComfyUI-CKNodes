import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "minimax_h3_temporal.py"
SPEC = importlib.util.spec_from_file_location("ck_minimax_h3_temporal_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeH3VideoVAE:
    def encode(self, video):
        frames, height, width, _ = video.shape
        latent_t = ((frames - 5) // 17) * 5 + 2
        return torch.zeros((1, 24, latent_t, height // 16, width // 16), dtype=video.dtype)


class MiniMaxH3TemporalTest(unittest.TestCase):
    def test_latent_boundary_frame_mapping(self):
        expected = [0, 1, 5, 9, 13, 17, 18, 22]
        self.assertEqual([MODULE.latent_boundary_to_frames(i) for i in range(8)], expected)

    def test_video_batch_vae_encode(self):
        video = torch.zeros((24, 770, 1346, 3))
        output, frames, latent_t, width, height = MODULE.CKMiniMaxH3VideoVAEEncode.execute(
            video, FakeH3VideoVAE(), "down", "center_crop_to_32"
        ).result
        self.assertEqual((frames, latent_t, width, height), (22, 7, 1344, 768))
        self.assertEqual(tuple(output["samples"].shape), (1, 24, 7, 48, 84))

    def test_strict_trim_synchronizes_audio(self):
        video = torch.arange(37, dtype=torch.float32).view(1, 1, 37, 1, 1).repeat(1, 24, 1, 2, 2)
        audio = torch.arange(207, dtype=torch.float32).view(1, 1, 1, 207).repeat(1, 32, 2, 1)
        latent = {"samples": MODULE.comfy.nested_tensor.NestedTensor((video, audio))}
        output, frames, video_t, audio_t = MODULE.CKMiniMaxH3TrimLatent.execute(
            latent, 5, 7, 24.0, "strict_h3"
        ).result
        output_video, output_audio = output["samples"].unbind()
        self.assertEqual((frames, video_t, audio_t), (22, 7, 37))
        self.assertTrue(torch.equal(output_video[:, :, 0], video[:, :, 5]))
        self.assertTrue(torch.equal(output_audio[..., 0], audio[..., 28]))

    def test_strict_trim_rejects_bad_phase(self):
        latent = {"samples": torch.zeros((1, 24, 37, 2, 2))}
        with self.assertRaises(ValueError):
            MODULE.CKMiniMaxH3TrimLatent.execute(latent, 1, 7, 24.0, "strict_h3")

    def test_h3_concat_keeps_legal_video_and_audio_length(self):
        first_video = torch.zeros((1, 24, 7, 2, 2))
        second_video = torch.ones((1, 24, 7, 2, 2))
        first_audio = torch.zeros((1, 32, 2, 37))
        second_audio = torch.ones((1, 32, 2, 37))
        first = {"samples": MODULE.comfy.nested_tensor.NestedTensor((first_video, first_audio))}
        second = {"samples": MODULE.comfy.nested_tensor.NestedTensor((second_video, second_audio))}
        output, frames, video_t, audio_t = MODULE.CKMiniMaxH3ConcatLatents.execute(
            first, second, 24.0, "h3_overlap", "linear"
        ).result
        video, audio = output["samples"].unbind()
        self.assertEqual((frames, video_t, audio_t), (39, 12, 65))
        self.assertEqual(video.shape[2], 12)
        self.assertEqual(audio.shape[-1], 65)

    def test_h3_span_mask_pooling(self):
        mask = torch.zeros((22, 32, 32))
        mask[1:5] = 1.0
        pooled = MODULE.normalized_video_mask(mask, (1, 24, 7, 2, 2), "h3_max")
        self.assertEqual(tuple(pooled.shape), (1, 1, 7, 2, 2))
        self.assertTrue(torch.all(pooled[:, :, 0] == 0))
        self.assertTrue(torch.all(pooled[:, :, 1] == 1))
        self.assertTrue(torch.all(pooled[:, :, 2:] == 0))

    def test_temporal_mask_applies_to_video_and_audio(self):
        video = torch.zeros((1, 24, 7, 2, 2))
        audio = torch.zeros((1, 32, 2, 37))
        latent = {"samples": MODULE.comfy.nested_tensor.NestedTensor((video, audio))}
        output = MODULE.CKMiniMaxH3TemporalMask.execute(
            latent, 1, 3, 1.0, 0.0, 0, True, 24.0, "replace"
        ).result[0]
        video_mask, audio_mask = output["noise_mask"].unbind()
        self.assertEqual(tuple(video_mask.shape), (1, 1, 7, 1, 1))
        self.assertTrue(torch.all(video_mask[:, :, 1:3] == 1))
        self.assertTrue(torch.all(video_mask[:, :, :1] == 0))
        self.assertGreater(float(audio_mask.sum()), 0.0)

    def test_apply_video_mask_preserves_audio_stream(self):
        video = torch.zeros((1, 24, 7, 2, 2))
        audio = torch.zeros((1, 32, 2, 37))
        mask = torch.ones((22, 32, 32))
        latent = {"samples": MODULE.comfy.nested_tensor.NestedTensor((video, audio))}
        output = MODULE.CKMiniMaxH3ApplyVideoMask.execute(
            latent, mask, False, "h3_max", "replace"
        ).result[0]
        output_video, output_audio = output["samples"].unbind()
        video_mask, _ = output["noise_mask"].unbind()
        self.assertIs(output_video, video)
        self.assertIs(output_audio, audio)
        self.assertTrue(torch.all(video_mask == 1))

    def test_masked_noise_encode_only_changes_white_region(self):
        video = torch.zeros((22, 32, 32, 3))
        mask = torch.zeros((22, 32, 32))
        mask[1:5] = 1.0
        noisy, clean, frames, latent_t = MODULE.CKMiniMaxH3VideoVAEEncodeMaskedNoise.execute(
            video, mask, FakeH3VideoVAE(), 123, 1.0, 1.0,
            False, "exact", "keep", "h3_max"
        ).result
        self.assertEqual((frames, latent_t), (22, 7))
        self.assertTrue(torch.all(clean["samples"] == 0))
        self.assertTrue(torch.all(noisy["samples"][:, :, 0] == 0))
        self.assertGreater(float(noisy["samples"][:, :, 1].abs().sum()), 0.0)
        self.assertTrue(torch.all(noisy["noise_mask"][:, :, 1] == 1))


if __name__ == "__main__":
    unittest.main()
