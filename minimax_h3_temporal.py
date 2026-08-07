import math

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io

from minimax_h3_latent import (
    AUDIO_LATENT_FPS,
    H3_CANVAS_MULTIPLE,
    H3_DIT_PATCH_SPATIAL,
    H3_SPATIAL_DOWNSCALE,
    align_frame_count,
    build_av_noise_mask,
    frames_to_audio_latent_t,
    is_nested_tensor,
    split_latent_streams,
    split_noise_masks,
    validate_audio_tensor,
    validate_video_tensor,
    video_latent_t_to_frames,
)


FRAME_SPAN_PATTERN = (1, 4, 4, 4, 4)
CANONICAL_VIDEO_OVERLAP_T = 2


def latent_boundary_to_frames(index):
    """Pixel-frame coordinate of a video-latent boundary on H3's native phase."""
    if index < 0:
        raise ValueError("latent index 不能小于 0")
    blocks, remainder = divmod(int(index), len(FRAME_SPAN_PATTERN))
    return blocks * sum(FRAME_SPAN_PATTERN) + sum(FRAME_SPAN_PATTERN[:remainder])


def latent_boundary_to_audio_t(index, frame_rate):
    if frame_rate <= 0:
        raise ValueError("frame_rate 必须大于 0")
    return int(round(latent_boundary_to_frames(index) / frame_rate * AUDIO_LATENT_FPS))


def normalized_video_mask(mask, video_shape, interpolation="h3_max"):
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask 必须是 torch.Tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        if mask.shape[1] == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
    elif mask.ndim != 5:
        raise ValueError(f"mask 必须是 [H,W]、[T,H,W] 或五维视频 mask，当前为 {tuple(mask.shape)}")

    mask = mask.float().clamp_(0.0, 1.0)
    target_t, target_h, target_w = video_shape[2:]
    if interpolation in ("h3_max", "h3_mean"):
        target_frames = latent_boundary_to_frames(target_t)
        if tuple(mask.shape[2:]) != (target_frames, target_h, target_w):
            mask = F.interpolate(
                mask, size=(target_frames, target_h, target_w), mode="trilinear", align_corners=False
            )
        pooled = []
        cursor = 0
        for index in range(target_t):
            span = FRAME_SPAN_PATTERN[index % len(FRAME_SPAN_PATTERN)]
            segment = mask[:, :, cursor:cursor + span]
            pooled.append(segment.amax(dim=2) if interpolation == "h3_max" else segment.mean(dim=2))
            cursor += span
        mask = torch.stack(pooled, dim=2)
    else:
        target_size = (target_t, target_h, target_w)
        mode = "nearest" if interpolation == "nearest" else "trilinear"
        if tuple(mask.shape[2:]) != target_size:
            mask = F.interpolate(mask, size=target_size, mode=mode, align_corners=False if mode == "trilinear" else None)
    if mask.shape[0] != video_shape[0]:
        mask = comfy.utils.repeat_to_batch_size(mask, video_shape[0])
    return mask[:, :1]


def full_video_mask(mask, video):
    if mask is None:
        return None
    return comfy.utils.reshape_mask(mask, video.shape)


def full_audio_mask(mask, audio):
    if mask is None:
        return None
    return comfy.utils.reshape_mask(mask, audio.shape)


def combine_mask(existing, generated, mode):
    if existing is None or mode == "replace":
        return generated
    existing = comfy.utils.reshape_mask(existing, generated.shape).to(generated)
    if mode == "multiply":
        return existing * generated
    if mode == "maximum":
        return torch.maximum(existing, generated)
    if mode == "minimum":
        return torch.minimum(existing, generated)
    raise ValueError(f"不支持的 mask combine mode：{mode}")


def slice_mask(mask, tensor, slices):
    if mask is None:
        return None
    mask = comfy.utils.reshape_mask(mask, tensor.shape)
    return mask[slices].clone()


def blend_overlap(first, second, overlap, blend_mode):
    if overlap <= 0:
        return torch.cat((first, second), dim=-1)
    if first.shape[-1] < overlap or second.shape[-1] < overlap:
        raise ValueError("输入长度小于拼接重叠长度")

    left = first[..., -overlap:]
    right = second[..., :overlap]
    if blend_mode == "keep_first":
        blended = left
    elif blend_mode == "keep_second":
        blended = right
    elif blend_mode == "linear":
        weight_right = torch.linspace(
            1.0 / (overlap + 1), overlap / (overlap + 1), overlap,
            device=left.device, dtype=left.dtype,
        )
        shape = [1] * left.ndim
        shape[-1] = overlap
        weight_right = weight_right.view(shape)
        blended = left * (1.0 - weight_right) + right.to(left) * weight_right
    else:
        raise ValueError(f"不支持的 blend_mode：{blend_mode}")
    return torch.cat((first[..., :-overlap], blended, second[..., overlap:].to(first)), dim=-1)


def blend_video_overlap(first, second, overlap, blend_mode):
    # Reuse the last-axis implementation by moving T to the end.
    first_t = first.movedim(2, -1)
    second_t = second.movedim(2, -1)
    return blend_overlap(first_t, second_t, overlap, blend_mode).movedim(-1, 2)


def prepare_video_and_mask(video, mask, canvas_mode, frame_align):
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError("video 必须是 [T,H,W,C] 的 IMAGE 帧批次")
    if video.shape[0] < 5:
        raise ValueError("MiniMax H3 视频 VAE 编码至少需要 5 帧")

    video = video[..., :3]
    source_t, source_h, source_w = video.shape[:3]

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask 必须是 MASK 张量")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
        mask_5d = mask.float().clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)
        if tuple(mask_5d.shape[2:]) != (source_t, source_h, source_w):
            mask_5d = F.interpolate(mask_5d, size=(source_t, source_h, source_w), mode="trilinear", align_corners=False)
        mask = mask_5d[0, 0]

    if canvas_mode == "center_crop_to_32":
        target_w = source_w // H3_CANVAS_MULTIPLE * H3_CANVAS_MULTIPLE
        target_h = source_h // H3_CANVAS_MULTIPLE * H3_CANVAS_MULTIPLE
        if target_w < H3_CANVAS_MULTIPLE or target_h < H3_CANVAS_MULTIPLE:
            raise ValueError("视频宽高必须至少为 32 像素")
        left = (source_w - target_w) // 2
        top = (source_h - target_h) // 2
        video = video[:, top:top + target_h, left:left + target_w]
        if mask is not None:
            mask = mask[:, top:top + target_h, left:left + target_w]
    elif canvas_mode == "resize_to_nearest_32":
        target_w = max(H3_CANVAS_MULTIPLE, round(source_w / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
        target_h = max(H3_CANVAS_MULTIPLE, round(source_h / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
        video = comfy.utils.common_upscale(video.movedim(-1, 1), target_w, target_h, "lanczos", "disabled").movedim(1, -1)
        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(1), size=(target_h, target_w), mode="bilinear", align_corners=False)[:, 0]
    elif canvas_mode != "keep":
        raise ValueError(f"不支持的 canvas_mode：{canvas_mode}")

    aligned_frames, exact = align_frame_count(video.shape[0], "exact" if frame_align == "exact" else frame_align)
    if frame_align == "exact" and not exact:
        raise ValueError("视频帧数不满足 17k+5")
    if aligned_frames < video.shape[0]:
        video = video[:aligned_frames]
        if mask is not None:
            mask = mask[:aligned_frames]
    elif aligned_frames > video.shape[0]:
        pad = aligned_frames - video.shape[0]
        video = torch.cat((video, video[-1:].repeat(pad, 1, 1, 1)), dim=0)
        if mask is not None:
            mask = torch.cat((mask, mask[-1:].repeat(pad, 1, 1)), dim=0)
    return video, mask, aligned_frames


class CKMiniMaxH3VideoVAEEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3VideoVAEEncode",
            display_name="CK MiniMax H3 Video VAE Encode",
            description="将 IMAGE 批次视为视频帧，按 17k+5 对齐后使用 MiniMax H3 视频 VAE 编码。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Image.Input("video"),
                io.Vae.Input("vae"),
                io.Combo.Input("frame_align", options=["down", "up", "exact"], default="down"),
                io.Combo.Input(
                    "canvas_mode",
                    options=["center_crop_to_32", "resize_to_nearest_32", "keep"],
                    default="center_crop_to_32",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Int.Output(display_name="video_frames"),
                io.Int.Output(display_name="video_latent_t"),
                io.Int.Output(display_name="encoded_width"),
                io.Int.Output(display_name="encoded_height"),
            ],
        )

    @classmethod
    def execute(cls, video, vae, frame_align, canvas_mode):
        video, _, frame_count = prepare_video_and_mask(video, None, canvas_mode, frame_align)
        encoded = vae.encode(video)
        validate_video_tensor(encoded, "H3 视频 VAE 编码结果")
        if encoded.shape[0] != 1:
            raise ValueError("MiniMax H3 视频编码结果必须是 batch 1")
        expected_frames = video_latent_t_to_frames(encoded.shape[2])
        if expected_frames != frame_count:
            raise ValueError(
                f"视频 VAE 输出 T={encoded.shape[2]} 对应 {expected_frames} 帧，但输入对齐后为 {frame_count} 帧"
            )
        if encoded.shape[3] % H3_DIT_PATCH_SPATIAL or encoded.shape[4] % H3_DIT_PATCH_SPATIAL:
            raise ValueError("编码后的 latent 高宽不能被 H3 DiT 的 2x2 patch 整除")
        output = {
            "samples": encoded,
            "ck_minimax_h3_kind": "video",
            "ck_minimax_h3_frame_count": frame_count,
        }
        return io.NodeOutput(
            output, frame_count, encoded.shape[2],
            encoded.shape[4] * H3_SPATIAL_DOWNSCALE,
            encoded.shape[3] * H3_SPATIAL_DOWNSCALE,
        )


class CKMiniMaxH3TrimLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3TrimLatent",
            display_name="CK MiniMax H3 Trim Latent",
            description="按 video latent T 索引截取视频，并按照 H3 时间坐标同步截取联合音频。",
            category="CK Nodes/MiniMax H3/Temporal",
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input("start_index", default=0, min=0, max=0x7FFFFFFF, step=1),
                io.Int.Input("latent_length", default=7, min=1, max=0x7FFFFFFF, step=1),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input("trim_mode", options=["strict_h3", "raw"], default="strict_h3"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="video_frames"),
                io.Int.Output(display_name="video_latent_t"),
                io.Int.Output(display_name="audio_latent_t"),
            ],
        )

    @classmethod
    def execute(cls, latent, start_index, latent_length, frame_rate, trim_mode):
        video, audio, is_av = split_latent_streams(latent)
        if video is None:
            raise ValueError("latent 中没有视频流")
        validate_video_tensor(video)
        end = start_index + latent_length
        if end > video.shape[2]:
            raise ValueError(f"截取区间 [{start_index}, {end}) 超出 video T={video.shape[2]}")
        if trim_mode == "strict_h3":
            if start_index % 5 != 0:
                raise ValueError("strict_h3 模式要求 start_index 是 5 的倍数，以保持 H3 时间相位")
            video_latent_t_to_frames(latent_length)

        output_video = video[:, :, start_index:end].clone()
        video_frames = latent_boundary_to_frames(latent_length)
        video_mask, audio_mask = split_noise_masks(latent, is_av)
        output_video_mask = slice_mask(video_mask, video, (slice(None), slice(None), slice(start_index, end), slice(None), slice(None)))

        output = latent.copy()
        output.pop("noise_mask", None)
        output_audio_t = 0
        if is_av:
            validate_audio_tensor(audio)
            audio_start = latent_boundary_to_audio_t(start_index, frame_rate)
            output_audio_t = frames_to_audio_latent_t(video_frames, frame_rate)
            audio_end = audio_start + output_audio_t
            if audio_end > audio.shape[-1]:
                raise ValueError(
                    f"同步音频区间 [{audio_start}, {audio_end}) 超出 audio T={audio.shape[-1]}，"
                    "请检查 FPS 或输入 AV latent 时长"
                )
            output_audio = audio[..., audio_start:audio_end].clone()
            output_audio_mask = slice_mask(audio_mask, audio, (slice(None), slice(None), slice(None), slice(audio_start, audio_end)))
            output["samples"] = comfy.nested_tensor.NestedTensor((output_video, output_audio))
            combined_mask = build_av_noise_mask(output_video, output_audio, output_video_mask, output_audio_mask)
            if combined_mask is not None:
                output["noise_mask"] = combined_mask
        else:
            output["samples"] = output_video
            if output_video_mask is not None:
                output["noise_mask"] = output_video_mask
        output["ck_minimax_h3_frame_rate"] = float(frame_rate)
        output["ck_minimax_h3_frame_count"] = video_frames
        return io.NodeOutput(output, video_frames, output_video.shape[2], output_audio_t)


class CKMiniMaxH3ConcatLatents(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3ConcatLatents",
            display_name="CK MiniMax H3 Concat Latents",
            description="拼接两个 H3 视频或联合 AV latent；标准模式重叠 2 个视频 token 和对应 5 帧音频时间。",
            category="CK Nodes/MiniMax H3/Temporal",
            inputs=[
                io.Latent.Input("first_latent"),
                io.Latent.Input("second_latent"),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input("concat_mode", options=["h3_overlap", "raw"], default="h3_overlap"),
                io.Combo.Input("blend_mode", options=["linear", "keep_first", "keep_second"], default="linear"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="video_frames"),
                io.Int.Output(display_name="video_latent_t"),
                io.Int.Output(display_name="audio_latent_t"),
            ],
        )

    @classmethod
    def execute(cls, first_latent, second_latent, frame_rate, concat_mode, blend_mode):
        first_video, first_audio, first_av = split_latent_streams(first_latent)
        second_video, second_audio, second_av = split_latent_streams(second_latent)
        if first_video is None or second_video is None:
            raise ValueError("两个输入都必须包含视频 latent")
        validate_video_tensor(first_video, "first_latent 视频流")
        validate_video_tensor(second_video, "second_latent 视频流")
        if first_video.shape[:2] + first_video.shape[3:] != second_video.shape[:2] + second_video.shape[3:]:
            raise ValueError("两个视频 latent 的 batch、通道和空间尺寸必须一致")
        if first_av != second_av:
            raise ValueError("两个输入必须同为联合 AV latent，或同为纯视频 latent")

        first_video_mask, first_audio_mask = split_noise_masks(first_latent, first_av)
        second_video_mask, second_audio_mask = split_noise_masks(second_latent, second_av)

        overlap_video = CANONICAL_VIDEO_OVERLAP_T if concat_mode == "h3_overlap" else 0
        if concat_mode == "h3_overlap":
            video_latent_t_to_frames(first_video.shape[2])
            video_latent_t_to_frames(second_video.shape[2])
        output_video = blend_video_overlap(first_video, second_video, overlap_video, blend_mode)

        fv_mask = full_video_mask(first_video_mask, first_video)
        sv_mask = full_video_mask(second_video_mask, second_video)
        output_video_mask = None
        if fv_mask is not None or sv_mask is not None:
            fv_mask = torch.ones_like(first_video) if fv_mask is None else fv_mask
            sv_mask = torch.ones_like(second_video) if sv_mask is None else sv_mask
            output_video_mask = blend_video_overlap(fv_mask, sv_mask, overlap_video, "keep_second" if blend_mode == "keep_second" else "keep_first")
            if blend_mode == "linear" and overlap_video:
                start = first_video.shape[2] - overlap_video
                output_video_mask[:, :, start:start + overlap_video] = torch.maximum(
                    fv_mask[:, :, -overlap_video:], sv_mask[:, :, :overlap_video].to(fv_mask)
                )

        output = first_latent.copy()
        for key, value in second_latent.items():
            if key not in ("samples", "noise_mask") and key not in output:
                output[key] = value
        output.pop("noise_mask", None)
        output_audio_t = 0
        if first_av:
            validate_audio_tensor(first_audio, "first_latent 音频流")
            validate_audio_tensor(second_audio, "second_latent 音频流")
            if first_audio.shape[:3] != second_audio.shape[:3]:
                raise ValueError("两个音频 latent 的 batch、通道和声道必须一致")
            if concat_mode == "h3_overlap":
                first_frames = video_latent_t_to_frames(first_video.shape[2])
                second_frames = video_latent_t_to_frames(second_video.shape[2])
                expected_audio_t = frames_to_audio_latent_t(first_frames + second_frames - 5, frame_rate)
                overlap_audio = first_audio.shape[-1] + second_audio.shape[-1] - expected_audio_t
                if overlap_audio < 0:
                    raise ValueError(
                        "两个输入的音频 latent 总长度小于 H3 拼接后所需长度；请检查 FPS 和 AV 时长"
                    )
            else:
                overlap_audio = 0
            output_audio = blend_overlap(first_audio, second_audio, overlap_audio, blend_mode)
            output_audio_t = output_audio.shape[-1]

            fa_mask = full_audio_mask(first_audio_mask, first_audio)
            sa_mask = full_audio_mask(second_audio_mask, second_audio)
            output_audio_mask = None
            if fa_mask is not None or sa_mask is not None:
                fa_mask = torch.ones_like(first_audio) if fa_mask is None else fa_mask
                sa_mask = torch.ones_like(second_audio) if sa_mask is None else sa_mask
                output_audio_mask = blend_overlap(fa_mask, sa_mask, overlap_audio, "keep_first")
                if overlap_audio:
                    start = first_audio.shape[-1] - overlap_audio
                    output_audio_mask[..., start:start + overlap_audio] = torch.maximum(
                        fa_mask[..., -overlap_audio:], sa_mask[..., :overlap_audio].to(fa_mask)
                    )

            output["samples"] = comfy.nested_tensor.NestedTensor((output_video, output_audio))
            combined_mask = build_av_noise_mask(output_video, output_audio, output_video_mask, output_audio_mask)
            if combined_mask is not None:
                output["noise_mask"] = combined_mask
        else:
            output["samples"] = output_video
            if output_video_mask is not None:
                output["noise_mask"] = output_video_mask

        try:
            video_frames = video_latent_t_to_frames(output_video.shape[2])
        except ValueError:
            video_frames = latent_boundary_to_frames(output_video.shape[2])
        output["ck_minimax_h3_frame_rate"] = float(frame_rate)
        output["ck_minimax_h3_frame_count"] = video_frames
        return io.NodeOutput(output, video_frames, output_video.shape[2], output_audio_t)


class CKMiniMaxH3TemporalMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3TemporalMask",
            display_name="CK MiniMax H3 Temporal Mask",
            description="按 video latent T 区间创建时间 noise mask，可同步映射到联合音频时间轴。",
            category="CK Nodes/MiniMax H3/Mask",
            inputs=[
                io.Latent.Input("latent"),
                io.Int.Input("start_index", default=0, min=0, max=0x7FFFFFFF, step=1),
                io.Int.Input("end_index", default=0, min=0, max=0x7FFFFFFF, step=1,
                             tooltip="0 表示直到视频 latent 末尾；其他值为不包含的结束索引。"),
                io.Float.Input("inside_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("outside_strength", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Int.Input("feather", default=0, min=0, max=4096, step=1),
                io.Boolean.Input("affect_audio", default=True),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input("combine_mode", options=["replace", "multiply", "maximum", "minimum"], default="replace"),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @staticmethod
    def interval_mask(length, start, end, inside, outside, feather, device, dtype):
        values = torch.full((length,), float(outside), device=device, dtype=dtype)
        values[start:end] = float(inside)
        feather = min(feather, max(0, (end - start) // 2))
        if feather > 0:
            ramp_in = torch.linspace(outside, inside, feather + 2, device=device, dtype=dtype)[1:-1]
            ramp_out = torch.linspace(inside, outside, feather + 2, device=device, dtype=dtype)[1:-1]
            values[start:start + feather] = ramp_in
            values[end - feather:end] = ramp_out
        return values

    @classmethod
    def execute(cls, latent, start_index, end_index, inside_strength, outside_strength,
                feather, affect_audio, frame_rate, combine_mode):
        video, audio, is_av = split_latent_streams(latent)
        if video is None:
            raise ValueError("latent 中没有视频流")
        validate_video_tensor(video)
        video_t = video.shape[2]
        end_index = video_t if end_index == 0 else end_index
        if not 0 <= start_index < end_index <= video_t:
            raise ValueError(f"时间遮罩区间必须满足 0 <= start < end <= {video_t}")

        vector = cls.interval_mask(
            video_t, start_index, end_index, inside_strength, outside_strength,
            feather, video.device, video.dtype,
        )
        generated_video_mask = vector.view(1, 1, video_t, 1, 1).repeat(video.shape[0], 1, 1, 1, 1)
        existing_video_mask, existing_audio_mask = split_noise_masks(latent, is_av)
        output_video_mask = combine_mask(existing_video_mask, generated_video_mask, combine_mode)

        output = latent.copy()
        output.pop("noise_mask", None)
        if is_av:
            validate_audio_tensor(audio)
            output_audio_mask = existing_audio_mask
            if affect_audio:
                audio_start = latent_boundary_to_audio_t(start_index, frame_rate)
                audio_end = min(audio.shape[-1], latent_boundary_to_audio_t(end_index, frame_rate))
                audio_feather_start = latent_boundary_to_audio_t(min(video_t, start_index + feather), frame_rate) - audio_start
                audio_feather_end = audio_end - latent_boundary_to_audio_t(max(0, end_index - feather), frame_rate)
                audio_feather = max(0, min(audio_feather_start, audio_feather_end))
                audio_vector = cls.interval_mask(
                    audio.shape[-1], audio_start, audio_end, inside_strength, outside_strength,
                    audio_feather, audio.device, audio.dtype,
                )
                generated_audio_mask = audio_vector.view(1, 1, 1, -1).repeat(audio.shape[0], 1, audio.shape[2], 1)
                output_audio_mask = combine_mask(existing_audio_mask, generated_audio_mask, combine_mode)
            output["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
            combined_mask = build_av_noise_mask(video, audio, output_video_mask, output_audio_mask)
            if combined_mask is not None:
                output["noise_mask"] = combined_mask
        else:
            output["samples"] = video
            output["noise_mask"] = output_video_mask
        return io.NodeOutput(output)


class CKMiniMaxH3ApplyVideoMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3ApplyVideoMask",
            display_name="CK MiniMax H3 Apply Video Mask",
            description="将像素视频 MASK 缩放到 H3 视频 latent 的时空尺寸，并写入视频 noise mask。",
            category="CK Nodes/MiniMax H3/Mask",
            inputs=[
                io.Latent.Input("latent"),
                io.Mask.Input("mask"),
                io.Boolean.Input("invert_mask", default=False),
                io.Combo.Input(
                    "interpolation",
                    options=["h3_max", "h3_mean", "trilinear", "nearest"],
                    default="h3_max",
                ),
                io.Combo.Input("combine_mode", options=["replace", "multiply", "maximum", "minimum"], default="replace"),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, mask, invert_mask, interpolation, combine_mode):
        video, audio, is_av = split_latent_streams(latent)
        if video is None:
            raise ValueError("latent 中没有视频流")
        validate_video_tensor(video)
        generated = normalized_video_mask(mask, video.shape, interpolation).to(video)
        if invert_mask:
            generated = 1.0 - generated

        video_mask, audio_mask = split_noise_masks(latent, is_av)
        video_mask = combine_mask(video_mask, generated, combine_mode)
        output = latent.copy()
        output.pop("noise_mask", None)
        if is_av:
            validate_audio_tensor(audio)
            output["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
            output["noise_mask"] = build_av_noise_mask(video, audio, video_mask, audio_mask)
        else:
            output["samples"] = video
            output["noise_mask"] = video_mask
        return io.NodeOutput(output)


class CKMiniMaxH3VideoVAEEncodeMaskedNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3VideoVAEEncodeMaskedNoise",
            display_name="CK MiniMax H3 Video VAE Encode Masked Noise",
            description="编码视频，并在 MASK 白色区域混合标准 latent 噪声，同时生成视频 noise mask。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Image.Input("video"),
                io.Mask.Input("mask"),
                io.Vae.Input("vae"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1),
                io.Float.Input("noise_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("denoise_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("invert_mask", default=False),
                io.Combo.Input("frame_align", options=["down", "up", "exact"], default="down"),
                io.Combo.Input(
                    "canvas_mode",
                    options=["center_crop_to_32", "resize_to_nearest_32", "keep"],
                    default="center_crop_to_32",
                ),
                io.Combo.Input(
                    "mask_interpolation",
                    options=["h3_max", "h3_mean", "trilinear", "nearest"],
                    default="h3_max",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="noisy_latent"),
                io.Latent.Output(display_name="clean_latent"),
                io.Int.Output(display_name="video_frames"),
                io.Int.Output(display_name="video_latent_t"),
            ],
        )

    @classmethod
    def execute(cls, video, mask, vae, seed, noise_strength, denoise_strength,
                invert_mask, frame_align, canvas_mode, mask_interpolation):
        video, pixel_mask, frame_count = prepare_video_and_mask(video, mask, canvas_mode, frame_align)
        encoded = vae.encode(video)
        validate_video_tensor(encoded, "H3 视频 VAE 编码结果")
        if encoded.shape[0] != 1:
            raise ValueError("MiniMax H3 视频编码结果必须是 batch 1")
        if encoded.shape[3] % H3_DIT_PATCH_SPATIAL or encoded.shape[4] % H3_DIT_PATCH_SPATIAL:
            raise ValueError("编码后的 latent 高宽不能被 H3 DiT 的 2x2 patch 整除")

        latent_mask = normalized_video_mask(pixel_mask, encoded.shape, mask_interpolation).to(encoded)
        if invert_mask:
            latent_mask = 1.0 - latent_mask

        generator = torch.Generator("cpu").manual_seed(int(seed))
        noise = torch.randn(encoded.shape, generator=generator, dtype=torch.float32, device="cpu").to(encoded)
        mix = latent_mask * float(noise_strength)
        noisy = encoded * (1.0 - mix) + noise * mix

        clean_output = {
            "samples": encoded,
            "ck_minimax_h3_kind": "video",
            "ck_minimax_h3_frame_count": frame_count,
        }
        noisy_output = clean_output.copy()
        noisy_output["samples"] = noisy
        noisy_output["noise_mask"] = latent_mask * float(denoise_strength)
        noisy_output["ck_minimax_h3_masked_noise_seed"] = int(seed)
        return io.NodeOutput(noisy_output, clean_output, frame_count, encoded.shape[2])


NODE_CLASS_MAPPINGS = {
    "CKMiniMaxH3VideoVAEEncode": CKMiniMaxH3VideoVAEEncode,
    "CKMiniMaxH3TrimLatent": CKMiniMaxH3TrimLatent,
    "CKMiniMaxH3ConcatLatents": CKMiniMaxH3ConcatLatents,
    "CKMiniMaxH3TemporalMask": CKMiniMaxH3TemporalMask,
    "CKMiniMaxH3ApplyVideoMask": CKMiniMaxH3ApplyVideoMask,
    "CKMiniMaxH3VideoVAEEncodeMaskedNoise": CKMiniMaxH3VideoVAEEncodeMaskedNoise,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CKMiniMaxH3VideoVAEEncode": "CK MiniMax H3 Video VAE Encode",
    "CKMiniMaxH3TrimLatent": "CK MiniMax H3 Trim Latent",
    "CKMiniMaxH3ConcatLatents": "CK MiniMax H3 Concat Latents",
    "CKMiniMaxH3TemporalMask": "CK MiniMax H3 Temporal Mask",
    "CKMiniMaxH3ApplyVideoMask": "CK MiniMax H3 Apply Video Mask",
    "CKMiniMaxH3VideoVAEEncodeMaskedNoise": "CK MiniMax H3 Video VAE Encode Masked Noise",
}
