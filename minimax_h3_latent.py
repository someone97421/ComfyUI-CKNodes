import math

import torch
import torchaudio

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
from comfy_api.latest import io


VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STEREO_CHANNELS = 2
AUDIO_LATENT_FPS = 40.0
VIDEO_FRAME_BLOCK = 17
VIDEO_FRAME_OFFSET = 5
VIDEO_LATENT_BLOCK = 5
VIDEO_LATENT_OFFSET = 2
H3_SPATIAL_DOWNSCALE = 16
H3_DIT_PATCH_SPATIAL = 2
H3_CANVAS_MULTIPLE = H3_SPATIAL_DOWNSCALE * H3_DIT_PATCH_SPATIAL


def is_nested_tensor(value):
    return getattr(value, "is_nested", False)


def align_to_sequence(value, block, offset, mode, minimum):
    """Align a number to offset + block*k and report whether it was exact."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("换算值必须是有限数字")

    value = max(float(minimum), value)
    position = (value - offset) / block
    exact = abs(position - round(position)) < 1e-6

    if mode == "exact":
        if not exact:
            raise ValueError(f"{value:g} 不满足 {block}k+{offset} 的 MiniMax H3 合法序列")
        index = round(position)
    elif mode == "down":
        index = math.floor(position + 1e-9)
    elif mode == "nearest":
        index = math.floor(position + 0.5)
    else:  # up
        index = math.ceil(position - 1e-9)

    index = max(0, index)
    return int(offset + block * index), exact


def align_frame_count(frame_count, mode="up"):
    return align_to_sequence(
        frame_count,
        VIDEO_FRAME_BLOCK,
        VIDEO_FRAME_OFFSET,
        mode,
        VIDEO_FRAME_OFFSET,
    )


def align_video_latent_t(latent_t, mode="up"):
    return align_to_sequence(
        latent_t,
        VIDEO_LATENT_BLOCK,
        VIDEO_LATENT_OFFSET,
        mode,
        VIDEO_LATENT_OFFSET,
    )


def align_spatial_pixels(value, mode="nearest"):
    """将像素宽高对齐到 H3 VAE + DiT 共同要求的 32 倍数。"""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("目标宽高必须是大于 0 的有限数字")

    units = value / H3_CANVAS_MULTIPLE
    if mode == "exact":
        if abs(units - round(units)) >= 1e-6:
            raise ValueError(f"{value:g} 不是 {H3_CANVAS_MULTIPLE} 的整数倍")
        aligned_units = round(units)
    elif mode == "down":
        aligned_units = math.floor(units + 1e-9)
    elif mode == "up":
        aligned_units = math.ceil(units - 1e-9)
    elif mode == "nearest":
        aligned_units = math.floor(units + 0.5)
    else:
        raise ValueError(f"不支持的空间对齐方式：{mode}")

    return max(H3_CANVAS_MULTIPLE, int(aligned_units) * H3_CANVAS_MULTIPLE)


def calculate_h3_resize(width, height, resize_mode, target_width, target_height, scale_by, align_mode):
    if width < H3_CANVAS_MULTIPLE or height < H3_CANVAS_MULTIPLE:
        raise ValueError("输入 latent 对应的像素宽高必须至少为 32")

    if resize_mode == "target_resolution":
        raw_width = target_width
        raw_height = target_height
    elif resize_mode == "scale_by":
        if not math.isfinite(scale_by) or scale_by <= 0:
            raise ValueError("scale_by 必须是大于 0 的有限数字")
        raw_width = width * scale_by
        raw_height = height * scale_by
    else:
        raise ValueError(f"不支持的 resize_mode：{resize_mode}")

    output_width = align_spatial_pixels(raw_width, align_mode)
    output_height = align_spatial_pixels(raw_height, align_mode)
    return output_width, output_height


def frames_to_video_latent_t(frame_count):
    frame_count, _ = align_frame_count(frame_count, "exact")
    return ((frame_count - VIDEO_FRAME_OFFSET) // VIDEO_FRAME_BLOCK) * VIDEO_LATENT_BLOCK + VIDEO_LATENT_OFFSET


def video_latent_t_to_frames(latent_t):
    latent_t, _ = align_video_latent_t(latent_t, "exact")
    return ((latent_t - VIDEO_LATENT_OFFSET) // VIDEO_LATENT_BLOCK) * VIDEO_FRAME_BLOCK + VIDEO_FRAME_OFFSET


def frames_to_audio_latent_t(frame_count, frame_rate):
    if frame_rate <= 0:
        raise ValueError("frame_rate 必须大于 0")
    return int(round(float(frame_count) / float(frame_rate) * AUDIO_LATENT_FPS))


def fit_tensor_length(tensor, target_length, dim, pad_mode="zeros", pad_value=0.0):
    """沿指定维度裁剪或补齐 Tensor，不修改已经匹配的输入。"""
    current_length = tensor.shape[dim]
    if target_length < 1:
        raise ValueError("目标长度必须至少为 1")
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - current_length
    if pad_mode == "repeat_last":
        index = [slice(None)] * tensor.ndim
        index[dim] = slice(current_length - 1, current_length)
        repeats = [1] * tensor.ndim
        repeats[dim] = pad_shape[dim]
        padding = tensor[tuple(index)].repeat(*repeats)
    elif pad_mode == "zeros":
        padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    else:
        raise ValueError(f"不支持的补齐方式：{pad_mode}")
    return torch.cat((tensor, padding), dim=dim)


def calculate_empty_audio_length(source_type, value, frame_rate=24.0, align_mode="up"):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("长度值必须是大于 0 的有限数字")
    if source_type == "audio_latent_t":
        audio_t = max(1, int(round(value)))
    elif source_type == "seconds":
        audio_t = max(1, int(round(value * AUDIO_LATENT_FPS)))
    elif source_type in ("video_frames", "video_latent_t"):
        audio_t = convert_h3_time(source_type, value, frame_rate, align_mode)["audio_latent_t"]
    else:
        raise ValueError(f"不支持的音频长度类型：{source_type}")
    return audio_t, audio_t / AUDIO_LATENT_FPS


def validate_video_tensor(video, name="视频 latent", require_h3_channels=True):
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if video.ndim != 5:
        raise ValueError(f"{name} 必须是 [B,C,T,H,W] 五维张量，当前为 {tuple(video.shape)}")
    if require_h3_channels and video.shape[1] != VIDEO_CHANNELS:
        raise ValueError(f"{name} 必须有 {VIDEO_CHANNELS} 个通道，当前为 {video.shape[1]}")
    if video.shape[0] < 1 or video.shape[2] < 1 or video.shape[3] < 1 or video.shape[4] < 1:
        raise ValueError(f"{name} 包含空维度：{tuple(video.shape)}")


def validate_audio_tensor(audio, name="音频 latent", require_h3_channels=True):
    if not isinstance(audio, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if audio.ndim != 4:
        raise ValueError(f"{name} 必须是 [B,C,2,T] 四维张量，当前为 {tuple(audio.shape)}")
    if require_h3_channels and audio.shape[1] != AUDIO_CHANNELS:
        raise ValueError(f"{name} 必须有 {AUDIO_CHANNELS} 个通道，当前为 {audio.shape[1]}")
    if audio.shape[2] != AUDIO_STEREO_CHANNELS:
        raise ValueError(f"{name} 必须是双声道，当前声道数为 {audio.shape[2]}")
    if audio.shape[0] < 1 or audio.shape[3] < 1:
        raise ValueError(f"{name} 包含空维度：{tuple(audio.shape)}")


def split_latent_streams(latent, require_av=False):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("输入必须是包含 samples 的 LATENT 字典")

    samples = latent["samples"]
    if is_nested_tensor(samples):
        streams = list(samples.unbind())
        if len(streams) != 2:
            raise ValueError(f"MiniMax H3 AV latent 必须包含视频、音频两个流，当前为 {len(streams)} 个")
        return streams[0], streams[1], True

    if require_av:
        raise ValueError("输入不是联合 AV latent，请先提供包含视频和音频的 NestedTensor")

    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples 必须是 Tensor 或 NestedTensor")
    if samples.ndim == 5:
        return samples, None, False
    if samples.ndim == 4 and samples.shape[1] == AUDIO_CHANNELS and samples.shape[2] == AUDIO_STEREO_CHANNELS:
        return None, samples, False
    raise ValueError(f"无法识别 latent 类型，samples shape={tuple(samples.shape)}")


def split_noise_masks(latent, is_av):
    mask = latent.get("noise_mask")
    if mask is None:
        return None, None
    if is_nested_tensor(mask):
        masks = list(mask.unbind())
        if len(masks) != 2:
            raise ValueError(f"联合 noise_mask 必须包含两个流，当前为 {len(masks)} 个")
        return masks[0], masks[1]
    if is_av:
        # ComfyUI 允许只给主视频流一个普通 mask。
        return mask, None
    return mask, None


def extract_video_stream(latent, name):
    video, _, is_av = split_latent_streams(latent)
    if video is None:
        raise ValueError(f"{name} 中没有视频 latent")
    validate_video_tensor(video, name)
    video_mask, _ = split_noise_masks(latent, is_av)
    return video, video_mask


def extract_audio_stream(latent, name):
    _, audio, is_av = split_latent_streams(latent)
    if audio is None:
        raise ValueError(f"{name} 中没有音频 latent")
    validate_audio_tensor(audio, name)
    _, audio_mask = split_noise_masks(latent, is_av)
    if not is_av and latent.get("noise_mask") is not None:
        audio_mask = latent["noise_mask"]
    return audio, audio_mask


def copy_latent_with_samples(source, samples):
    output = source.copy()
    output["samples"] = samples
    return output


def build_av_noise_mask(video, audio, video_mask, audio_mask):
    if video_mask is None and audio_mask is None:
        return None
    if video_mask is None:
        video_mask = torch.ones_like(video)
    if audio_mask is None:
        audio_mask = torch.ones_like(audio)
    return comfy.nested_tensor.NestedTensor((video_mask, audio_mask))


def convert_h3_time(source_type, value, frame_rate=24.0, align_mode="up"):
    if frame_rate <= 0:
        raise ValueError("frame_rate 必须大于 0")
    if value < 0:
        raise ValueError("换算值不能小于 0")

    source_exact = True
    if source_type == "video_frames":
        rounded = round(value)
        source_exact = abs(value - rounded) < 1e-6
        frame_count, sequence_exact = align_frame_count(rounded, align_mode)
        source_exact = source_exact and sequence_exact
    elif source_type == "video_latent_t":
        rounded = round(value)
        source_exact = abs(value - rounded) < 1e-6
        latent_t, sequence_exact = align_video_latent_t(rounded, align_mode)
        source_exact = source_exact and sequence_exact
        frame_count = video_latent_t_to_frames(latent_t)
    elif source_type == "audio_latent_t":
        rounded = max(0, round(value))
        source_exact = abs(value - rounded) < 1e-6
        raw_frames = rounded / AUDIO_LATENT_FPS * frame_rate
        raw_position = (max(VIDEO_FRAME_OFFSET, raw_frames) - VIDEO_FRAME_OFFSET) / VIDEO_FRAME_BLOCK
        nearby_frames = [
            VIDEO_FRAME_OFFSET + VIDEO_FRAME_BLOCK * max(0, index)
            for index in range(math.floor(raw_position) - 2, math.ceil(raw_position) + 3)
        ]
        exact_candidates = [
            frames for frames in nearby_frames
            if frames_to_audio_latent_t(frames, frame_rate) == rounded
        ]
        if exact_candidates:
            frame_count = min(exact_candidates, key=lambda frames: abs(frames - raw_frames))
        else:
            frame_count, _ = align_frame_count(raw_frames, align_mode)
            if align_mode == "exact":
                raise ValueError(
                    f"audio latent T={rounded} 在 {frame_rate:g} FPS 下不能精确对应合法的 17k+5 视频帧数"
                )
            source_exact = False
    elif source_type == "seconds":
        raw_frames = value * frame_rate
        frame_count, sequence_exact = align_frame_count(raw_frames, align_mode)
        source_exact = abs(raw_frames - round(raw_frames)) < 1e-6 and sequence_exact
    else:
        raise ValueError(f"不支持的 source_type：{source_type}")

    video_t = frames_to_video_latent_t(frame_count)
    audio_t = frames_to_audio_latent_t(frame_count, frame_rate)
    duration = frame_count / frame_rate
    return {
        "video_frames": frame_count,
        "video_latent_t": video_t,
        "audio_latent_t": audio_t,
        "duration_seconds": duration,
        "source_exact": source_exact,
    }


class CKMiniMaxH3SeparateAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3SeparateAVLatent",
            display_name="CK MiniMax H3 Separate AV Latent",
            description="将 MiniMax H3 联合 AV latent 分离为视频 latent 和音频 latent，并同步拆分 noise mask。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[io.Latent.Input("av_latent")],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def execute(cls, av_latent):
        video, audio, is_av = split_latent_streams(av_latent, require_av=True)
        validate_video_tensor(video)
        validate_audio_tensor(audio)
        if video.shape[0] != audio.shape[0]:
            raise ValueError("视频和音频 latent 的 batch 不一致")

        video_mask, audio_mask = split_noise_masks(av_latent, is_av)
        video_output = copy_latent_with_samples(av_latent, video)
        audio_output = copy_latent_with_samples(av_latent, audio)

        video_output.pop("noise_mask", None)
        audio_output.pop("noise_mask", None)
        if video_mask is not None:
            video_output["noise_mask"] = video_mask
        if audio_mask is not None:
            audio_output["noise_mask"] = audio_mask
        return io.NodeOutput(video_output, audio_output)


class CKMiniMaxH3CombineAVLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3CombineAVLatent",
            display_name="CK MiniMax H3 Combine AV Latent",
            description="将 MiniMax H3 视频 latent 和双声道音频 latent 合并为联合 NestedTensor，并同步组合 noise mask。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Latent.Input("video_latent"),
                io.Latent.Input("audio_latent"),
                io.Combo.Input(
                    "alignment_mode",
                    options=["none", "audio_to_video"],
                    default="none",
                    tooltip="audio_to_video 会按视频 latent 对应的帧数和 FPS 强制裁剪或补齐音频。",
                ),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input(
                    "audio_pad_mode",
                    options=["zeros", "repeat_last"],
                    default="zeros",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="av_latent"),
                io.Int.Output(display_name="audio_latent_t"),
            ],
        )

    @classmethod
    def execute(cls, video_latent, audio_latent, alignment_mode="none", frame_rate=24.0, audio_pad_mode="zeros"):
        video, video_mask = extract_video_stream(video_latent, "video_latent")
        audio, audio_mask = extract_audio_stream(audio_latent, "audio_latent")
        if video.shape[0] != audio.shape[0]:
            raise ValueError(f"视频 batch={video.shape[0]}，音频 batch={audio.shape[0]}，无法合并")

        if alignment_mode == "audio_to_video":
            video_frames = video_latent_t_to_frames(video.shape[2])
            target_audio_t = frames_to_audio_latent_t(video_frames, frame_rate)
            audio = fit_tensor_length(audio, target_audio_t, -1, audio_pad_mode)
            if audio_mask is not None:
                # 新补出的音频区域使用 1，确保采样时允许模型生成。
                audio_mask = fit_tensor_length(audio_mask, target_audio_t, -1, "zeros", pad_value=1.0)
        elif alignment_mode != "none":
            raise ValueError(f"不支持的 alignment_mode：{alignment_mode}")

        output = video_latent.copy()
        for key, value in audio_latent.items():
            if key not in ("samples", "noise_mask") and key not in output:
                output[key] = value
        output["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        output.pop("noise_mask", None)
        combined_mask = build_av_noise_mask(video, audio, video_mask, audio_mask)
        if combined_mask is not None:
            output["noise_mask"] = combined_mask
        output["ck_minimax_h3_frame_rate"] = float(frame_rate)
        output["ck_minimax_h3_audio_alignment"] = alignment_mode
        return io.NodeOutput(output, audio.shape[-1])


class CKMiniMaxH3AudioVAEEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3AudioVAEEncode",
            display_name="CK MiniMax H3 Audio VAE Encode",
            description="使用 H3 音频 VAE 将音频编码为 [1,32,2,T] latent；自动重采样到 32 kHz 并规范为双声道。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Audio.Input("audio"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("batch_index", default=0, min=0, max=0x7FFFFFFF, step=1),
            ],
            outputs=[
                io.Latent.Output(display_name="audio_latent"),
                io.Int.Output(display_name="audio_latent_t"),
                io.Float.Output(display_name="duration_seconds"),
                io.Int.Output(display_name="sample_rate"),
            ],
        )

    @classmethod
    def execute(cls, audio, audio_vae, batch_index):
        if audio is None or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("audio 必须包含 waveform 和 sample_rate")
        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ValueError("audio waveform 必须是 [B,C,L] 三维张量")
        if waveform.shape[0] < 1 or waveform.shape[-1] < 1:
            raise ValueError("audio waveform 为空")
        if batch_index >= waveform.shape[0]:
            raise ValueError(f"batch_index={batch_index} 超出音频 batch={waveform.shape[0]}")

        waveform = waveform[batch_index:batch_index + 1]
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > 2:
            waveform = waveform[:, :2]
        elif waveform.shape[1] != 2:
            raise ValueError(f"不支持的音频声道数：{waveform.shape[1]}")

        source_rate = int(audio["sample_rate"])
        target_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("音频采样率必须大于 0")
        if source_rate != target_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)

        samples_per_latent = int(getattr(audio_vae, "downscale_ratio", 800))
        if samples_per_latent < 1:
            samples_per_latent = 800
        right_pad = (-waveform.shape[-1]) % samples_per_latent
        if right_pad:
            waveform = torch.nn.functional.pad(waveform, (0, right_pad))

        encoded = audio_vae.encode(waveform.movedim(1, -1))
        validate_audio_tensor(encoded, "H3 音频 VAE 编码结果")
        if encoded.shape[0] != 1:
            raise ValueError(f"H3 音频编码结果必须是 batch 1，当前为 {encoded.shape[0]}")
        output = {
            "samples": encoded,
            "ck_minimax_h3_kind": "audio",
            "ck_minimax_h3_source": "audio_vae_encode",
            "ck_minimax_h3_audio_sample_rate": target_rate,
        }
        return io.NodeOutput(output, encoded.shape[-1], encoded.shape[-1] / AUDIO_LATENT_FPS, target_rate)


class CKMiniMaxH3EmptyVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3EmptyVideoLatent",
            display_name="CK MiniMax H3 Empty Video Latent",
            description="创建合法的 H3 空视频 latent，宽高自动对齐到 32 像素，时间自动对齐到 17k+5 帧。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Int.Input("width", default=1344, min=1, max=0x7FFFFFFF, step=1),
                io.Int.Input("height", default=768, min=1, max=0x7FFFFFFF, step=1),
                io.Combo.Input(
                    "length_type",
                    options=["video_frames", "video_latent_t", "seconds"],
                    default="video_frames",
                ),
                io.Float.Input("length_value", default=124.0, min=0.001, max=1000000.0, step=0.01),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input("temporal_align", options=["up", "down", "nearest", "exact"], default="up"),
                io.Combo.Input("spatial_align", options=["nearest", "down", "up", "exact"], default="nearest"),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Int.Output(display_name="video_frames"),
                io.Int.Output(display_name="video_latent_t"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, width, height, length_type, length_value, frame_rate, temporal_align, spatial_align):
        dimensions = convert_h3_time(length_type, length_value, frame_rate, temporal_align)
        width = align_spatial_pixels(width, spatial_align)
        height = align_spatial_pixels(height, spatial_align)
        video = torch.zeros(
            (1, VIDEO_CHANNELS, dimensions["video_latent_t"], height // H3_SPATIAL_DOWNSCALE, width // H3_SPATIAL_DOWNSCALE),
            device=comfy.model_management.intermediate_device(),
        )
        output = {
            "samples": video,
            "ck_minimax_h3_kind": "video",
            "ck_minimax_h3_source": "empty_video",
            "ck_minimax_h3_frame_count": dimensions["video_frames"],
            "ck_minimax_h3_frame_rate": float(frame_rate),
        }
        return io.NodeOutput(output, dimensions["video_frames"], dimensions["video_latent_t"], width, height)


class CKMiniMaxH3EmptyAudioLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3EmptyAudioLatent",
            display_name="CK MiniMax H3 Empty Audio Latent",
            description="按音频 T、秒数或视频时长创建 [1,32,2,T] H3 空音频 latent。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Combo.Input(
                    "length_type",
                    options=["audio_latent_t", "seconds", "video_frames", "video_latent_t"],
                    default="audio_latent_t",
                ),
                io.Float.Input("length_value", default=207.0, min=0.001, max=1000000.0, step=0.01),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input("temporal_align", options=["up", "down", "nearest", "exact"], default="up"),
            ],
            outputs=[
                io.Latent.Output(display_name="audio_latent"),
                io.Int.Output(display_name="audio_latent_t"),
                io.Float.Output(display_name="duration_seconds"),
            ],
        )

    @classmethod
    def execute(cls, length_type, length_value, frame_rate, temporal_align):
        audio_t, duration = calculate_empty_audio_length(length_type, length_value, frame_rate, temporal_align)
        audio = torch.zeros(
            (1, AUDIO_CHANNELS, AUDIO_STEREO_CHANNELS, audio_t),
            device=comfy.model_management.intermediate_device(),
        )
        output = {
            "samples": audio,
            "ck_minimax_h3_kind": "audio",
            "ck_minimax_h3_source": "empty_audio",
            "ck_minimax_h3_frame_rate": float(frame_rate),
        }
        return io.NodeOutput(output, audio_t, duration)


class CKMiniMaxH3LatentResize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3LatentResize",
            display_name="CK MiniMax H3 Latent Resize",
            description="按目标分辨率或缩放倍数调整 H3 视频 latent，自动对齐到 32 像素合法尺寸；联合 AV 的音频流保持不变。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Latent.Input("latent"),
                io.Combo.Input(
                    "resize_mode",
                    options=["target_resolution", "scale_by"],
                    default="target_resolution",
                ),
                io.Int.Input("target_width", default=1344, min=1, max=0x7FFFFFFF, step=1),
                io.Int.Input("target_height", default=768, min=1, max=0x7FFFFFFF, step=1),
                io.Float.Input("scale_by", default=1.5, min=0.01, max=100.0, step=0.01),
                io.Combo.Input(
                    "align_mode",
                    options=["nearest", "down", "up", "exact"],
                    default="nearest",
                    tooltip="将最终像素宽高对齐到 32 的倍数；exact 会在输入不合法时报错。",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
                    default="bicubic",
                ),
                io.Combo.Input(
                    "crop",
                    options=["disabled", "center"],
                    default="disabled",
                    tooltip="center 会先居中裁剪到目标宽高比；disabled 会直接缩放。",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Float.Output(display_name="actual_scale_x"),
                io.Float.Output(display_name="actual_scale_y"),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        resize_mode,
        target_width,
        target_height,
        scale_by,
        align_mode,
        upscale_method,
        crop,
    ):
        video, audio, is_av = split_latent_streams(latent)
        if video is None:
            raise ValueError("输入 latent 中没有可缩放的视频流")
        validate_video_tensor(video)

        input_height = video.shape[-2] * H3_SPATIAL_DOWNSCALE
        input_width = video.shape[-1] * H3_SPATIAL_DOWNSCALE
        output_width, output_height = calculate_h3_resize(
            input_width,
            input_height,
            resize_mode,
            target_width,
            target_height,
            scale_by,
            align_mode,
        )
        latent_width = output_width // H3_SPATIAL_DOWNSCALE
        latent_height = output_height // H3_SPATIAL_DOWNSCALE

        resized_video = comfy.utils.common_upscale(
            video,
            latent_width,
            latent_height,
            upscale_method,
            crop,
        )
        validate_video_tensor(resized_video, "缩放后视频 latent")
        if resized_video.shape[-2] % H3_DIT_PATCH_SPATIAL or resized_video.shape[-1] % H3_DIT_PATCH_SPATIAL:
            raise RuntimeError("内部错误：缩放后的 latent 空间尺寸未对齐到 H3 2x2 patch")

        video_mask, audio_mask = split_noise_masks(latent, is_av)
        resized_video_mask = None
        if video_mask is not None:
            resized_video_mask = comfy.utils.common_upscale(
                video_mask,
                latent_width,
                latent_height,
                "nearest-exact",
                crop,
            )

        output = latent.copy()
        if is_av:
            output["samples"] = comfy.nested_tensor.NestedTensor((resized_video, audio))
        else:
            output["samples"] = resized_video

        output.pop("noise_mask", None)
        if is_av:
            combined_mask = build_av_noise_mask(resized_video, audio, resized_video_mask, audio_mask)
            if combined_mask is not None:
                output["noise_mask"] = combined_mask
        elif resized_video_mask is not None:
            output["noise_mask"] = resized_video_mask

        output["ck_minimax_h3_resize"] = {
            "input_width": input_width,
            "input_height": input_height,
            "output_width": output_width,
            "output_height": output_height,
            "resize_mode": resize_mode,
            "align_mode": align_mode,
        }
        return io.NodeOutput(
            output,
            output_width,
            output_height,
            output_width / input_width,
            output_height / input_height,
        )


class CKMiniMaxH3ImageVAEEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3ImageVAEEncode",
            display_name="CK MiniMax H3 Image VAE Encode",
            description="使用 MiniMax H3 视频 VAE 将指定图片编码为单时间位置的 24 通道视频 latent。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.Int.Input("batch_index", default=0, min=0, max=0x7FFFFFFF, step=1),
                io.Combo.Input(
                    "canvas_mode",
                    options=["center_crop_to_32", "resize_to_nearest_32", "keep"],
                    default="center_crop_to_32",
                    tooltip="H3 DiT 的视频 latent 使用 2x2 patch，像素宽高最好是 32 的倍数。",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="video_latent"),
                io.Int.Output(display_name="encoded_width"),
                io.Int.Output(display_name="encoded_height"),
            ],
        )

    @staticmethod
    def prepare_image(image, batch_index, canvas_mode):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("image 必须是 [B,H,W,C] 四维 IMAGE 张量")
        if image.shape[0] < 1:
            raise ValueError("image batch 为空")
        if batch_index >= image.shape[0]:
            raise ValueError(f"batch_index={batch_index} 超出图片 batch={image.shape[0]}")

        selected = image[batch_index:batch_index + 1, ..., :3]
        height, width = selected.shape[1:3]
        if canvas_mode == "center_crop_to_32":
            target_width = width // H3_CANVAS_MULTIPLE * H3_CANVAS_MULTIPLE
            target_height = height // H3_CANVAS_MULTIPLE * H3_CANVAS_MULTIPLE
            if target_width < H3_CANVAS_MULTIPLE or target_height < H3_CANVAS_MULTIPLE:
                raise ValueError("图片宽高必须至少为 32 像素")
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            selected = selected[:, top:top + target_height, left:left + target_width]
        elif canvas_mode == "resize_to_nearest_32":
            target_width = max(H3_CANVAS_MULTIPLE, round(width / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
            target_height = max(H3_CANVAS_MULTIPLE, round(height / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
            selected = comfy.utils.common_upscale(
                selected.movedim(-1, 1), target_width, target_height, "lanczos", "disabled"
            ).movedim(1, -1)
        elif canvas_mode != "keep":
            raise ValueError(f"不支持的 canvas_mode：{canvas_mode}")
        return selected

    @classmethod
    def execute(cls, image, vae, batch_index, canvas_mode):
        selected = cls.prepare_image(image, batch_index, canvas_mode)
        encoded = vae.encode(selected)
        validate_video_tensor(encoded, "H3 VAE 编码结果")
        if encoded.shape[0] != 1:
            raise ValueError(f"H3 图片编码应返回 batch 1，当前为 {encoded.shape[0]}")
        if encoded.shape[2] != 1:
            raise ValueError(f"H3 图片编码应返回 T=1，当前为 T={encoded.shape[2]}；请确认使用的是 MiniMax H3 视频 VAE")
        if encoded.shape[3] % H3_DIT_PATCH_SPATIAL != 0 or encoded.shape[4] % H3_DIT_PATCH_SPATIAL != 0:
            raise ValueError(
                f"编码后的 latent 空间尺寸 {encoded.shape[4]}x{encoded.shape[3]} 不能被 H3 的 2x2 patch 整除；"
                "请使用裁剪或缩放到 32 倍数的模式"
            )
        output = {
            "samples": encoded,
            "ck_minimax_h3_kind": "video",
            "ck_minimax_h3_source": "image_vae_encode",
        }
        return io.NodeOutput(output, encoded.shape[4] * H3_SPATIAL_DOWNSCALE, encoded.shape[3] * H3_SPATIAL_DOWNSCALE)


class CKMiniMaxH3ReplaceVideoLatentByIndex(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3ReplaceVideoLatentByIndex",
            display_name="CK MiniMax H3 Replace Video Latent By Index",
            description="从指定 video latent 时间索引开始，用 replacement_latent 替换目标视频 latent；联合 AV 输入的音频流保持不变。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Latent.Input("target_latent"),
                io.Latent.Input("replacement_latent"),
                io.Int.Input("time_index", default=0, min=0, max=0x7FFFFFFF, step=1,
                             tooltip="视频 latent 的 T 索引，不是像素视频帧索引。"),
                io.Combo.Input("overflow_mode", options=["error", "trim"], default="error"),
                io.Combo.Input(
                    "noise_mask_mode",
                    options=["preserve_target", "use_replacement_if_present", "freeze_replaced", "denoise_replaced"],
                    default="preserve_target",
                    tooltip="控制被替换区段的 noise mask：保留目标、使用替换输入、冻结为0、允许去噪为1。",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="replaced_start"),
                io.Int.Output(display_name="replaced_length"),
                io.Int.Output(display_name="replaced_end_exclusive"),
            ],
        )

    @classmethod
    def execute(cls, target_latent, replacement_latent, time_index, overflow_mode, noise_mask_mode):
        target_video, target_audio, target_is_av = split_latent_streams(target_latent)
        if target_video is None:
            raise ValueError("target_latent 中没有视频 latent")
        validate_video_tensor(target_video, "target_latent 视频流")

        replacement_video, replacement_mask = extract_video_stream(replacement_latent, "replacement_latent")
        if target_video.shape[0] != replacement_video.shape[0]:
            raise ValueError("目标和替换视频 latent 的 batch 不一致")
        if target_video.shape[1] != replacement_video.shape[1]:
            raise ValueError("目标和替换视频 latent 的通道数不一致")
        if target_video.shape[3:] != replacement_video.shape[3:]:
            raise ValueError(
                "目标和替换视频 latent 的空间尺寸必须一致："
                f"target={tuple(target_video.shape[3:])}, replacement={tuple(replacement_video.shape[3:])}"
            )
        if time_index >= target_video.shape[2]:
            raise ValueError(f"time_index={time_index} 超出目标视频 latent T={target_video.shape[2]}")

        available = target_video.shape[2] - time_index
        replacement_t = replacement_video.shape[2]
        if replacement_t > available and overflow_mode == "error":
            raise ValueError(
                f"替换区间 [{time_index}, {time_index + replacement_t}) 超出目标 T={target_video.shape[2]}；"
                "可将 overflow_mode 改为 trim"
            )
        replace_length = min(replacement_t, available)
        end = time_index + replace_length

        output_video = target_video.clone()
        output_video[:, :, time_index:end].copy_(
            replacement_video[:, :, :replace_length].to(device=output_video.device, dtype=output_video.dtype)
        )

        target_video_mask, target_audio_mask = split_noise_masks(target_latent, target_is_av)
        output_video_mask = target_video_mask
        should_edit_mask = noise_mask_mode != "preserve_target"
        if noise_mask_mode == "use_replacement_if_present" and replacement_mask is None:
            should_edit_mask = False
        if should_edit_mask:
            if output_video_mask is None:
                output_video_mask = torch.ones_like(target_video)
            else:
                output_video_mask = comfy.utils.reshape_mask(output_video_mask, target_video.shape).clone()

            if noise_mask_mode == "use_replacement_if_present" and replacement_mask is not None:
                replacement_mask = comfy.utils.reshape_mask(replacement_mask, replacement_video.shape)
                output_video_mask[:, :, time_index:end].copy_(
                    replacement_mask[:, :, :replace_length].to(output_video_mask)
                )
            elif noise_mask_mode == "freeze_replaced":
                output_video_mask[:, :, time_index:end] = 0.0
            elif noise_mask_mode == "denoise_replaced":
                output_video_mask[:, :, time_index:end] = 1.0

        output = target_latent.copy()
        output.pop("noise_mask", None)
        if target_is_av:
            validate_audio_tensor(target_audio, "target_latent 音频流")
            output["samples"] = comfy.nested_tensor.NestedTensor((output_video, target_audio))
            combined_mask = build_av_noise_mask(output_video, target_audio, output_video_mask, target_audio_mask)
            if combined_mask is not None:
                output["noise_mask"] = combined_mask
        else:
            output["samples"] = output_video
            if output_video_mask is not None:
                output["noise_mask"] = output_video_mask
        return io.NodeOutput(output, time_index, replace_length, end)


class CKMiniMaxH3LatentInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3LatentInfo",
            display_name="CK MiniMax H3 Latent Info",
            description="读取 MiniMax H3 视频、音频或联合 AV latent 的形状、帧数、时长和合法性信息。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Latent.Input("latent"),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
            ],
            outputs=[
                io.String.Output(display_name="info"),
                io.Boolean.Output(display_name="is_av"),
                io.Boolean.Output(display_name="is_valid_h3"),
                io.Int.Output(display_name="batch_size"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="video_frames"),
                io.Int.Output(display_name="video_latent_t"),
                io.Int.Output(display_name="audio_latent_t"),
                io.Float.Output(display_name="duration_seconds"),
            ],
        )

    @classmethod
    def execute(cls, latent, frame_rate):
        if frame_rate <= 0:
            raise ValueError("frame_rate 必须大于 0")
        video, audio, is_av = split_latent_streams(latent)
        warnings = []
        valid = True

        batch_size = 0
        width = height = 0
        video_frames = video_t = audio_t = 0
        video_duration = audio_duration = 0.0

        if video is not None:
            try:
                validate_video_tensor(video)
            except (TypeError, ValueError) as error:
                valid = False
                warnings.append(str(error))
            batch_size = video.shape[0]
            video_t = video.shape[2]
            height = video.shape[3] * H3_SPATIAL_DOWNSCALE
            width = video.shape[4] * H3_SPATIAL_DOWNSCALE
            if video_t == 1:
                video_frames = 1
                video_duration = 1.0 / frame_rate
                warnings.append("T=1 是图片 VAE 条件/替换 latent，不是 H3 目标生成序列")
            else:
                try:
                    video_frames = video_latent_t_to_frames(video_t)
                    video_duration = video_frames / frame_rate
                except ValueError:
                    valid = False
                    video_frames = -1
                    warnings.append(f"video latent T={video_t} 不满足 5k+2，无法精确换算合法像素帧数")
            if video.shape[3] % H3_DIT_PATCH_SPATIAL or video.shape[4] % H3_DIT_PATCH_SPATIAL:
                valid = False
                warnings.append("视频 latent 高宽不能被 H3 DiT 的 2x2 patch 整除")

        if audio is not None:
            try:
                validate_audio_tensor(audio)
            except (TypeError, ValueError) as error:
                valid = False
                warnings.append(str(error))
            if batch_size and batch_size != audio.shape[0]:
                valid = False
                warnings.append("视频和音频 batch 不一致")
            batch_size = batch_size or audio.shape[0]
            audio_t = audio.shape[3]
            audio_duration = audio_t / AUDIO_LATENT_FPS

        if batch_size != 1:
            valid = False
            warnings.append(f"当前 batch={batch_size}，MiniMax H3 DiT 采样要求 batch=1")

        if video is not None and audio is not None and video_frames >= 0:
            expected_audio_t = frames_to_audio_latent_t(video_frames, frame_rate)
            if audio_t != expected_audio_t:
                warnings.append(
                    f"按 {frame_rate:g} FPS，视频期望 audio T={expected_audio_t}，当前为 {audio_t}"
                )

        duration = video_duration if video is not None and video_frames >= 0 else audio_duration
        kind = "AV" if is_av else ("video" if video is not None else "audio")
        lines = [
            "MiniMax H3 Latent 信息",
            f"类型: {kind}",
            f"batch: {batch_size}",
            f"frame_rate: {frame_rate:g}",
        ]
        if video is not None:
            lines.extend([
                f"video shape: {tuple(video.shape)}",
                f"像素尺寸: {width}x{height}",
                f"video latent T: {video_t}",
                f"换算视频帧数: {video_frames}",
            ])
        if audio is not None:
            lines.extend([
                f"audio shape: {tuple(audio.shape)}",
                f"audio latent T: {audio_t}",
                f"音频时长: {audio_duration:.6f}s",
            ])
        lines.append(f"换算时长: {duration:.6f}s")
        lines.append(f"H3 结构合法: {'是' if valid else '否'}")
        if warnings:
            lines.append("提示:")
            lines.extend(f"- {warning}" for warning in warnings)

        return io.NodeOutput(
            "\n".join(lines), is_av, valid, batch_size, width, height,
            video_frames, video_t, audio_t, duration,
        )


class CKMiniMaxH3TimeConvert(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3TimeConvert",
            display_name="CK MiniMax H3 Frame/Latent Convert",
            description="在视频帧数、视频 latent T、音频 latent T 和秒之间快捷换算，并按 H3 合法序列对齐。",
            category="CK Nodes/MiniMax H3/Latent",
            inputs=[
                io.Combo.Input(
                    "source_type",
                    options=["video_frames", "video_latent_t", "audio_latent_t", "seconds"],
                    default="video_frames",
                ),
                io.Float.Input("value", default=124.0, min=0.0, max=1000000.0, step=0.01),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Combo.Input("align_mode", options=["up", "down", "nearest", "exact"], default="up"),
            ],
            outputs=[
                io.Int.Output(display_name="video_frames"),
                io.Int.Output(display_name="video_latent_t"),
                io.Int.Output(display_name="audio_latent_t"),
                io.Float.Output(display_name="duration_seconds"),
                io.Boolean.Output(display_name="source_was_exact"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, source_type, value, frame_rate, align_mode):
        result = convert_h3_time(source_type, value, frame_rate, align_mode)
        info = (
            f"输入: {source_type}={value:g}\n"
            f"对齐: {align_mode}\n"
            f"FPS: {frame_rate:g}\n"
            f"视频帧数: {result['video_frames']}\n"
            f"视频 latent T: {result['video_latent_t']}\n"
            f"音频 latent T: {result['audio_latent_t']}\n"
            f"时长: {result['duration_seconds']:.6f}s\n"
            f"输入无需对齐: {'是' if result['source_exact'] else '否'}"
        )
        return io.NodeOutput(
            result["video_frames"],
            result["video_latent_t"],
            result["audio_latent_t"],
            result["duration_seconds"],
            result["source_exact"],
            info,
        )


NODE_CLASS_MAPPINGS = {
    "CKMiniMaxH3SeparateAVLatent": CKMiniMaxH3SeparateAVLatent,
    "CKMiniMaxH3CombineAVLatent": CKMiniMaxH3CombineAVLatent,
    "CKMiniMaxH3AudioVAEEncode": CKMiniMaxH3AudioVAEEncode,
    "CKMiniMaxH3EmptyVideoLatent": CKMiniMaxH3EmptyVideoLatent,
    "CKMiniMaxH3EmptyAudioLatent": CKMiniMaxH3EmptyAudioLatent,
    "CKMiniMaxH3LatentResize": CKMiniMaxH3LatentResize,
    "CKMiniMaxH3ImageVAEEncode": CKMiniMaxH3ImageVAEEncode,
    "CKMiniMaxH3ReplaceVideoLatentByIndex": CKMiniMaxH3ReplaceVideoLatentByIndex,
    "CKMiniMaxH3LatentInfo": CKMiniMaxH3LatentInfo,
    "CKMiniMaxH3TimeConvert": CKMiniMaxH3TimeConvert,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CKMiniMaxH3SeparateAVLatent": "CK MiniMax H3 Separate AV Latent",
    "CKMiniMaxH3CombineAVLatent": "CK MiniMax H3 Combine AV Latent",
    "CKMiniMaxH3AudioVAEEncode": "CK MiniMax H3 Audio VAE Encode",
    "CKMiniMaxH3EmptyVideoLatent": "CK MiniMax H3 Empty Video Latent",
    "CKMiniMaxH3EmptyAudioLatent": "CK MiniMax H3 Empty Audio Latent",
    "CKMiniMaxH3LatentResize": "CK MiniMax H3 Latent Resize",
    "CKMiniMaxH3ImageVAEEncode": "CK MiniMax H3 Image VAE Encode",
    "CKMiniMaxH3ReplaceVideoLatentByIndex": "CK MiniMax H3 Replace Video Latent By Index",
    "CKMiniMaxH3LatentInfo": "CK MiniMax H3 Latent Info",
    "CKMiniMaxH3TimeConvert": "CK MiniMax H3 Frame/Latent Convert",
}
