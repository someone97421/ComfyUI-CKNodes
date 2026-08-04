import math

import torch
import torchaudio

import comfy.ldm.minimax.model as minimax_model
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
import nodes
from comfy_api.latest import io


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
AUDIO_LATENT_FPS = 40


def align_frame_count(frame_count):
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length, frame_rate):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / frame_rate
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width, nominal_height = nominal_width * scale, nominal_height * scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resize_image(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def empty_av_latent(width, height, length, frame_rate, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length, frame_rate)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2, audio_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


class CKMiniMaxH3ReferenceBlocks(list):
    def __init__(self, items, frame_rate):
        super().__init__(items)
        self.ck_frame_rate = float(frame_rate)
        self.ck_frame_rescale = AUDIO_LATENT_FPS / self.ck_frame_rate


def video_t_spans(count, frame_rescale):
    return [frame_rescale * minimax_model.FRAME_PER_TOKEN[index % 5] for index in range(count)]


def video_t_grid(count, origin, frame_rescale):
    spans = torch.tensor(video_t_spans(count, frame_rescale), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def video_grid(latent_t, frame, cursor, frame_rescale):
    grid = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
    grid[:, :, 0] = video_t_grid(latent_t, cursor, frame_rescale)[:, None]
    grid[:, :, 1:] = frame[None]
    return grid.reshape(-1, 3)


class CKMiniMaxH3PackedLayout:
    """H3 packed layout with video time coordinates derived from the selected FPS."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        target_frame_rescale = refs.ck_frame_rescale
        frame, width_grid = minimax_model._frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]
        grid = torch.zeros(text_len, 3, dtype=torch.float64)
        grid[:, 0] = torch.arange(text_len, dtype=torch.float64)
        positions = [grid]

        image_positions = []
        image_updates = []
        audio_positions = []
        audio_updates = []
        cursor = text_len
        row = text_len

        if keyframes:
            for keyframe in keyframes:
                pixel_index = keyframe["resolved_frame_index"]
                if pixel_index == 0:
                    condition_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    condition_t = float(text_len) + sum(video_t_spans(latent_t, target_frame_rescale)) - target_frame_rescale
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                grid = torch.empty(frame_rows, 3, dtype=torch.float64)
                grid[:, 0] = condition_t
                grid[:, 1:] = frame
                segments.append(("cond", frame_rows))
                positions.append(grid)
                image_positions.append(torch.arange(row, row + frame_rows))
                image_updates.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

        target_audio_width = (float(width_grid[0]), float(width_grid[-1]))
        if refs:
            cursor = float(text_len)
            for block in refs:
                kind = block["kind"]
                reference_frame_rescale = float(block.get("ck_frame_rescale", target_frame_rescale))
                if kind == "image":
                    reference_frame, _ = minimax_model._frame_grid(block["latent_h"], block["latent_w"])
                    count = reference_frame.shape[0]
                    grid = torch.empty(count, 3, dtype=torch.float64)
                    grid[:, 0] = cursor
                    grid[:, 1:] = reference_frame
                    segments.append(("ref_img", count))
                    positions.append(grid)
                    image_positions.append(torch.arange(row, row + count))
                    image_updates.append(torch.zeros(count, dtype=torch.bool))
                    row += count
                    cursor += 1.0
                elif kind == "audio":
                    reference_audio_t = block["ref_audio_t"]
                    if reference_audio_t > 0:
                        count = reference_audio_t * 2
                        segments.append(("ref_audio", count))
                        positions.append(minimax_model._audio_grid(cursor, reference_audio_t, *target_audio_width))
                        audio_positions.append(torch.arange(row, row + count))
                        audio_updates.append(torch.zeros(count, dtype=torch.bool))
                        row += count
                    cursor += float(reference_audio_t)
                elif kind in ("video", "video_audio"):
                    reference_audio_t = block["ref_audio_t"]
                    reference_latent_t = block["latent_t"]
                    reference_frame, reference_width_grid = minimax_model._frame_grid(block["latent_h"], block["latent_w"])
                    if reference_audio_t > 0:
                        count = reference_audio_t * 2
                        segments.append(("ref_audio", count))
                        positions.append(minimax_model._audio_grid(
                            cursor,
                            reference_audio_t,
                            float(reference_width_grid[0]),
                            float(reference_width_grid[-1]),
                        ))
                        audio_positions.append(torch.arange(row, row + count))
                        audio_updates.append(torch.zeros(count, dtype=torch.bool))
                        row += count
                    count = reference_latent_t * reference_frame.shape[0]
                    segments.append(("ref_img", count))
                    positions.append(video_grid(reference_latent_t, reference_frame, cursor, reference_frame_rescale))
                    image_positions.append(torch.arange(row, row + count))
                    image_updates.append(torch.zeros(count, dtype=torch.bool))
                    row += count
                    cursor += max(
                        float(reference_audio_t),
                        sum(video_t_spans(reference_latent_t, reference_frame_rescale)),
                    )

        count = audio_t * 2
        segments.append(("audio", count))
        positions.append(minimax_model._audio_grid(cursor, audio_t, *target_audio_width))
        audio_positions.append(torch.arange(row, row + count))
        audio_updates.append(torch.ones(count, dtype=torch.bool))
        row += count

        count = latent_t * frame_rows
        segments.append(("video", count))
        positions.append(video_grid(latent_t, frame, cursor, target_frame_rescale))
        image_positions.append(torch.arange(row, row + count))
        image_updates.append(torch.ones(count, dtype=torch.bool))
        row += count

        self.seq_len = row
        self.position_ids = torch.cat(positions)
        self.img_pos = torch.cat(image_positions)
        self.img_update = torch.cat(image_updates)
        self.audio_pos = torch.cat(audio_positions)
        self.audio_update = torch.cat(audio_updates)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)

        absolute_segments = []
        offset = 0
        for kind, count in segments:
            absolute_segments.append((offset, offset + count, kind))
            offset += count
        self.segments = absolute_segments


def install_packed_layout_dispatch():
    current_layout = minimax_model.PackedLayout
    if getattr(current_layout, "_ck_fps_dispatch", False):
        return

    original_layout = current_layout

    def packed_layout(text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        if hasattr(refs, "ck_frame_rescale"):
            return CKMiniMaxH3PackedLayout(
                text_len,
                latent_t,
                latent_h,
                latent_w,
                audio_t,
                keyframes=keyframes,
                refs=refs,
                frame_count=frame_count,
            )
        return original_layout(
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=keyframes,
            refs=refs,
            frame_count=frame_count,
        )

    packed_layout._ck_fps_dispatch = True
    packed_layout._ck_original_layout = original_layout
    minimax_model.PackedLayout = packed_layout


install_packed_layout_dispatch()


class CKMiniMaxH3ReferenceToVideoFPS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKMiniMaxH3ReferenceToVideoFPS",
            display_name="CK MiniMax H3 Reference to Video (Adjustable FPS)",
            description="MiniMax H3 local reference conditioning with automatic audio duration, Qwen timestamps, and DiT timeline adjustment for the selected frame rate.",
            category="CKNodes/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count, snapped up to the model's 17k+5 grid."),
                io.Float.Input("frame_rate", default=16.0, min=1.0, max=120.0, step=0.01,
                    tooltip="Timeline FPS. Lower values stretch the same frame count over more time. Set the final video encoder to the same FPS."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' limits pixel area to the target; 'max' allows a 2048px short edge."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference frames interpreted at frame_rate."),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @staticmethod
    def encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
        if sample_rate != vae_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
        latent = audio_vae.encode(waveform[:1].movedim(1, -1))
        return latent, latent.shape[-1]

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        length,
        frame_rate=16.0,
        ref_image_size="match",
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        latent, frame_count = empty_av_latent(width, height, length, frame_rate)
        reference_items = []
        reference_blocks = []

        for image in (ref_images or {}).values():
            if image is None:
                continue
            image_height, image_width = image.shape[1], image.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (image_width * image_height)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(image_width, image_height))
            target_width = max(CANVAS_MULTIPLE, round(image_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            target_height = max(CANVAS_MULTIPLE, round(image_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = resize_image(image[:1], target_width, target_height, "disabled")
            encoded = vae.encode(resized)
            reference_items.append({"type": "image", "data": resized})
            reference_blocks.append({
                "kind": "image",
                "latent_h": target_height // 16,
                "latent_w": target_width // 16,
                "latent": encoded,
            })

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            video_height, video_width = video_frames.shape[1], video_frames.shape[2]
            canvas_width, canvas_height = adapt_canvas(video_width, video_height)
            if video_width * video_height < canvas_width * canvas_height:
                canvas_width = max(CANVAS_MULTIPLE, round(video_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                canvas_height = max(CANVAS_MULTIPLE, round(video_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = resize_image(video_frames, canvas_width, canvas_height, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            aligned_count = frames.shape[0]
            if aligned_count < 5:
                raise ValueError(f"MiniMax H3 reference videos need at least 5 frames ({5 / frame_rate:.2f}s at {frame_rate:g} fps)")
            while aligned_count % 17 != 5:
                aligned_count -= 1
            frames = frames[:aligned_count]
            encoded = vae.encode(frames)

            audio_latent, reference_audio_t = None, 0
            if soundtrack is not None:
                audio_latent, reference_audio_t = cls.encode_ref_audio(audio_vae, soundtrack)
                reference_items.append({"type": "audio"})

            sample_step = max(1, round(frame_rate / 2.0))
            sample_indices = list(range(0, frames.shape[0], sample_step))
            qwen_frames = frames[sample_indices]
            reference_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [index / frame_rate for index in sample_indices],
            })
            reference_blocks.append({
                "kind": "video_audio" if reference_audio_t else "video",
                "latent_t": encoded.shape[2],
                "latent_h": canvas_height // 16,
                "latent_w": canvas_width // 16,
                "ref_audio_t": reference_audio_t,
                "latent": encoded,
                "audio_latent": audio_latent,
                "ck_frame_rescale": AUDIO_LATENT_FPS / frame_rate,
            })

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, reference_audio_t = cls.encode_ref_audio(audio_vae, audio)
            reference_items.append({"type": "audio"})
            reference_blocks.append({
                "kind": "audio",
                "ref_audio_t": reference_audio_t,
                "audio_latent": audio_latent,
            })

        tokens = clip.tokenize(prompt, minimax_ref_items=reference_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        refs = CKMiniMaxH3ReferenceBlocks(reference_blocks, frame_rate)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": refs})
        return io.NodeOutput(conditioning, latent)


NODE_CLASS_MAPPINGS = {
    "CKMiniMaxH3ReferenceToVideoFPS": CKMiniMaxH3ReferenceToVideoFPS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKMiniMaxH3ReferenceToVideoFPS": "CK MiniMax H3 Reference to Video (Adjustable FPS)",
}
