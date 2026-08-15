from fractions import Fraction

import torch


def _fps_fraction(value):
    """使用十进制字符串构造有理数，避免浮点步进产生累计误差。"""
    fps = Fraction(str(value))
    if fps <= 0:
        raise ValueError("FPS 必须大于 0。")
    return fps


def _round_half_up(value):
    """对非负有理数执行四舍五入，避免 Python 银行家舍入的不确定边界。"""
    return (value.numerator * 2 + value.denominator) // (value.denominator * 2)


def calculate_frame_indices(frame_count, input_fps, output_fps):
    """
    按时间戳最近邻重采样计算源帧索引。

    输入的 N 帧被视为覆盖 N / input_fps 秒。输出帧数取最接近
    N * output_fps / input_fps 的整数；输出帧 j 对应时间 j / output_fps，
    并匹配时间上最近的输入帧。
    """
    if frame_count < 0:
        raise ValueError("帧数不能为负数。")
    if frame_count == 0:
        return []

    input_rate = _fps_fraction(input_fps)
    output_rate = _fps_fraction(output_fps)
    output_count = max(1, _round_half_up(frame_count * output_rate / input_rate))

    indices = []
    for output_index in range(output_count):
        source_position = Fraction(output_index) * input_rate / output_rate
        source_index = _round_half_up(source_position)
        indices.append(min(source_index, frame_count - 1))
    return indices


def _compact_indices(indices):
    """将索引压缩为便于阅读的信息，同时保留重复帧次数。"""
    if not indices:
        return "(空)"

    runs = []
    for index in indices:
        if runs and runs[-1][0] == index:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])

    groups = []
    position = 0
    while position < len(runs):
        index, count = runs[position]
        if count > 1:
            groups.append(f"{index}×{count}")
            position += 1
            continue

        end = position
        while (
            end + 1 < len(runs)
            and runs[end + 1][1] == 1
            and runs[end + 1][0] == runs[end][0] + 1
        ):
            end += 1

        end_index = runs[end][0]
        groups.append(str(index) if end_index == index else f"{index}-{end_index}")
        position = end + 1

    return ", ".join(groups)


def build_match_info(frame_count, indices, input_fps, output_fps):
    input_duration = frame_count / float(input_fps) if frame_count else 0.0
    output_duration = len(indices) / float(output_fps) if indices else 0.0
    duration_error = output_duration - input_duration
    unique_count = len(set(indices))
    duplicate_count = len(indices) - unique_count
    dropped_count = max(0, frame_count - unique_count)
    mode = "抽帧" if output_fps < input_fps else "重复帧补齐" if output_fps > input_fps else "原样输出"

    return "\n".join(
        [
            "CK 帧率匹配结果",
            f"模式: {mode}（时间轴最近邻匹配，无插值）",
            f"输入: {frame_count} 帧 @ {float(input_fps):.6g} FPS，时长 {input_duration:.6f} 秒",
            f"输出: {len(indices)} 帧 @ {float(output_fps):.6g} FPS，时长 {output_duration:.6f} 秒",
            f"时长误差: {duration_error:+.9f} 秒",
            f"使用源帧: {unique_count} 帧；丢弃: {dropped_count} 帧；重复输出: {duplicate_count} 帧",
            f"源帧索引（从 0 开始）: {_compact_indices(indices)}",
        ]
    )


class MatchBatchFrameRate:
    """将 IMAGE 批次按输入、输出 FPS 进行时间轴匹配。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "按时间顺序排列的输入帧批次。"}),
                "input_fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.001,
                        "round": 0.000001,
                        "tooltip": "输入帧序列原本对应的帧率。",
                    },
                ),
                "output_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.001,
                        "round": 0.000001,
                        "tooltip": "希望输出帧序列对应的帧率。高于输入 FPS 时会重复帧。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    OUTPUT_TOOLTIPS = (
        "完成帧率匹配后的 IMAGE 批次。",
        "帧数、时长、误差、丢帧/重复帧数量及源帧索引。",
    )
    FUNCTION = "match_frame_rate"
    CATEGORY = "CK Nodes/Video/Batch"
    DESCRIPTION = "依据输入和输出 FPS，在时间轴上自动匹配最接近的源帧，避免固定间隔抽帧造成累计漂移。"

    def match_frame_rate(self, images, input_fps, output_fps):
        frame_count = int(images.shape[0])
        indices = calculate_frame_indices(frame_count, input_fps, output_fps)
        info = build_match_info(frame_count, indices, input_fps, output_fps)

        if indices:
            index_tensor = torch.tensor(indices, dtype=torch.long, device=images.device)
            output_images = torch.index_select(images, 0, index_tensor)
        else:
            output_images = images

        print(info)
        return output_images, info


NODE_CLASS_MAPPINGS = {
    "CKMatchBatchFrameRate": MatchBatchFrameRate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKMatchBatchFrameRate": "CK Match Batch Frame Rate",
}
