# MiniMax H3 Latent 实现与技术特征调查

## 1. 文档目的

本文记录对当前扩展父级 ComfyUI 主仓库中 MiniMax H3 原生实现的调查结果，重点说明：

- MiniMax H3 在 ComfyUI 中的模块划分与调用关系。
- 视频、音频和联合 AV latent 的数据结构。
- 视频 VAE 的非均匀时间压缩规则。
- 音视频联合 DiT 的 token 打包与时间坐标。
- 视频和音频使用不同 flow shift 时的采样适配。
- 通用 ComfyUI latent 节点对 H3 的兼容边界。
- 后续在 `ComfyUI-CKNodes` 中开发 H3 latent 操作节点时必须遵守的约束。

本文以 2026 年 8 月 7 日的本地代码和本地模型文件为调查基准。调查期间未修改父级 ComfyUI 源码。

## 2. 调查范围

当前扩展目录：

```text
D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-CKNodes
```

ComfyUI 主目录：

```text
D:\ComfyUI\ComfyUI
```

重点检查的源码如下。表中的路径均相对于 ComfyUI 主目录。

| 模块 | 路径 | 作用 |
|---|---|---|
| H3 原生节点 | `comfy_extras/nodes_minimax_h3.py` | 创建空 AV latent、首尾帧条件、参考媒体条件和 Sigma Shift |
| H3 DiT | `comfy/ldm/minimax/model.py` | 音视频 token 打包、位置编码、联合去噪和输出拆分 |
| 视频 VAE | `comfy/ldm/minimax/vae.py` | 像素视频与 24 通道视频 latent 互转 |
| 音频 VAE | `comfy/ldm/minimax/audio_vae.py` | 32 kHz 立体声音频与 32 通道音频 latent 互转 |
| 文本/视觉编码器 | `comfy/text_encoders/minimax.py` | Qwen3-VL 条件展示序列和模态标签 |
| 模型包装 | `comfy/model_base.py` | 构造 H3 条件 payload，处理音频采样缩放 |
| 模型识别 | `comfy/supported_models.py` | 注册 H3 latent 格式、推理 dtype 和默认 flow shift |
| Latent 格式 | `comfy/latent_formats.py` | 声明视频 latent 通道数、维度和压缩比例 |
| 多模态容器 | `comfy/nested_tensor.py` | 保存视频和音频两个不同形状的 latent 流 |
| 多模态打包 | `comfy/utils.py` | 将多个 latent 流扁平化供采样器使用，并在之后还原 |
| 采样器 | `comfy/samplers.py` | 在采样前打包 NestedTensor，在采样后拆包 |
| AV flow 调度 | `comfy/model_sampling.py` | 将音频流映射到视频采样时间线 |
| 通用 AV 拆合 | `comfy_extras/nodes_lt.py` | 拆分、合并和替换 AV latent 流 |

## 3. 本地实现状态

本地 Git 历史显示：

- 2026 年 8 月 2 日，MiniMax H3 原生支持首次合入 ComfyUI。
- 2026 年 8 月 3 日，进行了 VAE 参数设备转换等修复。
- 2026 年 8 月 6 日，修复了 MiniMax H3 音频流与普通采样器、SDE 采样器的兼容问题。

因此该实现仍处于快速演进阶段。后续自定义节点应尽量通过公开的数据结构和 H3 原生辅助逻辑工作，避免复制过多内部采样器实现。

## 4. 总体调用关系

```text
MiniMax H3 条件节点
    |
    |-- prompt / 图片 / 抽样视频帧
    |       -> Qwen3-VL-32B
    |       -> 文本隐藏状态 + token 模态标签
    |
    |-- 图片 / 完整参考视频
    |       -> 视频 VAE
    |       -> 视频条件 latent
    |
    |-- 参考音频
    |       -> 音频 VAE
    |       -> 音频条件 latent
    |
    `-- 目标尺寸、帧数、FPS
            -> 空视频 latent + 空音频 latent
            -> NestedTensor

以上内容进入 H3 PackedLayout：

[text | keyframe/reference blocks | target audio | target video]
                            |
                            v
                   MiniMax H3 单流 DiT
                            |
                            v
                    视频、音频联合采样
                            |
                            v
             NestedTensor(video, audio)
                    |                 |
                    v                 v
                视频 VAE           音频 VAE
                    |                 |
                    v                 v
                 视频帧            立体声音频
```

## 5. 联合 AV Latent 数据结构

H3 的采样输入不是普通 Tensor，而是两个 Tensor 组成的 `NestedTensor`：

```python
{
    "samples": comfy.nested_tensor.NestedTensor((
        video_latent,
        audio_latent,
    ))
}
```

### 5.1 视频流

```text
[B, 24, video_T, latent_H, latent_W]
```

其中：

```text
latent_H = pixel_height / 16
latent_W = pixel_width / 16
```

视频 latent 的技术特征：

- 24 个 latent 通道。
- 5D Tensor。
- 空间压缩率为 16。
- 名义时间压缩率为 4，但真实帧数映射还受到 17 帧分块和 token drop 影响。
- latent 已经过逐通道均值、标准差归一化。

### 5.2 音频流

```text
[B, 32, 2, audio_T]
```

音频 latent 的技术特征：

- 32 个 latent 通道。
- 固定两个立体声声道。
- 音频采样率为 32,000 Hz。
- 每个音频 latent 时间单元对应 800 个 waveform 采样点。
- latent 时间频率固定为 40 Hz。
- 左右声道由同一个单声道编码器/解码器分别处理。
- latent 已经过逐通道均值、标准差归一化。

### 5.3 默认示例

目标参数：

```text
width  = 1344
height = 768
length = 124
FPS    = 24
```

得到：

```text
video_latent = [1, 24, 37, 48, 84]
audio_latent = [1, 32, 2, 207]
```

联合结构为：

```python
NestedTensor((
    torch.Tensor([1, 24, 37, 48, 84]),
    torch.Tensor([1, 32, 2, 207]),
))
```

### 5.4 Batch 限制

空 latent 的数据结构本身可以表达多个 batch，但当前 H3 DiT 在 forward 中明确检查：

```text
batch_size == 1
```

因此后续节点即使支持 latent batch 操作，也不能宣称最终 H3 采样支持批量大于 1。

## 6. 视频帧数和 Latent 时间长度

### 6.1 合法目标帧数

H3 原生节点会将请求帧数向上对齐，直到满足：

```text
frame_count % 17 == 5
```

即：

```text
frame_count = 17k + 5
```

合法序列包括：

```text
5, 22, 39, 56, 73, 90, 107, 124, 141, ...
```

目标空 latent 可以向上对齐，因为多出的帧尚不存在真实内容。

参考视频则只能向下裁剪到最近的合法长度，不能凭空补出参考内容。

### 6.2 视频 Latent T

计算方式为：

```python
if frame_count <= 5:
    video_T = 2
else:
    video_T = ((frame_count - 5) // 17) * 5 + 2
```

典型映射：

| 像素帧数 | 视频 latent T |
|---:|---:|
| 5 | 2 |
| 22 | 7 |
| 39 | 12 |
| 56 | 17 |
| 73 | 22 |
| 90 | 27 |
| 107 | 32 |
| 124 | 37 |

等价关系为：

```text
frame_count = 17k + 5
video_T     = 5k + 2
```

### 6.3 音频 Latent T

音频长度按照实际时长计算：

```python
duration_seconds = frame_count / frame_rate
audio_T = round(duration_seconds * 40)
```

原生 H3 使用 24 FPS，因此 124 帧时：

```text
duration = 124 / 24 = 5.166666... 秒
audio_T  = round(5.166666... * 40) = 207
```

音频解码后的理论采样点数为：

```text
207 * 800 = 165600 samples
```

对应约 5.175 秒。由于 `audio_T` 必须取整数，音频末端与视频理论时长可能存在一个 latent 单位以内的量化差异。

## 7. 视频 VAE 技术特征

视频 VAE 由两种不同结构组成：

- 编码器：3D causal CNN。
- 解码器：ViT3D Decoder。

主要配置：

```text
输入通道            3
latent 通道         24
空间压缩率          16
基础时间压缩率      4
编码分块长度        17 帧
token_drop          3
编码器 causal       是
解码器 causal       否
解码器 Transformer  36 层
```

### 7.1 视频 Latent 归一化

编码输出使用 posterior mean，不执行随机采样：

```python
mean = posterior_moments.chunk(2, dim=1)[0]
normalized = (mean - latents_mean) / latents_std
```

解码时执行反变换：

```python
latent = normalized * latents_std + latents_mean
```

因此 H3 节点中看到的 latent 已处于标准化空间。

### 7.2 单张图片编码

单帧输入会增加时间维，并在编码后只保留最后一个时间位置：

```text
[B, H, W, C]
    -> VAE
[B, 24, 1, H/16, W/16]
```

该路径主要用于首帧、尾帧和参考图片条件。

### 7.3 多帧编码

多帧输入会先补齐到 17 帧分块，再逐块编码，最后丢弃尾部 3 个 token。

因此视频 latent 的时间结构不是简单的逐帧等距下采样。对已经编码的 latent 进行拼接、重叠或裁剪时，必须考虑：

- 17 帧编码块边界。
- 全局 `token_drop=3`。
- latent 时间相位。
- 解码器对短 latent 的补齐规则。

## 8. 音频 VAE 技术特征

音频 VAE 使用：

- DAC 系列 waveform encoder。
- 因果注意力 posterior projection。
- BigVGAN decoder。
- Snake/SnakeBeta 周期激活。
- 抗混叠上采样和下采样。

编码器 stride：

```text
2 * 4 * 4 * 5 * 5 = 800
```

所以：

```text
32000 / 800 = 40 latent frames/second
```

输入 waveform 会在右侧补零到 800 采样点的整数倍。推理时直接使用 posterior mean，不进行随机采样。

## 9. 文本与多模态条件

H3 使用为该模型准备的 Qwen3-VL-32B 文本/视觉编码器。

本地权重元数据表明：

```text
读取第 50 层之后、最终归一化之前的隐藏状态
隐藏维度 = 5120
```

进入 H3 DiT 前会经过：

```text
5120 -> condition_proj -> 5376 -> 2 层 token refiner
```

Tokenizer 会构造带编号的展示序列：

```text
<Picture 1>:
<Audio 1>:
<Video 1>:
<0.5 seconds>
...
用户 prompt
```

同时为文本序列中的每个 token 保存模态标签：

```text
0 = 视觉
1 = 文本
2 = 音频
```

音频 waveform 不直接输入 Qwen3-VL。Qwen 只看到 `<Audio j>` 标签；真正的声音内容由音频 VAE latent 通过 DiT 参考块注入。

## 10. H3 DiT 结构

本地实现的主要参数：

```text
Transformer 层数       50
hidden size             5376
attention heads         56
head dimension          128
FFN hidden size         14336
视频 latent 通道        24
音频 latent 通道        32
视频 patch              1 x 2 x 2
文本输入维度            5120
```

### 10.1 视频 Patchify

视频 patch 大小为：

```text
1 x 2 x 2
```

所以每个视频 token 的原始输入维度是：

```text
24 * 1 * 2 * 2 = 96
```

默认 `1344x768` 画布对应视频 latent 空间：

```text
48 x 84
```

经过 `2x2` patch 后，每个 latent 时间位置的 token 数量为：

```text
24 x 42 = 1008
```

124 帧对应 37 个 latent 时间位置，因此目标视频部分包含：

```text
37 * 1008 = 37296 video tokens
```

### 10.2 音频 Pack

音频张量：

```text
[B, 32, 2, audio_T]
```

会按声道优先展开为：

```text
[2 * audio_T, 32]
```

124 帧、24 FPS 时：

```text
2 * 207 = 414 audio tokens
```

### 10.3 单流序列

T2VA/FL2VA 序列：

```text
[text | first/last-frame condition rows | target audio | target video]
```

REF2VA 序列：

```text
[text | reference image/video/audio blocks | target audio | target video]
```

目标音频和目标视频位于最后两个连续 segment。

参考条件行会在每一个采样步骤重新注入，但不会作为目标 latent 被正常去噪。

## 11. H3 时间坐标

H3 的三轴位置坐标为：

```text
(time, height, width)
```

### 11.1 视频时间跨度

每个视频 latent 时间单元的基础跨度循环为：

```text
1, 4, 4, 4, 4
```

原生 24 FPS 下，坐标换算系数为：

```text
FRAME_RESCALE = 5 / 3 = 40 / 24
```

所以实际时间跨度循环为：

```text
5/3, 20/3, 20/3, 20/3, 20/3
```

这将视频 token 坐标映射到与音频 40 Hz latent 相同的时间尺度。

### 11.2 音频时间坐标

音频 token 每个时间位置递增 1。左右声道共享相同时间坐标，但分别固定在空间宽度轴的两端，用于区分两个声道。

### 11.3 空间坐标

视频空间坐标不是简单的像素索引，而是按照 latent 画布面积归一化，使不同宽高比的目标和参考媒体能够进入统一的三轴 RoPE 坐标系统。

## 12. 视频和音频的双 Flow Shift

默认设置：

```text
video shift = 12.0
audio shift = 3.0
```

采样器最终只能处理一个扁平 Tensor 和一条 Sigma 时间线，因此 ComfyUI 使用 `ModelSamplingAV` 将音频流映射到视频采样时间线。

音频携带缩放：

```text
audio_scale = video_shift / audio_shift
            = 12 / 3
            = 4
```

整体过程：

```text
标准化视频 latent + 标准化音频 latent
                |
                | process_latent_in
                v
音频 latent 乘以 audio_scale
                |
                v
视频和音频扁平拼接，交给普通采样器
                |
                v
H3 forward 根据 video sigma 推导 audio sigma
                |
                v
还原音频自己的 latent 和 velocity 表达
                |
                v
DiT 联合推理
                |
                v
输出再次转换回统一采样器坐标
                |
                | process_latent_out
                v
恢复标准化音频 latent
```

因此自定义节点应操作采样前或采样后的 `NestedTensor`，不能把采样器内部已经扁平化、缩放过的音频区段当作标准音频 latent。

## 13. NestedTensor 和采样器打包

`NestedTensor` 只是一层轻量容器。它支持：

- 加、减、乘、除。
- `.to()`、`.cpu()`、`.float()`。
- 对每个内部 Tensor 应用相同索引。
- 返回第一个流的 `shape`、`size`、`device` 和 `dtype`。

采样前，ComfyUI 会将每个流变形为：

```text
[B, 1, flattened_length]
```

然后沿最后一维拼接，并保存原始形状：

```python
latent_shapes = [video_shape, audio_shape]
```

采样结束后根据 `latent_shapes` 切分并恢复成新的 `NestedTensor`。

这意味着：

- 普通采样器不需要理解视频和音频的原始维度。
- H3 模型必须通过 `latent_shapes` 才能正确拆分两个流。
- 自定义 latent 节点不能丢失两个流的顺序。
- 第一个流必须是视频，第二个流必须是音频。

## 14. 条件 Latent 与目标 Latent 的区别

目标 latent：

- 位于最终 `NestedTensor` 中。
- 由采样器加噪和去噪。
- 采样结束后送入 VAE 解码。

条件 latent：

- 存放于 `CONDITIONING` 中的 H3 payload。
- 图片和参考视频使用视频 VAE 编码。
- 参考音频使用音频 VAE 编码。
- 每个采样步骤重新注入。
- 不参与目标流的正常去噪更新。

视觉条件默认使用：

```text
visual_cond_noise_aug = 0.999
```

音频条件默认使用：

```text
audio_cond_noise_aug = 1.0
```

因此视觉条件会混入极少量确定性噪声；音频条件默认不混入条件噪声。

## 15. 模型任务类型

本地安装了两类主要 H3 DiT 权重。

### 15.1 FL2VA

同一套权重覆盖：

- 不连接图片：T2VA。
- 只连接首帧：首帧引导生成。
- 同时连接首帧和尾帧：FL2VA。

当前实现只接受第一帧和最后一帧作为关键帧锚点，不支持任意中间帧索引。

### 15.2 REF2VA

支持：

- 多张参考图片。
- 多个参考视频。
- 参考视频配套音轨。
- 独立参考音频。

参考图片、视频和音频同时进入语义条件通路和 latent 条件通路。

### 15.3 本地模型规模快照

通过本地 safetensors 张量头统计得到：

| 文件类型 | 近似参数量 |
|---|---:|
| 完整 H3 DiT | 331.23 亿 |
| Pruned H3 DiT | 201.11 亿 |
| 本地量化 Qwen3-VL 编码器文件 | 257.57 亿张量元素 |
| H3 视频 VAE | 26.04 亿 |
| H3 音频 VAE | 1.51 亿 |

该统计反映本地转换后权重文件中的张量元素数量，不等同于官方对模型规模的命名口径。

## 16. 通用 Latent 节点兼容性

### 16.1 相对安全的操作

以下操作在明确处理两个流、并保持 metadata 的前提下可以支持：

- 查看视频和音频形状。
- 分离 AV latent。
- 合并 AV latent。
- 替换视频流。
- 替换音频流。
- 对两个流分别执行数值缩放。
- 沿 batch 维选择，但最终采样仍必须是 batch 1。
- 复制 latent 字典并保留未知字段。

ComfyUI 已提供可参考的通用实现：

```text
comfy_extras/nodes_lt.py
    LTXVConcatAVLatent
    LTXVSeparateAVLatent
```

虽然类名带有 LTXV，但节点描述已经明确支持包括 MiniMax H3 在内的任意 AV 模型。

### 16.2 不能直接复用的普通节点

普通图像 latent 节点通常假设：

```text
[B, C, H, W]
```

而 H3 视频流是 5D，音频流又是另一套 4D 语义。因此以下操作必须单独实现或逐个审查：

- Latent Upscale。
- Latent Crop。
- Latent Rotate。
- Latent Flip。
- Latent Composite。
- Latent Blend。
- 时间切片和时间拼接。
- Repeat Batch。
- Noise Mask 设置和变换。

主要风险包括：

- `NestedTensor.shape` 只返回视频流形状。
- 同一个维度编号在视频和音频中含义不同。
- 对 NestedTensor 应用相同切片可能错误裁剪音频声道或时间轴。
- 普通上采样函数可能只接受 4D 图像 Tensor。
- latent 字典内的 `noise_mask` 可能仍是 NestedTensor，必须同步操作。

## 17. Latent 数值操作边界

视频和音频 latent 都处于标准化空间，因此数值 0 表示训练 latent 分布的均值附近，不严格表示：

- 黑色画面。
- 透明画面。
- 静止画面。
- 绝对静音。

所以：

- 若需要严格静音，应优先构造零 waveform 后通过音频 VAE 编码。
- 若需要严格固定画面，应优先构造目标像素帧后通过视频 VAE 编码。
- latent 线性插值属于标准化特征空间插值，其视觉或听觉含义不能按像素/波形线性关系理解。
- 用零 latent 填充待生成区域时，应配合 noise mask，让模型生成该区域，而不是直接把零 latent 当作最终内容解码。

## 18. 时间裁剪、拼接和变速的关键约束

### 18.1 视频和音频不能使用相同 T 索引

视频 latent 时间轴和音频 latent 时间轴的频率不同：

```text
视频：非均匀的 17 帧 -> 5 token 映射
音频：固定 40 token/秒
```

所有同步操作都应先转换为共同的时间单位，再分别计算视频和音频范围。

建议使用以下逻辑单位之一：

- 秒。
- 像素帧索引和 FPS。
- H3 内部 40 Hz 时间坐标。

### 18.2 视频 Raw Latent 拼接风险

直接执行：

```python
torch.cat([video_a, video_b], dim=2)
```

可能产生以下问题：

- 两段 latent 各自携带的 VAE 分块头尾语义被直接保留。
- 拼接点不一定符合原始 17 帧分块关系。
- DiT 的 `1,4,4,4,4` 时间跨度相位会按照拼接后的全局索引继续计算。
- 拼接后的 latent T 未必满足 `5k+2`。
- 无法仅根据拼接后的 T 无歧义恢复原始像素帧数。

因此时间拼接节点至少应提供：

- 严格模式：只接受可证明合法的输入和拼接方式。
- 实验模式：允许 raw latent 拼接，但明确标记为训练分布外操作。
- 像素域模式：先解码、拼接像素帧，再重新编码，成本更高但语义更清晰。

### 18.3 可调 FPS

当前扩展已有：

```text
minimax_h3_reference_fps.py
MINIMAX_H3_FPS_PRINCIPLE.md
MINIMAX_H3_REFERENCE_FPS.md
```

该实现将视频时间坐标换算系数从固定：

```text
40 / 24
```

改为：

```text
40 / frame_rate
```

并同步调整目标音频 latent 长度、参考视频时间坐标和 Qwen 时间戳。

后续 latent 时间操作节点不能写死 24 FPS。如果节点接收来自可调 FPS 工作流的 latent，应显式接收或携带帧率/时间轴元数据。

## 19. Noise Mask 约束

联合 AV latent 的 noise mask 也可以是：

```python
NestedTensor((video_noise_mask, audio_noise_mask))
```

处理规则：

- 替换视频流时同步替换或校正视频 mask。
- 替换音频流时同步替换或校正音频 mask。
- 某一流没有 mask、另一流有 mask 时，需要为缺失流创建全 1 mask。
- 音频补短时，补出的尾部应保持可去噪，使模型生成剩余音频。
- 音频裁短时，对应 mask 必须以相同范围裁剪。
- 不应无条件删除 latent 字典中的其他 metadata。

## 20. 后续节点设计建议

建议将 H3 latent 节点分成三层。

### 20.1 AV 容器层

负责结构而不改变具体内容：

- H3 AV Latent Inspect。
- H3 Separate AV Latent。
- H3 Combine AV Latent。
- H3 Replace Video Stream。
- H3 Replace Audio Stream。
- H3 Validate AV Latent。

验证内容应包括：

- 是否为 `NestedTensor`。
- 是否恰好包含两个流。
- 视频是否为 `[B,24,T,H,W]`。
- 音频是否为 `[B,32,2,T]`。
- 两个流 batch 是否一致。
- 是否满足 batch 1 采样要求。
- 视频空间尺寸是否适配 `2x2` DiT patch。
- noise mask 结构是否与 samples 对应。

### 20.2 单模态操作层

视频和音频分别处理：

- 视频 latent 数值混合。
- 视频空间裁剪和补边。
- 视频空间缩放。
- 音频 latent 裁剪和补齐。
- 音频左右声道处理。
- 视频或音频 latent 标准化统计检查。

这类节点不应假定另一流使用相同维度或相同长度。

### 20.3 时间同步层

负责视频、音频和 FPS 的共同变化：

- 按秒裁剪 AV latent。
- 按像素帧范围裁剪 AV latent。
- 延长或缩短目标音频流以适配视频时长。
- 计算合法的 `17k+5` 帧数。
- 计算 `video_T` 与 `audio_T`。
- 检查 H3 时间相位和训练分布边界。
- 为可调 FPS 工作流保存和读取时间轴信息。

## 21. 实现原则

后续代码建议遵守以下原则：

1. 不直接把 H3 AV latent 当普通 Tensor。
2. 不依赖 `samples.shape` 判断整个 AV 结构。
3. 始终显式拆分视频流和音频流。
4. 操作完成后重新构造 `NestedTensor`。
5. 保留输入 latent 字典中的未知 metadata。
6. 同步维护 `noise_mask`。
7. 对视频和音频分别验证 shape。
8. 所有时间操作都显式使用 FPS 或秒。
9. 不将视频时间压缩简化成固定 4 倍。
10. 不将零 latent 描述为严格黑帧或严格静音。
11. 区分目标 latent 与条件 latent。
12. 区分标准化 latent 和采样器内部缩放后的扁平 latent。
13. 默认拒绝 batch 大于 1 的 H3 采样输入。
14. 对 raw latent 时间拼接等实验能力提供明确警告。
15. 优先调用 ComfyUI 已有的 NestedTensor 和 AV 打包机制，不自行维护采样器私有格式。

## 22. 当前扩展已有相关实现

当前 `ComfyUI-CKNodes` 已包含：

| 文件 | 作用 |
|---|---|
| `minimax_h3_reference_fps.py` | MiniMax H3 Reference-to-Video 可调 FPS 节点 |
| `MINIMAX_H3_FPS_PRINCIPLE.md` | 可调 FPS 的时间轴原理 |
| `MINIMAX_H3_REFERENCE_FPS.md` | 可调 FPS Reference-to-Video 节点详细说明 |

其中 `minimax_h3_reference_fps.py` 已经通过自定义 PackedLayout 调度，使目标视频、目标音频、参考视频和 Qwen 时间戳共同使用选定 FPS。

新的 latent 操作节点需要兼容该机制，尤其不能假设所有 H3 latent 都来自原生 24 FPS 节点。

## 23. 核心结论

MiniMax H3 latent 操作的本质不是对一个视频 Tensor 做常规变换，而是维护一套联合多模态状态：

```text
视频 latent
+ 音频 latent
+ 两种时间频率
+ 非均匀视频帧映射
+ Noise Mask
+ FPS/时间轴语义
+ 采样器内部音频缩放
+ 条件 latent 与目标 latent 的区别
```

其中最容易出错的部分是：

1. 把 NestedTensor 当作普通视频 Tensor。
2. 把视频 latent T 当作均匀的四倍时间压缩。
3. 对视频和音频使用相同时间索引。
4. 在采样器扁平空间直接编辑音频段。
5. 时间裁剪或拼接后不处理 H3 的 `17k+5` 与 `5k+2` 约束。
6. 修改 samples 后遗漏对应的 noise mask。
7. 忽略可调 FPS 工作流已经改变了 H3 内部时间坐标。

后续节点应先建立统一的 H3 AV latent 校验和时间换算基础模块，再在其上实现裁剪、混合、拼接、替换和同步等具体能力。

## 24. 已实现的第一阶段 Latent 节点

第一阶段工具节点已实现在：

```text
minimax_h3_latent.py
```

包括：

- 联合 AV latent 拆分。
- 视频和音频 latent 合并。
- MiniMax H3 图片 VAE 编码。
- 按 video latent T 索引替换视频 latent。
- Latent 结构、形状、帧数和时长信息读取。
- 视频帧、视频 latent T、音频 latent T 和秒的双向换算。

使用方法及参数语义见：

```text
MINIMAX_H3_LATENT_NODES.md
```
