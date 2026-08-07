# CK MiniMax H3 Reference to Video（Adjustable FPS）工作原理

## 1. 文档目的

本文说明 `CK MiniMax H3 Reference to Video (Adjustable FPS)` 本地节点的职责、数据流、参考媒体处理方式、MiniMax H3 联合音视频采样机制，以及可调帧率时间映射的实现原理。

该节点位于：

```text
custom_nodes/ComfyUI-CKNodes/minimax_h3_reference_fps.py
```

节点 ID：

```text
CKMiniMaxH3ReferenceToVideoFPS
```

节点分类：

```text
CKNodes/minimax
```

该实现复制并扩展了 ComfyUI 本地 `MiniMax H3 Reference to Video` 的处理逻辑，不涉及 MiniMax 云端 API。

---

## 2. 节点定位

该节点不是采样器，也不直接输出最终视频。它是 MiniMax H3 的复合条件准备节点，负责：

1. 创建目标视频与目标音频的联合空 latent。
2. 预处理参考图片、参考视频和参考音频。
3. 使用视频 VAE 和音频 VAE 编码参考媒体。
4. 为 Qwen3-VL 构造包含图片、视频和音频标签的多模态展示序列。
5. 将参考 latent 及其空间、时间元数据写入 `CONDITIONING`。
6. 根据 `frame_rate` 自动调整目标音频长度、Qwen 时间戳和 H3 DiT 时间坐标。

节点输出：

```text
positive : CONDITIONING
latent   : LATENT
```

完整生成流程仍需连接采样器和解码节点：

```text
CK MiniMax H3 Reference to Video (Adjustable FPS)
                            │
                            ├── positive ──┐
                            │              │
                            └── latent ────┴──> Sampler
                                                │
                                                ├──> 视频 latent
                                                └──> 音频 latent
                                                        │
                                          Separate AV Latent
                                             │          │
                                      视频 VAE 解码  音频 VAE 解码
                                             │          │
                                             └──> 音视频合成/保存
```

---

## 3. 输入与输出

### 3.1 必需输入

| 输入 | 类型 | 说明 |
|---|---|---|
| `clip` | `CLIP` | MiniMax H3 使用的 Qwen3-VL 条件编码器 |
| `vae` | `VAE` | MiniMax H3 视频 VAE，用于编码参考图片和参考视频 |
| `audio_vae` | `VAE` | MiniMax H3 音频 VAE，用于编码参考音频 |
| `prompt` | `STRING` | 文本提示词，可通过 `<Picture i>`、`<Video k>`、`<Audio j>` 引用媒体 |
| `width` | `INT` | 目标视频宽度，UI 以 32 为步长，调用方应保证为 32 的倍数 |
| `height` | `INT` | 目标视频高度，UI 以 32 为步长，调用方应保证为 32 的倍数 |
| `length` | `INT` | 请求的目标视频帧数，节点会向上对齐到 H3 合法帧数 |
| `frame_rate` | `FLOAT` | H3 模型时间轴的帧率，默认 16 FPS |
| `ref_image_size` | `COMBO` | 参考图片尺寸策略：`match` 或 `max` |

### 3.2 可变参考输入

| 输入 | 最大数量 | 说明 |
|---|---:|---|
| `ref_images` | 9 | 参考图片 |
| `ref_videos` | 3 | 参考视频帧序列，输入类型为 `IMAGE` batch |
| `ref_video_audios` | 3 | 与同编号参考视频配对的音轨 |
| `ref_audios` | 3 | 独立参考音频 |

视频和视频音轨按输入名末尾编号配对：

```text
ref_video_0  <->  ref_video_audio_0
ref_video_1  <->  ref_video_audio_1
ref_video_2  <->  ref_video_audio_2
```

### 3.3 输出

| 输出 | 类型 | 内容 |
|---|---|---|
| `positive` | `CONDITIONING` | Qwen3-VL hidden states、模态标签和 H3 参考块 |
| `latent` | `LATENT` | 包含目标视频和目标音频的联合 `NestedTensor` |

---

## 4. 总体数据流

节点内部将参考媒体同时送入两个条件通路。

### 4.1 语义条件通路

```text
参考图片像素 / 参考视频抽样帧 / 媒体标签
                    │
                    v
               H3 Tokenizer
                    │
                    v
               Qwen3-VL-32B
                    │
                    v
       文本与视觉语义 hidden states
```

这条通路负责让模型理解：

- 参考图片中的人物、物体、服装、构图和风格。
- 参考视频中的场景、主体和大致时间变化。
- prompt 中的 `<Picture i>`、`<Video k>`、`<Audio j>` 分别指向哪个参考输入。

音频 waveform 不直接进入 Qwen3-VL。Qwen 只接收 `<Audio j>` 文本标签，真正的声音内容通过音频 latent 条件进入 DiT。

### 4.2 Latent 条件通路

```text
参考图片 / 完整参考视频 ──> 视频 VAE ──> 视频参考 latent
参考音频                ──> 音频 VAE ──> 音频参考 latent
                                                │
                                                v
                                      MiniMax H3 PackedLayout
                                                │
                                                v
                                  每个采样步骤参与联合 Attention
```

这条通路负责向 H3 DiT 提供高密度参考内容。参考 latent 在每个采样步骤中都会重新注入，但不会作为目标噪声被正常去噪。视觉参考默认会先执行强度为 `0.999` 的确定性条件噪声增强；增强结果在采样步骤间保持稳定，但不等于完全未经处理的原始 VAE latent。音频参考的默认增强强度为 `1.0`，即不混入条件噪声。

两条通路缺一不可：

- 只有 Qwen 条件时，模型知道参考媒体“表达什么”，但缺少高密度 latent 细节。
- 只有 VAE latent 时，模型拥有参考特征，但 prompt 无法可靠建立 `<Picture i>`、`<Video k>` 与参考内容的语义对应关系。

---

## 5. 目标联合 AV Latent

MiniMax H3 联合生成视频和立体声音频。节点首先根据 `width`、`height`、`length` 和 `frame_rate` 创建目标空 latent。

### 5.1 合法视频帧数

H3 视频帧数必须满足：

```text
frame_count = 17k + 5
```

合法帧数序列为：

```text
5, 22, 39, 56, 73, 90, 107, 124, 141, ...
```

节点会将用户请求的 `length` 向上对齐。例如：

```text
输入 length = 120
实际 frame_count = 124
```

与 `length` 不同，节点不会在执行阶段主动修正目标 `width` 和 `height`。标准 UI 控件以 32 为步长，但 API、旧工作流或动态输入仍须自行保证宽高为 32 的倍数。

### 5.2 视频 latent 时间长度

视频 latent 的时间长度计算为：

```text
frame_count <= 5:
    video_latent_t = 2

frame_count > 5:
    video_latent_t = ((frame_count - 5) / 17) * 5 + 2
```

124 个像素帧对应：

```text
video_latent_t = 37
```

### 5.3 音频 latent 时间长度

H3 音频 latent 时间频率固定为 40 Hz：

```text
duration_seconds = frame_count / frame_rate
audio_latent_t = round(duration_seconds * 40)
```

124 帧时：

| 时间轴帧率 | 视频时长 | 音频 latent T |
|---:|---:|---:|
| 24 FPS | 5.1667 秒 | 207 |
| 16 FPS | 7.7500 秒 | 310 |

### 5.4 联合 latent 结构

视频 latent：

```text
[B, 24, video_T, H/16, W/16]
```

音频 latent：

```text
[B, 32, 2, audio_T]
```

其中音频维度中的 `2` 表示立体声。

两者包装为：

```python
{"samples": NestedTensor((video_latent, audio_latent))}
```

默认 1344×768、124 帧、16 FPS 时：

```text
video = [1, 24, 37, 48, 84]
audio = [1, 32, 2, 310]
```

---

## 6. 参考图片处理

每个参考图片插槽执行以下步骤。

### 6.1 Batch 选择

节点只使用每个参考图片输入的第一张：

```python
image[:1]
```

如果要引用多张图片，应分别连接到多个 `ref_image` 插槽，而不是把多张图片组成一个 IMAGE batch 后连接到单个插槽。

### 6.2 `match` 尺寸策略

`match` 保持参考图宽高比，并在必要时将其像素面积限制到目标画面面积：

```text
scale = min(1, sqrt(target_area / reference_area))
```

特点：

- 只缩小，不放大。
- 不强行拉伸到目标宽高比。
- 降低参考 token 数和采样成本。
- 适合大多数参考图工作流。

### 6.3 `max` 尺寸策略

`max` 将参考图片短边限制到 2048 像素：

```text
scale = min(1, 2048 / short_edge)
```

特点：

- 保留更多身份、纹理和局部细节。
- 参考 token 会经过每一层、每一个采样步骤。
- 显存占用和采样时间可能显著增加。

### 6.4 双通路输出

缩放后的图片同时产生：

```python
# Qwen3-VL 展示数据
{
    "type": "image",
    "data": resized_image,
}

# H3 DiT 参考块
{
    "kind": "image",
    "latent_h": latent_height,
    "latent_w": latent_width,
    "latent": video_vae_latent,
}
```

---

## 7. 参考视频处理

参考视频以 IMAGE 帧序列输入：

```text
[T, H, W, C]
```

节点按照以下顺序处理。

### 7.1 参考画布适配

参考视频画布与目标输出尺寸相互独立。节点执行：

1. 保持参考视频宽高比。
2. 以短边 768 像素为基础尺寸。
3. 将总面积限制在 `768 * 1344` 以内。
4. 将宽高对齐到 32 的倍数。
5. 当源视频比计算画布更小时避免不必要的放大。

这使参考视频能够保留自己的构图，而不是被强制拉伸到目标视频宽高比。

### 7.2 帧数限制和对齐

参考视频首先被限制到不超过目标 `frame_count`：

```text
reference_frames <= target_frame_count
```

随后向下裁剪到最近的 `17k+5`：

```text
120 帧 -> 107 帧
124 帧 -> 124 帧
130 帧 -> 124 帧
```

目标空 latent 可以向上对齐，因为它没有真实内容；参考视频只能向下裁剪，不能凭空补出参考帧。

参考视频至少需要 5 帧。

### 7.3 完整视频 VAE 编码

对齐后的完整帧序列通过视频 VAE 编码：

```python
encoded = vae.encode(frames)
```

这部分 latent 主要承载：

- 运动模式。
- 时序结构。
- 场景变化。
- 高密度视觉细节。

### 7.4 Qwen 视觉抽样

完整 16/24 FPS 帧序列不直接全部送入 Qwen3-VL。节点按约 2 FPS 抽样：

```text
sample_step = round(frame_rate / 2)
```

示例：

| `frame_rate` | 抽帧步长 | Qwen 采样频率 |
|---:|---:|---:|
| 24 | 12 帧 | 约 2 FPS |
| 16 | 8 帧 | 约 2 FPS |

每个抽样帧的时间戳根据原始帧索引计算：

```text
timestamp = frame_index / frame_rate
```

因此 16 FPS 下的索引 `0, 8, 16, 24` 对应：

```text
0.0s, 0.5s, 1.0s, 1.5s
```

### 7.5 视频与音轨绑定

如果同编号的 `ref_video_audio` 已连接，节点会将视频 latent 与音频 latent 组成一个关联参考块：

```python
{
    "kind": "video_audio",
    "latent_t": video_latent_t,
    "latent_h": video_latent_h,
    "latent_w": video_latent_w,
    "ref_audio_t": audio_latent_t,
    "latent": video_latent,
    "audio_latent": audio_latent,
    "ck_frame_rescale": 40 / frame_rate,
}
```

没有配套音轨时，类型为 `video`。

---

## 8. 参考音频处理

参考音频输入结构为：

```text
waveform    = [B, channels, samples]
sample_rate = 输入采样率
```

处理过程：

1. 读取 H3 Audio VAE 的目标采样率，默认 32000 Hz。
2. 输入采样率不同时使用 `torchaudio` 重采样。
3. 只使用音频 batch 的第一项。
4. 将波形从 `[B, C, L]` 调整为 `[B, L, C]`。
5. 调用 `audio_vae.encode()`。

H3 PackedLayout 按立体声分配音频行，因此参考音频应为双声道。双声道输入的音频 latent 结构为：

```text
[1, 32, 2, T]
```

节点不会自动把单声道复制为双声道。若输入单声道，Audio VAE 可能产生 `[1, 32, 1, T]`，与 PackedLayout 按双声道分配的行数不一致并导致采样失败。单声道参考音频应在进入本节点前转换为双声道。

需要注意：采样率转换不是时间拉伸。它负责满足 Audio VAE 的输入采样率，不会自动把参考音频变为慢动作。

独立参考音频最终形成：

```python
{
    "kind": "audio",
    "ref_audio_t": audio_latent_t,
    "audio_latent": audio_latent,
}
```

---

## 9. Qwen3-VL 展示序列

节点维护 `reference_items`，用于构造 Qwen3-VL 能理解的多模态序列。

引用顺序固定为：

```text
所有参考图片
-> 所有参考视频及其配套音轨标签
-> 所有独立参考音频
-> 用户 prompt
```

每种类型独立编号，编号从 1 开始。

例如输入：

- 两张参考图片。
- 一个带音轨参考视频。
- 一个独立参考音频。

展示序列大致为：

```text
<Picture 1>: [图片视觉块]
<Picture 2>: [图片视觉块]

<Audio 1>:
<Video 1>:
    <0.2 seconds> [两帧视频视觉块]
    <1.2 seconds> [两帧视频视觉块]
    ...

<Audio 2>:

[用户 prompt]
```

视频视觉块每两张抽样帧组成一个 temporal patch，标签时间取两张帧时间戳的平均值。

Prompt 应使用同样的标签引用媒体，例如：

```text
Use the character identity and clothing from <Picture 1>.
Follow the camera movement and pacing of <Video 1>.
Use the voice characteristics from <Audio 1>.
```

Qwen3-VL 输出第 50 层的 hidden states，并附带模态标签：

```text
0 = 视觉/视频 token
1 = 文本 token
2 = 音频 latent token（由 H3 PackedLayout 分配）
```

---

## 10. H3 Reference Blocks

节点维护另一套结构 `reference_blocks`，用于 H3 DiT 的 latent 条件。

### 10.1 图片参考块

```python
{
    "kind": "image",
    "latent_h": H,
    "latent_w": W,
    "latent": image_latent,
}
```

### 10.2 视频参考块

```python
{
    "kind": "video",
    "latent_t": T,
    "latent_h": H,
    "latent_w": W,
    "ref_audio_t": 0,
    "latent": video_latent,
    "audio_latent": None,
    "ck_frame_rescale": 40 / frame_rate,
}
```

### 10.3 带音轨的视频参考块

```python
{
    "kind": "video_audio",
    "latent_t": video_T,
    "latent_h": H,
    "latent_w": W,
    "ref_audio_t": audio_T,
    "latent": video_latent,
    "audio_latent": audio_latent,
    "ck_frame_rescale": 40 / frame_rate,
}
```

### 10.4 独立音频参考块

```python
{
    "kind": "audio",
    "ref_audio_t": T,
    "audio_latent": audio_latent,
}
```

这些结构通过以下字段写入 `CONDITIONING`：

```text
minimax_refs
```

采样准备阶段，ComfyUI 的 `MiniMaxH3.extra_conds()` 会汇总 Qwen 条件和节点写入的参考块，并构造统一 payload。数据来源分别是：

```text
minimax_refs                         -> refs
minimax_refs 中的视频/图片 latent    -> cond_video_latents
minimax_refs 中的音频 latent         -> cond_audio_latents
Qwen 编码器产生的 minimax_token_tags -> text_token_tags
上述条件和目标 latent shape          -> layout
```

`text_token_tags` 不由 `minimax_refs` 生成。它来自 Qwen tokenizer/编码器对文本和视觉 token 位置的标记，这正是语义条件通路与 latent 条件通路保持独立的体现。

---

## 11. Packed Sequence 与联合采样

H3 不是传统的“文本 cross-attention + 单一视频 latent”模型。它将不同模态打包进一条序列：

```text
[Qwen 文本/视觉 tokens
 | 参考图片/视频 latent rows
 | 参考音频 latent rows
 | 目标音频 latent rows
 | 目标视频 latent rows]
```

顺序细节：

- 图片参考写入 `ref_img` 段。
- 独立音频写入 `ref_audio` 段。
- 带音轨视频按 `ref_audio -> ref_img` 顺序排列。
- 目标音频和目标视频始终是最后两个段。

### 11.1 参考块不作为目标去噪

布局为所有视频和音频行构造 update mask：

```text
参考图片/视频 latent : update = false
参考音频 latent      : update = false
目标视频 latent      : update = true
目标音频 latent      : update = true
```

每一个采样步骤中：

1. 当前目标噪声 latent 被放入目标位置。
2. 稳定的参考条件 latent 被重新放入参考位置；视觉参考默认包含确定性 `0.999` 条件噪声增强。
3. 所有模态共同经过单流 Transformer attention。
4. 最终层只输出目标视频和目标音频的速度预测。

因此，参考 latent 会持续影响每一层、每一步，但不会被当作输出目标逐步去噪。

这也说明了高分辨率参考图为什么会显著增加成本：参考 token 不只编码一次，而是贯穿整个采样过程。

---

## 12. 可调 FPS 时间映射

### 12.1 三个容易混淆的概念

必须区分：

1. `length`：实际生成多少张视频帧。
2. `frame_rate`：H3 模型内部如何解释这些帧覆盖的时间。
3. 最终保存节点 FPS：视频容器按多快播放这些帧。

CK 节点的 `frame_rate` 不会自动插帧或删帧。例如：

```text
length = 124
frame_rate = 16
```

仍然生成 124 帧，只是模型将其解释为：

```text
124 / 16 = 7.75 秒
```

如果最终保存节点仍设置为 24 FPS，容器会在约 5.17 秒内播放完 124 帧。因此最终视频保存/合成节点也必须设为相同 FPS。

### 12.2 统一时间单位

H3 音频 latent 固定为 40 Hz，因此内部可以使用：

```text
1 秒 = 40 个时间单位
```

每个像素帧对应的时间缩放为：

```text
frame_rescale = 40 / frame_rate
```

24 FPS：

```text
frame_rescale = 40 / 24 = 1.6666667
```

16 FPS：

```text
frame_rescale = 40 / 16 = 2.5
```

16 FPS 下，相邻帧在模型时间轴上的间距是 24 FPS 的 1.5 倍：

```text
2.5 / 1.6666667 = 1.5
```

### 12.3 H3 视频时间压缩模式

H3 视频 latent 的时间跨度按五个 latent token 一组计算：

```text
FRAME_PER_TOKEN = [1, 4, 4, 4, 4]
```

每组对应 17 个像素帧：

```text
1 + 4 + 4 + 4 + 4 = 17
```

每个 latent token 的时间跨度为：

```text
token_span = FRAME_PER_TOKEN[i % 5] * frame_rescale
```

因此改变 FPS 时，不能只增加音频 latent 长度，还必须同步扩大视频 token 的时间坐标。

### 12.4 24 FPS 与 16 FPS 数值对比

124 帧对应 37 个视频 latent 时间位置。

| 项目 | 24 FPS | 16 FPS |
|---|---:|---:|
| 像素帧数 | 124 | 124 |
| 视频 latent T | 37 | 37 |
| 视频时长 | 5.1667 秒 | 7.7500 秒 |
| 音频 latent T | 207 | 310 |
| 视频 token 覆盖总时长（40 Hz 单位） | 206.6667 | 310.0000 |

16 FPS 下：

```text
目标视频 token 覆盖总时长 = 310
目标音频时间长度 = 310
```

视频和音频在同一 40 Hz 时间轴上的覆盖时长保持一致。这里的 `310` 是所有视频 token span 的总和，不是最后一个视频 token 的起始 position ID。`_video_t_grid` 使用 exclusive cumulative sum；16 FPS、37 个视频 token 时，最后一个 token 从时间坐标 300 开始并覆盖到概念结束位置 310，而音频 position ID 为 0 至 309。

### 12.5 为什么不能只修改 `FPS = 16`

如果只将空 latent 的时长计算改为 16 FPS：

```text
音频时间长度 = 310
视频时间坐标仍按 24 FPS = 206.67
```

将导致：

- 音频和视频的时间范围不一致。
- 参考音轨与参考视频错位。
- 参考视频之后的其他参考块及目标块，其起始游标会与新的参考视频时间解释不一致。
- 模型可能生成节奏异常或后半段无对应画面的音频。

CK 节点因此同时调整：

1. 目标音频 latent 长度。
2. 目标视频时间位置。
3. 参考视频时间位置。
4. 带音轨参考块的结束游标。
5. Qwen 抽帧间隔和时间戳。

---

## 13. CK PackedLayout 扩展方式

官方 H3 `PackedLayout` 使用固定 24 FPS 时间缩放。为了不修改 ComfyUI 核心文件，CK 节点采用条件分派。

### 13.1 专用引用容器

CK 节点使用 `CKMiniMaxH3ReferenceBlocks` 保存：

```text
ck_frame_rate
ck_frame_rescale = 40 / frame_rate
```

它仍然是 `list` 的子类，因此官方 `MiniMaxH3.extra_conds()` 可以按普通引用列表处理。

### 13.2 布局分派

模块加载时会为 `comfy.ldm.minimax.model.PackedLayout` 安装一个薄分派器：

```text
refs 带有 ck_frame_rescale
    -> CKMiniMaxH3PackedLayout

普通官方 refs
    -> 官方 PackedLayout
```

因此：

- CK 可调 FPS 节点使用动态时间坐标。
- 官方 MiniMax H3 节点继续使用原生 24 FPS。
- 不需要修改 `comfy/model_base.py`。
- 不需要修改 `comfy/ldm/minimax/model.py`。
- 不会改变其他模型和普通 conditioning 的行为。

### 13.3 自定义布局调整范围

CK 布局参数化以下位置：

- 目标视频 `_video_t_grid`。
- 参考视频 `_video_t_grid`。
- 参考视频结束后的 `cursor`。
- 首尾关键帧时间坐标兼容逻辑。
- 目标音频与目标视频的统一时间范围。

空间坐标、音频网格、segment 类型和 update mask 仍保持官方 H3 规则。

### 13.4 运行时分派的维护边界

该分派器会在 CK 节点模块导入时执行进程级替换：

```python
minimax_model.PackedLayout = packed_layout
```

它不会改写 ComfyUI 核心文件，但属于运行时全局包装，而不是只在单个节点函数作用域内生效。正常情况下，普通官方 refs 会继续转发给模块加载时保存的官方 `PackedLayout`；CK refs 才进入自定义布局。

维护时需要注意：

- CK 布局复制并依赖官方 `PackedLayout` 的构造参数、segment 顺序、cursor 规则、update mask 和公开属性。
- ComfyUI 升级若修改这些内部契约，本扩展可能需要同步更新。
- 其他 custom node 若也包装 `minimax_model.PackedLayout`，最终包装链与加载顺序有关。
- 删除或禁用本节点文件后，需要重启 ComfyUI 才能恢复干净的进程状态。
- `_ck_fps_dispatch` 只防止本分派器重复安装，不是通用的第三方 monkey patch 冲突解决机制。

---

## 14. 16 FPS 是否保证慢动作

需要分别看“时间定义”和“画面语义”。

### 14.1 时间定义层面

在相同帧数下，16 FPS 必然比 24 FPS 覆盖更长时间：

```text
playback_speed = 16 / 24 = 0.6667
duration_scale = 24 / 16 = 1.5
```

只要最终保存节点同样设为 16 FPS，124 帧就必然播放 7.75 秒，而不是 5.17 秒。

### 14.2 模型生成语义层面

CK 节点还会告诉 H3：这些帧在内部时间轴上覆盖 7.75 秒。模型通常会倾向于将动作铺展到更长时间，但无法数学保证所有动作严格变成 2/3 倍速。

模型仍可能：

- 提前完成动作并在后续保持姿态。
- 在增加的时间中生成额外动作。
- 优先遵从 prompt 中的快速动作描述。
- 在超出训练时长分布时出现质量下降。

如果要求逐帧内容完全不变、绝对保证 2/3 倍速，最确定的方法仍是：

```text
按原生 24 FPS 生成
-> 保持全部帧不变
-> 以 16 FPS 封装
-> 对音频做 1.5 倍时长的保音高 time-stretch
```

CK 节点实现的是“按 16 FPS 构造的一致 AV 模型时间轴”，不是生成后的简单容器重解释，也不是模型权重或官方接口原生提供的 16 FPS 模式。

---

## 15. 推荐工作流

### 15.1 动态 16 FPS 时间映射的联合音视频生成

```text
H3 模型加载器
    ├── MODEL -> H3 Sigma Shift -> Sampler
    ├── CLIP  -> CK H3 Reference FPS
    ├── 视频 VAE -> CK H3 Reference FPS / 视频解码
    └── 音频 VAE -> CK H3 Reference FPS / 音频解码

CK H3 Reference FPS
    ├── positive -> Sampler
    └── latent   -> Sampler

Sampler
    -> Separate AV Latent
        ├── 视频 VAE Decode
        └── 音频 VAE Decode
    -> 视频合成节点，FPS = 16
```

关键设置：

```text
CK 节点 frame_rate = 16
最终视频保存/合成 FPS = 16
```

上述流程只展示 H3 特有的数据路径，不是采样器全部输入的完整接线图。实际 sampler 还需要其常规输入，例如 MODEL、negative conditioning、noise/seed、steps、sigmas 或 scheduler 等。`MiniMax H3 Sigma Shift` 应连接在 MODEL 到 sampler 的路径上；本节点只输出 positive conditioning，不生成 negative conditioning。

### 15.2 保持官方原生 24 FPS

将 CK 节点设置为：

```text
frame_rate = 24
```

并将最终视频保存节点设置为 24 FPS。此时目标音频长度和视频时间坐标与官方 H3 原生时间轴一致。

---

## 16. 与官方 VAE 节点及 Latent 节点的关系

### 16.1 图片和视频 VAE Encode

官方 `VAE Encode` 可以直接编码 H3 图片或视频帧序列。CK Reference 节点内部保留了 VAE 编码，是因为它还需要同步维护：

- Qwen 使用的像素媒体。
- H3 使用的 VAE latent。
- 参考类型和编号。
- 空间尺寸。
- 视频时间长度。
- 视频和音轨绑定关系。
- FPS 时间缩放。

若未来继续模块化，可以将媒体预处理和 VAE Encode 拆出，但仍需一个 H3 Reference Pack 节点重新绑定这些信息。

### 16.2 AV Latent 拆分和合并

ComfyUI 已有：

```text
Separate AV Latent
Concat AV Latent
```

它们支持 MiniMax H3，可在采样前后拆装视频和音频 latent。

### 16.3 普通 latent 替换不等于 Reference

`Replace Video Latent Frames` 可以替换目标视频 latent 的时间切片，但这不等价于 H3 Reference：

```text
普通 latent 替换：替换内容属于目标流，默认参与去噪。
H3 Reference：参考内容属于固定条件流，每一步重新注入，不作为输出目标去噪。
```

因此不能用普通 latent 替换完全复刻 H3 的图片、视频或音频参考机制。

---

## 17. 使用限制与注意事项

### 17.1 最终保存 FPS 必须匹配

CK 节点不会控制最终视频容器。`frame_rate=16` 时，视频合成或保存节点也必须设为 16 FPS。

### 17.2 参考视频按 `frame_rate` 解释

当前节点使用同一个 `frame_rate` 解释目标视频和所有参考视频。如果输入参考视频实际为 24 FPS，而节点设为 16 FPS，该参考会被解释为 1.5 倍时长。

这适合主动制造慢时间轴。如果未来需要“目标 16 FPS、参考保持真实 24 FPS”，应增加独立的 `reference_frame_rate` 输入。

### 17.3 配套参考音频不会自动 time-stretch

当参考视频从 24 FPS 改按 16 FPS 解释时，参考视频时长增加，但已连接的配套音频 waveform 不会自动做保音高时间拉伸。

节点只执行采样率适配。因此需要严格音画同步时，应在输入节点之前对参考音频完成对应的 time-stretch。

### 17.4 帧数不会因 FPS 自动改变

`frame_rate` 改变时间解释，不改变 `length`。要保持固定秒数，应自行换算目标帧数：

```text
length ~= duration_seconds * frame_rate
```

然后再对齐到最近的 `17k+5`。

### 17.5 参考图片成本

`ref_image_size=max` 可能产生数倍于 `match` 的参考 token。参考 token 贯穿每个采样步骤，可能显著增加显存和时间成本。

### 17.6 Batch 限制

- 每个参考图片插槽只使用 IMAGE batch 第一张。
- 每个参考音频插槽只使用 AUDIO batch 第一项。
- 参考音频必须为双声道；节点不会自动把单声道复制为双声道。
- H3 模型当前只支持目标 batch size 1。

### 17.7 非原生 FPS 属于训练分布外推

官方 H3 时间映射基于 24 FPS。16 FPS 动态时间坐标在数学上保持音视频一致，但属于对模型时间位置的扩展使用。应对动作稳定性、长时一致性和音画质量进行实际测试。

---

## 18. 故障排查

### 18.1 输出仍然像 24 FPS

检查：

```text
CK 节点 frame_rate 是否为 16
最终视频合成/保存节点 FPS 是否也为 16
工作流是否误用了官方同名节点
```

正确节点名称为：

```text
CK MiniMax H3 Reference to Video (Adjustable FPS)
```

### 18.2 音频比视频短或参考音画错位

检查配套参考音频是否提前做了与视频相同倍率的时间拉伸。节点只重采样到 Audio VAE 采样率，不改变音频语义时长。

### 18.3 参考视频被意外裁短

参考视频会：

1. 裁到不超过目标帧数。
2. 向下裁到最近的 `17k+5`。

例如 120 帧会变成 107 帧，这是 H3 视频 VAE 时间结构要求，不是随机丢帧。

### 18.4 改变 FPS 后动作没有严格减速

动态 FPS 调整的是模型时间坐标和输出时长，不是对已生成帧做确定性的逐帧重定时。需要绝对慢动作时，应使用生成后降低封装 FPS 的方案。

### 18.5 节点未出现

确认文件存在：

```text
custom_nodes/ComfyUI-CKNodes/minimax_h3_reference_fps.py
```

然后完整重启 ComfyUI。CKNodes 的 `__init__.py` 会自动扫描目录中的 Python 文件并合并 `NODE_CLASS_MAPPINGS`。

---

## 19. 代码对应关系

### CK 扩展

```text
custom_nodes/ComfyUI-CKNodes/minimax_h3_reference_fps.py
```

主要职责：

```text
媒体预处理
目标 AV latent 创建
动态 FPS 时间映射
Qwen presentation 构造
H3 reference blocks 构造
CK PackedLayout 分派
```

### ComfyUI 官方本地 H3 节点

```text
comfy_extras/nodes_minimax_h3.py
```

主要职责：

```text
官方 24 FPS Reference/Image-to-Video/Empty Latent 节点
```

### H3 文本和视觉编码器

```text
comfy/text_encoders/minimax.py
```

主要职责：

```text
<Picture>/<Video>/<Audio> presentation
视频 2 FPS temporal vision blocks
Qwen3-VL hidden states
模态 token tags
```

### H3 条件整理

```text
comfy/model_base.py
```

`MiniMaxH3.extra_conds()` 负责将 conditioning 元数据整理为模型 payload，并预构建 packed layout。

### H3 DiT

```text
comfy/ldm/minimax/model.py
```

主要职责：

```text
视频和音频 patch/pack
PackedLayout
时间与空间位置坐标
联合音视频单流 Transformer
参考条件固定注入
视频和音频双 schedule
目标视频和目标音频速度输出
```

### H3 视频和音频 VAE

```text
comfy/ldm/minimax/vae.py
comfy/ldm/minimax/audio_vae.py
```

---

## 20. 核心结论

`CK MiniMax H3 Reference to Video (Adjustable FPS)` 的本质不是简单修改一个 FPS 标签，而是为 H3 联合音视频生成建立一条与所选帧率一致的完整时间轴。

当 `frame_rate` 从 24 改为 16 时，节点保持视频帧数和视频 latent 数量不变，但会同步完成：

```text
目标时长扩大 1.5 倍
目标音频 latent 长度扩大 1.5 倍
目标视频时间坐标扩大 1.5 倍
参考视频时间坐标扩大 1.5 倍
Qwen 视频时间戳按 16 FPS 重算
参考块和目标块的时间游标保持一致
```

最终保存节点使用相同的 16 FPS 后，视频在播放层面必然具有 1.5 倍时长；模型内部也会按动态构造的 16 FPS 时间轴联合生成视频和音频。该模式并非官方权重原生提供的 16 FPS 能力。画面动作是否严格按固定倍率减速仍取决于模型生成行为，如需绝对确定的逐帧慢动作，应使用生成后重定时方案。
