# CK MiniMax H3 Latent 节点说明

## 1. 节点列表

节点分类：

```text
CK Nodes/MiniMax H3/Latent
```

本模块提供以下节点：

| 节点 | 用途 |
|---|---|
| `CK MiniMax H3 Separate AV Latent` | 将联合 AV latent 分成视频和音频两个 latent |
| `CK MiniMax H3 Combine AV Latent` | 将视频和音频 latent 重新组合成联合 AV latent |
| `CK MiniMax H3 Latent Resize` | 按目标分辨率或倍数缩放视频 latent，并自动对齐 H3 合法空间尺寸 |
| `CK MiniMax H3 Image VAE Encode` | 使用 H3 视频 VAE 将图片编码成 `[1,24,1,H/16,W/16]` latent |
| `CK MiniMax H3 Replace Video Latent By Index` | 按视频 latent 的时间索引替换一段 latent |
| `CK MiniMax H3 Latent Info` | 查看 latent 类型、形状、帧数、时长与结构合法性 |
| `CK MiniMax H3 Frame/Latent Convert` | 在视频帧、视频 latent T、音频 latent T 和秒之间换算 |

实现文件：

```text
minimax_h3_latent.py
```

## 2. AV Latent 拆分

输入：

```text
av_latent: LATENT
```

输入必须包含：

```python
NestedTensor((video_latent, audio_latent))
```

输出：

```text
video_latent
audio_latent
```

节点会同时拆分联合 `noise_mask`。输入 latent 字典中的其他 metadata 会保留到两个输出中。

节点会验证：

- 视频是否为 `[B,24,T,H,W]`。
- 音频是否为 `[B,32,2,T]`。
- 视频和音频 batch 是否一致。

## 3. AV Latent 合并

输入：

```text
video_latent
audio_latent
```

输出：

```python
{
    "samples": NestedTensor((video, audio))
}
```

如果输入带有 noise mask，节点会将其组合成：

```python
NestedTensor((video_noise_mask, audio_noise_mask))
```

只有一个输入带 mask 时，另一个流会创建全 1 mask，表示该流正常参与去噪。

合并节点不会擅自裁剪或补齐音频长度。需要先通过信息或换算节点确认视频和音频时长是否匹配。

## 4. Latent 尺寸缩放

该节点接受纯视频 latent 或联合 AV latent，只改变视频流的空间尺寸：

```text
[B,24,T,H,W] -> [B,24,T,new_H,new_W]
```

时间长度 `T`、batch 和通道数保持不变。联合 AV 输入中的音频 latent 原样保留；视频 `noise_mask` 使用最近邻方式同步缩放，音频 mask 原样保留。

### 尺寸模式

| 模式 | 行为 |
|---|---|
| `target_resolution` | 使用 `target_width` 和 `target_height` 作为目标像素分辨率 |
| `scale_by` | 根据输入像素尺寸乘以 `scale_by` 计算目标分辨率 |

H3 视频 VAE 的空间压缩率为 16，DiT 使用 `2x2` latent patch，所以最终像素宽高必须是 32 的倍数。节点会根据 `align_mode` 自动修正：

| 对齐方式 | 行为 |
|---|---|
| `nearest` | 对齐到最近的 32 倍数，默认模式 |
| `down` | 向下对齐，最低为 32 |
| `up` | 向上对齐 |
| `exact` | 不自动修正，输入不合法时直接报错 |

例如输入 latent 对应 `96x64` 像素，倍数为 `1.5`：

```text
原始计算: 144x96
合法输出: 160x96
latent:   10x6
```

节点额外输出实际像素宽高和横纵实际倍数，便于检查对齐后是否产生宽高比偏差。

`crop=disabled` 会直接拉伸到目标尺寸；`crop=center` 会先按目标宽高比居中裁剪，再缩放。

## 5. 图片 VAE 编码

该节点必须连接 MiniMax H3 视频 VAE。

输入：

| 输入 | 说明 |
|---|---|
| `image` | ComfyUI IMAGE，格式 `[B,H,W,C]` |
| `vae` | MiniMax H3 视频 VAE |
| `batch_index` | 从图片 batch 中选择一张；H3 采样要求 batch 1 |
| `canvas_mode` | 编码前的画布处理方式 |

画布模式：

### `center_crop_to_32`

默认模式。以图片中心为基准，将宽高向下裁到 32 的倍数，不执行拉伸。

### `resize_to_nearest_32`

将宽高分别缩放到最近的 32 倍数。可能轻微改变宽高比。

### `keep`

不主动改变输入图片尺寸，由 VAE 执行自身的尺寸裁剪。若编码后的 latent 高宽不能被 DiT 的 `2x2` patch 整除，节点会拒绝输出。

标准输出形状：

```text
[1, 24, 1, encoded_height/16, encoded_width/16]
```

图片编码得到的 `T=1` latent 适合：

- 作为替换节点的一帧 replacement latent。
- 作为其他自定义条件流程的图片 latent。

它不是合法的 H3 空目标生成序列；目标序列的 T 通常满足 `5k+2`。

## 6. 按时间索引替换视频 Latent

输入：

| 输入 | 说明 |
|---|---|
| `target_latent` | 纯视频 latent 或联合 AV latent |
| `replacement_latent` | 纯视频 latent 或联合 AV latent，联合输入只读取其视频流 |
| `time_index` | 视频 latent 的 T 索引，从 0 开始 |
| `overflow_mode` | 替换内容越界时的处理方式 |
| `noise_mask_mode` | 被替换区段的 noise mask 处理方式 |

注意：

```text
time_index 是视频 latent 索引，不是像素视频帧索引。
```

例如默认 124 帧目标视频的 video latent T 为 37，其合法索引范围为：

```text
0 ... 36
```

替换节点要求目标与 replacement 的以下维度一致：

```text
batch
channels
latent height
latent width
```

replacement 的 T 可以不同。

如果目标是联合 AV latent，节点只替换视频流，音频流保持不变。

### 越界模式

`error`：

- replacement 超出目标尾部时报错。
- 适合严格工作流。

`trim`：

- 自动截断超出目标尾部的 replacement。
- 节点会输出实际替换长度和结束索引。

### Noise Mask 模式

`preserve_target`：

- 保留目标原有 mask。
- 不创建新 mask。

`use_replacement_if_present`：

- replacement 有 mask 时，将对应部分复制到目标。
- replacement 没有 mask 时，保留目标对应区域。

`freeze_replaced`：

- 将替换区段 mask 设为 0。
- 采样时尽量保持替换 latent。

`denoise_replaced`：

- 将替换区段 mask 设为 1。
- 允许模型正常去噪替换区域。

## 7. Latent 信息获取

输入 latent 可以是：

- 联合 AV latent。
- 纯视频 latent。
- 纯音频 latent。

输出包括：

```text
info
is_av
is_valid_h3
batch_size
width
height
video_frames
video_latent_t
audio_latent_t
duration_seconds
```

检查内容：

- 视频和音频通道数。
- 维度数量。
- 双声道结构。
- batch 一致性。
- H3 采样 batch 1 限制。
- 视频 latent T 是否满足 `5k+2`。
- 视频 latent 高宽是否能被 DiT 的 `2x2` patch 整除。
- 指定 FPS 下音频 T 是否与视频时长相符。

图片 VAE 编码产生的 `T=1` 会识别为合法的图片条件/替换 latent，并提示它不是目标生成序列。

## 8. 帧数与 Latent 快捷换算

支持以下输入类型：

```text
video_frames
video_latent_t
audio_latent_t
seconds
```

统一输出：

```text
video_frames
video_latent_t
audio_latent_t
duration_seconds
source_was_exact
info
```

H3 合法序列：

```text
video_frames   = 17k + 5
video_latent_t = 5k + 2
audio_latent_t = round(video_frames / FPS * 40)
```

对齐模式：

| 模式 | 行为 |
|---|---|
| `up` | 向上对齐到最近合法值，适合创建目标 latent |
| `down` | 向下对齐到最近合法值，适合裁剪真实参考视频 |
| `nearest` | 对齐到距离最近的合法值 |
| `exact` | 输入不合法时直接报错 |

默认 24 FPS 示例：

```text
video_frames   = 124
video_latent_t = 37
audio_latent_t = 207
duration       = 5.166667 秒
```

## 9. 推荐工作流

### 图片替换视频 Latent

```text
图片
  -> CK MiniMax H3 Image VAE Encode
  -> replacement_latent

H3 联合 AV latent
  -> CK MiniMax H3 Replace Video Latent By Index
  -> Sampler 或后续 latent 操作
```

### 单独处理音频流

```text
联合 AV latent
  -> Separate AV Latent
       |-- video_latent
       `-- audio_latent -> 音频 latent 操作

video_latent + 修改后的 audio_latent
  -> Combine AV Latent
  -> 联合 AV latent
```

### 操作前后检查

```text
输入 latent
  -> Latent Info
  -> latent 操作
  -> Latent Info
```

建议在换 FPS、裁剪、替换或重新组合后再次运行信息节点，确认视频帧数、音频 T 和时长仍然匹配。
