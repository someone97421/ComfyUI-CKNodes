# CK MiniMax H3 时序、拼接与遮罩节点

## 1. 节点列表

| 节点 | 用途 |
|---|---|
| `CK MiniMax H3 Video VAE Encode` | 将 IMAGE 批次作为视频帧编码成 H3 视频 latent |
| `CK MiniMax H3 Trim Latent` | 按视频 latent T 截取，并同步截取音频 |
| `CK MiniMax H3 Concat Latents` | 使用 H3 合法边界重叠拼接两个视频或 AV latent |
| `CK MiniMax H3 Temporal Mask` | 兼容节点：创建视频时间遮罩，并可同步映射音频 |
| `CK MiniMax H3 Video Temporal Mask` | 按 video latent T 只修改视频遮罩 |
| `CK MiniMax H3 Audio Temporal Mask` | 按 audio latent T 只修改音频遮罩 |
| `CK MiniMax H3 Apply Video Mask` | 将逐帧像素 MASK 转换为视频 latent mask |
| `CK MiniMax H3 Video VAE Encode Masked Noise` | 视频 VAE 编码时在 MASK 白色区域加入 latent 噪声 |

## 2. 单图与视频编码的区别

`CK MiniMax H3 Image VAE Encode`：

- 从 IMAGE batch 中通过 `batch_index` 选择一张图片。
- 输出固定 `T=1`。
- 用于图片条件或单个 latent 时间位置替换。

`CK MiniMax H3 Video VAE Encode`：

- 将整个 IMAGE batch 解释为视频时间帧。
- 帧数按照 `17k+5` 对齐。
- 输出视频 latent T 满足 `5k+2`。
- 用于完整视频 VAE 编码。

两者是不同语义，因此保留为独立节点。

## 3. 视频帧对齐

视频编码节点提供三种模式：

| 模式 | 行为 |
|---|---|
| `down` | 从尾部裁掉多余帧，向下对齐到最近的 `17k+5` |
| `up` | 重复最后一帧，向上补齐到最近的 `17k+5` |
| `exact` | 输入帧数不满足 `17k+5` 时直接报错 |

真实参考视频通常使用 `down`；需要保持全部帧时可以使用 `up`。

## 4. Latent 截取

截取索引使用视频 latent T，不是像素视频帧索引。

严格模式要求：

```text
start_index % 5 == 0
latent_length = 5k + 2
```

这样可以保持 H3 的 `1,4,4,4,4` 时间跨度相位。

联合 AV latent 会按照：

```text
视频 latent 边界
    -> 对应像素帧位置
    -> FPS
    -> 40 Hz 音频 latent 位置
```

同步截取音频。

`raw` 模式允许任意索引，但输出会从新的零相位重新解释时间跨度，属于实验性操作。

## 5. Latent 拼接

两个合法 H3 视频 latent 的长度为：

```text
T1 = 5k1 + 2
T2 = 5k2 + 2
```

直接拼接会得到：

```text
T1 + T2 = 5(k1+k2) + 4
```

这不是合法目标长度。

`h3_overlap` 模式将：

- 第一段最后 2 个视频 token 与第二段最前 2 个 token 重叠。
- 第二段从索引 2 开始继续，保持全局五阶段时间相位。
- 对应像素时间重叠为 5 帧。
- 音频重叠长度根据最终视频帧数和 40 Hz 目标长度反算，避免四舍五入产生一帧误差。

拼接后：

```text
Tout = T1 + T2 - 2
Fout = F1 + F2 - 5
```

重叠区支持：

- 线性混合。
- 保留第一段。
- 保留第二段。

## 6. 时间遮罩

时间遮罩现在分为三种明确语义：

| 节点 | 索引单位 | 修改范围 |
|---|---|---|
| `Video Temporal Mask` | video latent T | 只修改视频 mask，音频 mask 原样保留 |
| `Audio Temporal Mask` | audio latent T（40 Hz） | 只修改音频 mask，视频 mask 原样保留 |
| `Temporal Mask` | video latent T | 兼容旧工作流，可通过 `affect_audio` 同步映射音频 |

区间统一使用左闭右开形式：

```text
[start_index, end_index)
```

白色或强度 1 表示参与加噪和去噪，黑色或强度 0 表示尽量保留原 latent。

支持：

- 区间内强度。
- 区间外强度。
- 时间边界羽化。
- 纯视频、纯音频或联合 AV 的独立遮罩处理。
- 兼容节点可同步映射联合音频。
- 与已有遮罩替换、相乘、最大值或最小值组合。

音频独立遮罩不再通过视频帧跨度换算，输入索引直接对应 40 Hz 音频 latent，因此例如：

```text
[40, 80) = 第 1 秒到第 2 秒
```

## 7. 视频 Mask 到 Latent Mask

输入 MASK 可以是：

- 单张静态 mask。
- 与视频帧数一致的 mask batch。
- 其他帧数的 mask batch，节点会进行时间适配。

推荐的 H3 映射模式：

### `h3_max`

按照 `1,4,4,4,4` 的像素帧跨度聚合。某个视频 latent token 覆盖范围内只要存在白色区域，该区域就保留到 latent mask。

适合修补和局部重绘，默认使用。

### `h3_mean`

对每个 latent token 覆盖的像素帧 mask 取平均值，产生更平滑的时间强度。

### `trilinear` / `nearest`

将像素 mask 当作普通三维体直接插值到 latent 时空尺寸，不考虑 H3 非均匀帧跨度。

## 8. 视频 VAE 遮罩加噪编码

处理流程：

```text
视频 IMAGE batch
    -> 帧数与画布对齐
    -> H3 视频 VAE 编码
    -> 标准化视频 latent

像素 MASK batch
    -> 同步帧数与画布
    -> H3 时间跨度聚合
    -> latent noise mask

标准正态 latent 噪声
    -> 仅在 mask 白色区域与编码 latent 混合
```

计算形式：

```text
mix = latent_mask * noise_strength
noisy_latent = clean_latent * (1 - mix) + noise * mix
noise_mask = latent_mask * denoise_strength
```

输出：

- `noisy_latent`：遮罩区域已经加噪，并带有 noise mask。
- `clean_latent`：未经加噪的原始 VAE 编码，便于比较或复用。
- 实际视频帧数。
- 视频 latent T。

## 9. Mask 语义

本套节点统一遵循 ComfyUI noise mask 语义：

```text
白色 / 1.0 = 加噪、允许模型去噪生成
黑色 / 0.0 = 保留原 latent
灰色         = 按强度混合
```

如果输入 mask 语义相反，开启 `invert_mask`。
