# MiniMax H3 可调帧率节点原理

## 1. 调整目标

`CK MiniMax H3 Reference to Video (Adjustable FPS)` 的 `frame_rate` 输入用于改变 H3 对视频帧时间跨度的解释。

例如同样生成 124 帧：

```text
24 FPS：124 / 24 = 5.1667 秒
16 FPS：124 / 16 = 7.7500 秒
```

从 24 FPS 调整到 16 FPS 后，帧数不变，时间长度扩大为：

```text
24 / 16 = 1.5 倍
```

该输入不会插帧或删帧，而是同步调整 H3 内部的视频和音频时间轴。

## 2. 为什么不能只修改播放 FPS

MiniMax H3 会联合生成视频和音频。若只将最终视频设置为 16 FPS：

- 视频会从 5.1667 秒变为 7.75 秒。
- 模型生成的音频仍然只有约 5.1667 秒。
- 模型采样期间仍按原生 24 FPS 理解视频运动。

若只增加目标音频 latent 长度，而不修改视频时间坐标，又会导致模型内部的视频和音频时长不一致。

因此，帧率变化必须同时作用于：

1. 目标音频 latent 长度。
2. 目标视频的 DiT 时间坐标。
3. 参考视频的 DiT 时间坐标。
4. 参考视频提供给 Qwen3-VL 的抽帧间隔和时间戳。

## 3. 统一时间基准

H3 音频 latent 的时间频率固定为 40 Hz，即：

```text
1 秒 = 40 个音频 latent 时间单位
```

视频每帧在这条统一时间轴上的跨度为：

```text
frame_rescale = 40 / frame_rate
```

对应数值：

```text
24 FPS：40 / 24 = 1.6666667
16 FPS：40 / 16 = 2.5
```

因此，16 FPS 下相邻视频帧的时间间距是 24 FPS 的 1.5 倍。

## 4. 目标音频 Latent 长度

目标视频帧数会先对齐到 H3 要求的 `17k+5`，随后按所选帧率计算时长：

```text
duration = frame_count / frame_rate
audio_latent_t = round(duration * 40)
```

以 124 帧为例：

| 帧率 | 视频时长 | 音频 latent T |
|---:|---:|---:|
| 24 FPS | 5.1667 秒 | 207 |
| 16 FPS | 7.7500 秒 | 310 |

视频 latent 的数量仍为 37，不会随 FPS 改变；变化的是视频 latent 在时间轴上的间距，以及与其匹配的音频 latent 长度。

## 5. 视频 DiT 时间坐标

H3 的视频 VAE 使用非逐帧时间压缩。每 5 个视频 latent token 对应 17 个像素帧：

```text
FRAME_PER_TOKEN = [1, 4, 4, 4, 4]
```

每个视频 token 的时间跨度为：

```text
token_span = FRAME_PER_TOKEN[i % 5] * frame_rescale
```

官方 H3 固定使用：

```text
frame_rescale = 40 / 24
```

CK 节点将其替换为：

```text
frame_rescale = 40 / 用户选择的 frame_rate
```

124 帧对应 37 个视频 latent token：

| 帧率 | 视频 token 覆盖总时长 | 音频 latent T |
|---:|---:|---:|
| 24 FPS | 206.6667 | 207 |
| 16 FPS | 310.0000 | 310 |

这样视频和音频在同一条 40 Hz 时间轴上的覆盖时长保持一致。

## 6. 参考视频时间映射

参考视频同样需要使用新的 `frame_rescale`。否则参考视频仍按 24 FPS 排列，而目标视频按 16 FPS 排列，参考运动速度和后续条件块的起始时间都会错位。

CK 节点为每个参考视频记录：

```text
ck_frame_rescale = 40 / frame_rate
```

自定义 `PackedLayout` 使用该值计算：

- 参考视频 token 的时间坐标。
- 参考视频结束后的时间游标。
- 后续参考块和目标音视频块的起始位置。

## 7. Qwen3-VL 时间戳

Qwen3-VL 不查看参考视频的全部帧，而是按约 2 FPS 抽样：

```text
sample_step = round(frame_rate / 2)
timestamp = frame_index / frame_rate
```

例如：

```text
24 FPS：每 12 帧抽一帧
16 FPS：每 8 帧抽一帧
```

两种设置都得到约 0.5 秒一次的视觉采样，并为 Qwen 提供与新时间轴一致的时间戳。

## 8. 实现隔离

CK 节点不修改 ComfyUI 核心文件。它使用带有 `ck_frame_rescale` 的专用参考列表，并在运行时分派 `PackedLayout`：

```text
CK 节点产生的 refs -> CKMiniMaxH3PackedLayout
普通官方 refs       -> 官方 PackedLayout
```

因此官方 MiniMax H3 节点继续使用原生 24 FPS 时间映射，只有 CK 可调帧率节点使用动态时间坐标。

## 9. 使用边界

### 最终输出 FPS 必须一致

CK 节点只调整模型内部时间轴，不控制最终视频容器。节点设置为 16 FPS 时，视频合成或保存节点也必须设置为 16 FPS。

### 帧率调整不是插帧

`length=124` 时始终生成 124 帧。降低 FPS 只会让这些帧覆盖更长时间。

### 参考音频不会自动拉伸

节点会调整目标音频 latent 长度，但不会对输入的参考音频 waveform 执行保音高 time-stretch。若参考视频从 24 FPS 按 16 FPS 解释，配套参考音频也应在输入前拉伸 1.5 倍，以保持严格同步。

### 不保证生成动作严格等比例减速

16 FPS 设置能够保证时间轴和最终播放时长扩大 1.5 倍，但模型可能提前完成动作、增加动作或保持姿态。它不能数学保证画面内容严格按 2/3 倍速运动。

若要求逐帧内容完全不变且必定慢放，应采用：

```text
按 24 FPS 生成
-> 保持所有视频帧不变
-> 以 16 FPS 封装
-> 将音频保音高拉伸 1.5 倍
```

## 10. 核心结论

该节点的帧率调整不是简单修改输出 FPS，而是按照：

```text
frame_rescale = 40 / frame_rate
```

同步重建 H3 的联合音视频时间关系，包括：

```text
目标音频 latent 长度
目标视频 token 时间坐标
参考视频 token 时间坐标
参考块与目标块的时间游标
Qwen3-VL 抽帧间隔和时间戳
```

在 124 帧条件下，将 24 FPS 改为 16 FPS，会使模型时间轴和最终播放时长从约 5.17 秒扩大到 7.75 秒，同时保持视频与目标音频的内部时间跨度一致。
