# ComfyUI-CKNodes

个人使用的 ComfyUI 节点合集。部分节点来自其他项目并保留来源说明，部分为实验性工具，接口和行为可能随实际工作流调整。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/someone97421/ComfyUI-CKNodes.git
cd ComfyUI-CKNodes
pip install -r requirements.txt
```

安装后完整重启 ComfyUI。节点通常显示在 `CKNodes`、`CK` 或各自功能分类下。

## MiniMax H3 可调帧率参考节点

### CK MiniMax H3 Reference to Video (Adjustable FPS)

为 ComfyUI 本地 MiniMax H3 Reference to Video 工作流提供可调模型时间轴帧率。

节点位置：

```text
CKNodes/minimax
```

节点 ID：

```text
CKMiniMaxH3ReferenceToVideoFPS
```

该节点基于 ComfyUI 本地 MiniMax H3 Reference to Video 节点扩展，保留参考图片、参考视频、视频配套音轨和独立参考音频的处理能力，并新增 `frame_rate` 输入。

`frame_rate` 不会插帧或删帧，而是按照 H3 音频 latent 的 40 Hz 时间基准同步调整：

```text
frame_rescale = 40 / frame_rate
```

调整范围包括：

- 目标音频 latent 长度。
- 目标视频 DiT 时间坐标。
- 参考视频 DiT 时间坐标。
- 参考块与目标块的时间游标。
- Qwen3-VL 参考视频抽帧间隔和时间戳。

例如同样生成 124 帧：

| 模型时间轴帧率 | 视频时长 | 目标音频 latent T |
|---:|---:|---:|
| 24 FPS | 5.1667 秒 | 207 |
| 16 FPS | 7.7500 秒 | 310 |

从 24 FPS 改为 16 FPS 后，帧数保持不变，模型内部视频和音频时间跨度扩大为 1.5 倍。

使用要求：

- 最终视频合成或保存节点必须设置为相同 FPS。节点设为 16 FPS 时，输出节点也应设为 16 FPS。
- 参考视频使用同一个 `frame_rate` 解释。将实际 24 FPS 参考视频按 16 FPS 输入时，相当于把参考运动时间轴拉长 1.5 倍。
- 节点不会自动对参考音频执行保音高 time-stretch。需要严格同步时，应在输入前处理参考音频时长。
- 参考音频应为双声道。
- 16 FPS 能保证相同帧数覆盖更长播放时间，但不能保证生成动作严格按固定倍率减速。
- 该实现会在运行时为 CK 条件分派自定义 H3 `PackedLayout`；普通官方 H3 节点仍使用原生 24 FPS 布局。
- 依赖包含本地 MiniMax H3 支持的较新 ComfyUI 版本。ComfyUI 修改 H3 内部布局接口后，本节点可能需要同步适配。

## 节点一览

| 节点/模块 | 主要功能 | 备注 |
|---|---|---|
| **CK MiniMax H3 Reference to Video (Adjustable FPS)** | 本地 MiniMax H3 多媒体参考条件与可调音视频时间轴 | 默认 16 FPS |
| **AnyNullNode** | 任意类型空值、占位和断开连接工具 | 工具节点 |
| **ExtractFrames** | 从 IMAGE batch 的开头或结尾提取指定数量帧 | 视频帧处理 |
| **LTXV Context (Forward/Reverse)** | 将相邻视频片段的首尾帧编码并注入 LTXV latent | 支持前向和反向衔接 |
| **LoadTextFile** | 从路径读取文本文件并输出字符串 | 来源见源码 |
| **MaskBorderDrawer** | 绘制和处理遮罩边界 | 图像/遮罩工具 |
| **Net-Debug** | 网络请求调试工具 | 调试节点 |
| **NetSettings** | 网络请求相关设置 | 调试节点 |
| **QwenVL Local Loader** | 本地加载 Qwen-VL/Qwen3-VL，用于图像理解 | 改自 1038lab/ComfyUI-QwenVL |
| **SaveImageCK** | 支持多种编码格式的增强图像保存节点 | 改自 SaveImageKJ |
| **Simple LLM Assistant** | 简易 LLM 提示词处理、翻译和问答 | 需要对应模型或服务配置 |
| **Simple Claude LLM** | Claude 模型调用节点 | 需要对应 API 配置 |
| **Smart Merge Images** | 局部图像融合 | 选自 supElement/ComfyUI_Element_easy |
| **TextConcatenate** | 使用指定分隔符拼接字符串 | 来源见源码 |
| **any_list_count** | 统计任意列表或数组中的元素数量 | 工具节点 |
| **text_line_count** | 统计文本行数 | 工具节点 |

## 注意事项

- 部分节点依赖模型文件、外部服务或 API key，请按节点输入和源码要求配置。
- `requirements.txt` 包含 Qwen-VL 等节点所需依赖；MiniMax H3 节点本身复用 ComfyUI 已有的 `torch`、`torchaudio` 和节点 API。
- 仓库中的节点由 `__init__.py` 自动扫描并注册。单个节点导入失败时，ComfyUI 控制台会显示对应文件名和异常信息。
- 更新 ComfyUI 或第三方依赖后，如节点加载失败，请先检查控制台导入错误和当前依赖版本。
