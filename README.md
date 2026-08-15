# ComfyUI-CKNodes

个人使用的 ComfyUI 节点合集。部分节点来自其他项目并保留来源说明，部分为实验性工具，接口和行为可能随实际工作流调整。

# MiniMax H3 节点已迁移

需要 MiniMax H3 工具的用户，请前往独立节点套件仓库获取：

**[ComfyUI MiniMax H3 Tools](https://github.com/someone97421/ComfyUI_Minimax_H3_Tools)**

原有 MiniMax H3 工作流使用的节点 ID 保持不变，安装独立套件后仍可正常索引。不要同时启用本仓库和独立套件中的同名 MiniMax H3 节点。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/someone97421/ComfyUI-CKNodes.git
cd ComfyUI-CKNodes
pip install -r requirements.txt
```

安装后完整重启 ComfyUI。节点通常显示在 `CK Nodes`、`CK` 或各自功能分类下。

MiniMax H3 节点已迁移到独立套件 `minimax_h3_tools`。原有 MiniMax H3 工作流使用的节点 ID 保持不变，请安装独立套件以继续加载这些节点。

## 节点一览

| 节点/模块 | 主要功能 | 备注 |
|---|---|---|
| **AnyNullNode** | 任意类型空值、占位和断开连接工具 | 工具节点 |
| **ExtractFrames** | 从 IMAGE batch 的开头或结尾提取指定数量帧 | 视频帧处理 |
| **Match Batch Frame Rate** | 根据输入/输出 FPS 沿时间轴自动匹配抽帧，并输出帧数、时长与索引信息 | 支持降帧及重复帧升帧，不做插值 |
| **LTXV Context (Forward/Reverse)** | 将相邻视频片段的首尾帧编码并注入 LTXV latent | 支持前向和反向衔接 |
| **LoadTextFile** | 从路径读取文本文件并输出字符串 | 来源见源码 |
| **MaskBorderDrawer** | 绘制和处理遮罩边界 | 图像/遮罩工具 |
| **Net-Debug** | 网络请求调试工具 | 调试节点 |
| **NetSettings** | 网络请求相关设置 | 调试节点 |
| **QwenVL Local Loader** | 本地加载 Qwen-VL/Qwen3-VL，用于图像理解 | 改自 1038lab/ComfyUI-QwenVL |
| **SaveImageCK** | 支持多种编码格式的增强图像保存 | 改自 SaveImageKJ |
| **Simple LLM Assistant** | 简易 LLM 提示词处理、翻译和问答 | 需要对应模型或服务配置 |
| **Simple Claude LLM** | Claude 模型调用节点 | 需要对应 API 配置 |
| **Smart Merge Images** | 局部图像融合 | 选自 supElement/ComfyUI_Element_easy |
| **TextConcatenate** | 使用指定分隔符拼接字符串 | 来源见源码 |
| **any_list_count** | 统计任意列表或数组中的元素数量 | 工具节点 |
| **text_line_count** | 统计文本行数 | 工具节点 |

## 注意事项

- 部分节点依赖模型文件、外部服务或 API key，请按节点输入和源码要求配置。
- `requirements.txt` 包含 Qwen-VL 等节点所需依赖。
- 仓库中的节点由 `__init__.py` 自动扫描并注册。单个节点导入失败时，ComfyUI 控制台会显示对应文件名和异常信息。
- 更新 ComfyUI 或第三方依赖后，如节点加载失败，请先检查控制台导入错误和当前依赖版本。
