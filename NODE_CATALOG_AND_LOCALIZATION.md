# CKNodes 节点分类与中文本地化

## 1. 统一分类体系

所有节点使用稳定的英文分类路径，避免切换界面语言时改变节点菜单结构或影响搜索记录。

```text
CK Nodes
├─ MiniMax H3
│  ├─ Conditioning
│  ├─ Latent
│  ├─ Temporal
│  └─ Mask
├─ Video
│  ├─ LTXV
│  ├─ Batch
│  └─ Output
├─ Image
│  ├─ Mask
│  ├─ Composition
│  └─ Output
├─ Text
├─ Logic
├─ AI
│  ├─ LLM
│  └─ Vision Language
└─ System
   └─ Network
```

## 2. 节点分布

### MiniMax H3 / Conditioning

- `CKMiniMaxH3ReferenceToVideoFPS`

### MiniMax H3 / Latent

- `CKMiniMaxH3SeparateAVLatent`
- `CKMiniMaxH3CombineAVLatent`
- `CKMiniMaxH3ImageVAEEncode`
- `CKMiniMaxH3ReplaceVideoLatentByIndex`
- `CKMiniMaxH3LatentInfo`
- `CKMiniMaxH3TimeConvert`
- `CKMiniMaxH3VideoVAEEncode`
- `CKMiniMaxH3VideoVAEEncodeMaskedNoise`

### MiniMax H3 / Temporal

- `CKMiniMaxH3TrimLatent`
- `CKMiniMaxH3ConcatLatents`

### MiniMax H3 / Mask

- `CKMiniMaxH3TemporalMask`
- `CKMiniMaxH3ApplyVideoMask`

### Video / LTXV

- `LTXVContext_TTP`
- `LTXVContext_Reverse_TTP`

### Video / Batch

- `ExtractFramesFromBatch`

### Video / Output

- `VHS_VideoCombineIsolated`

### Image / Mask

- `MaskBorderDrawer`

### Image / Composition

- `CKSmartMergeImages`

### Image / Output

- `SaveImageCK`

### Text

- `Text_Load_From_File`
- `TextLineCount`
- `Text_Concatenate`

### Logic

- `AnyNullNode`
- `AnyBooleanSwitch`
- `AnyListCount`

### AI / LLM

- `SimpleOpenAI_LLM`
- `SimpleClaude_LLM`

### AI / Vision Language

- `QwenVL_Local_Loader`

### System / Network

- `TemporaryNetSettings`
- `NetDebugNodeAny`

## 3. 中文语言包

中文翻译文件：

```text
locales/zh/nodeDefs.json
```

当 ComfyUI 语言切换为中文时，前端会根据稳定的节点 ID 加载对应翻译。

翻译范围包括：

- 节点显示名称。
- 节点功能说明。
- 输入参数名称。
- 输入参数提示。
- 输出名称。
- Combo 枚举选项。

枚举翻译只改变前端显示。例如：

```text
后端真实值: preserve_target
中文显示:   保留目标遮罩
```

工作流中仍保存 `preserve_target`，因此切换语言不会改变执行逻辑，也不会破坏已有工作流。

## 4. 默认语言约定

- Python 节点定义和 `NODE_DISPLAY_NAME_MAPPINGS` 使用英文默认名称。
- 中文界面使用 `locales/zh/nodeDefs.json` 覆盖显示文本。
- 节点 ID、函数参数名和枚举真实值保持不变。
- 分类路径保持英文，不随语言切换。

## 5. Video Combine 依赖

`VHS_VideoCombineIsolated` 不再维护一份残缺的 VideoHelperSuite 源码副本，而是继承已安装的：

```text
comfyui-VideoHelperSuite/videohelpersuite.nodes.VideoCombine
```

这样可以：

- 跟随 VideoHelperSuite 的格式和编码修复。
- 避免缺少 `logger.py`、`utils.py` 等同级模块导致加载失败。
- 保持原节点 ID `VHS_VideoCombineIsolated`，兼容已有工作流。
- 单独设置到 `CK Nodes/Video/Output` 分类。

如果未安装 VideoHelperSuite，节点加载时会输出明确的缺失依赖错误。

## 6. Qwen3-VL Transformers 兼容

本地 Transformers 5.x 已移除 `AutoModelForVision2Seq` 导出，因此加载器现在采用兼容逻辑：

```text
Transformers 5.x -> AutoModelForImageTextToText
Transformers 4.x -> AutoModelForVision2Seq
```

节点 ID 和模型调用方式保持不变。

## 7. 新增节点注册

`Smart_merge_images.py` 原先只有节点类，没有 `NODE_CLASS_MAPPINGS`，因此不会出现在 ComfyUI 中。

现在注册为：

```text
节点 ID: CKSmartMergeImages
显示名称: CK Smart Merge Images
分类: CK Nodes/Image/Composition
```
