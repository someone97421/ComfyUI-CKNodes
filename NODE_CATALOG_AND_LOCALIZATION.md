# CKNodes 节点分类与中文本地化

## 1. 统一分类体系

所有节点使用稳定的英文分类路径，避免切换界面语言时改变节点菜单结构或影响搜索记录。

```text
CK Nodes
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

MiniMax H3 节点已迁移到独立的 `minimax_h3_tools` 节点包。该套件保留原有节点 ID、显示名称、输入参数和分类，因此已有工作流可以继续索引到迁移后的节点。

## 2. 节点分布

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

枚举翻译只改变前端显示，工作流中仍保存后端真实值，因此切换语言不会改变执行逻辑。

## 4. Video Combine 依赖

`VHS_VideoCombineIsolated` 不再维护一份残缺的 VideoHelperSuite 源码副本，而是继承已安装的 `comfyui-VideoHelperSuite` 节点实现。

如果未安装 VideoHelperSuite，节点加载时会输出明确的缺失依赖错误。

## 5. Qwen3-VL Transformers 兼容

本地 Transformers 5.x 已移除 `AutoModelForVision2Seq` 导出，因此加载器采用兼容逻辑：

```text
Transformers 5.x -> AutoModelForImageTextToText
Transformers 4.x -> AutoModelForVision2Seq
```

节点 ID 和模型调用方式保持不变。
