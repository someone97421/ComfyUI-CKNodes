# CKNodes 节点分类与中文本地化

## 1. 统一分类体系

所有节点使用稳定的英文分类路径，避免切换界面语言时改变节点菜单结构或影响搜索记录。

```text
CK Nodes
├─ Video
│  ├─ LTXV
│  └─ Batch
├─ Image
│  ├─ Mask
│  ├─ Composition
│  └─ Output
├─ Text
├─ Logic
├─ AI
│  └─ LLM
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
- `CKMatchBatchFrameRate`

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
