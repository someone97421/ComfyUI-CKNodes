
```markdown
# ComfyUI-CKNodes

一些自己使用 / AI辅助写的 ComfyUI 个人节点合集  
部分节点来自其他作者（已注明），部分为临时/实验性质节点

> 当前属于个人工具包，接口和功能可能不稳定，欢迎 issue/PR，但不保证长期维护。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/someone97421/ComfyUI-CKNodes.git
```

重启 ComfyUI 即可看到 CK 开头或相关分类的节点。

（如果有依赖，可运行 `pip install -r requirements.txt`）

## 节点一览

| 节点名称                        | 主要功能简述                                                                 | 备注                          |
|-------------------------------|-----------------------------------------------------------------------------|-------------------------------|
| **AnyNullNode**               | 万能空节点 / 占位 / 断开连接用（类似 None 输出）                              |                               |
| **ExtractFrames**             | 从批量图像/视频帧中向前或向后提取指定帧（支持负数索引）                        | 常用在视频帧处理流程          |
| **LTX_2_Context**             | （疑似）将某种格式转换为上下文相关结构（具体功能待补充）                       | 命名较特殊，待确认用途        |
| **LoadTextFile**              | 从指定路径读取纯文本文件内容并输出为 STRING                                     | 复制自 WAS Node Suite         |
| **Net-Debug**                 | 网络/调试相关节点（可能用于输出中间变量、打印信息等）                           | 具体功能待补充                |
| **NetSettings**               | 网络相关设置节点（可能用于配置 API、代理、超时等）                              | 具体功能待补充                |
| **QwenVL_Local_Loader**       | 本地加载 Qwen-VL (Qwen3-VL) 视觉语言模型，用于图像描述/理解任务                 | 改自 1038lab/ComfyUI-QwenVL   |
| **SaveImageCK**               | 增强版保存图像节点，支持选择输出编码格式（jpg/png/webp 等）                     | 改自 SaveImageKJ              |
| **Simple_LLM_Assistant**      | 简易本地/在线 LLM 助手节点（可用于改写提示词、翻译、简单问答等）                | 具体支持模型待补充            |
| **TextConcatenate**           | 简单地把多个字符串拼接起来（支持分隔符）                                       | 复制自 WAS Node Suite         |
| **any_list_count**            | 输入任意列表/数组，返回其中元素个数（INT）                                      | AI生成，小工具                |
| **text_line_count**           | 输入一段文本，返回总行数（INT）                                                | AI生成，小工具                |
| **zhenzhenVeo3APIPlus**       | 针对 ZhenZhen / Veo3 API 的增强调用节点（可能支持批量、参数优化等）             | 具体功能待补充，可能为实验节点 |

## 分类说明（大致）

- **文本处理**：LoadTextFile, TextConcatenate, text_line_count, Simple_LLM_Assistant
- **图像/视频帧**：ExtractFrames, SaveImageCK
- **模型加载**：QwenVL_Local_Loader
- **工具/调试**：AnyNullNode, any_list_count, Net-Debug, NetSettings
- **API/特殊**：zhenzhenVeo3APIPlus, LTX_2_Context

## 注意事项

- 大部分节点属于“能用就行”级别，输入输出类型校验不一定严格
- 部分节点依赖其他模型文件或 API key，请自行放置到对应目录
- 欢迎 issue 提出 bug、使用问题或想要的功能

祝使用愉快～
```
