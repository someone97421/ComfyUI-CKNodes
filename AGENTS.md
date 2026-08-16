# AGENTS.md — ComfyUI-CKNodes

个人使用的 ComfyUI 自定义节点合集。与用户用中文交流。

## 架构要点

- `__init__.py` 只扫描仓库根目录的 `.py` 文件并动态注册，新节点必须放在根目录单独一个 `.py` 里（子包不会被扫描，tests 也只 glob 根目录）。
- 每个节点文件定义 `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`；`__init__.py` 必须保留 `WEB_DIRECTORY = "./web"`（测试断言）。
- 根目录所有 `.py` 在 ComfyUI 启动时会被完整执行：模块级代码必须轻量，重依赖只允许在函数内部 import。
- MiniMax H3 节点已迁出到独立套件 `ComfyUI_Minimax_H3_Tools`，不要在本仓库重新添加同名节点。

## 节点开发硬性约定（tests/test_node_catalog.py 强制校验）

- `CATEGORY` 必须以 `CK Nodes/` 开头；英文显示名必须以 `CK ` 开头。
- 每个注册节点必须在 `locales/zh/nodeDefs.json` 中有完整中文条目（display_name、description、所有输入名、所有 Combo 选项），缺失即测试失败——新增/修改节点时必须同步更新该文件。
- Combo 枚举翻译只改前端显示，工作流中保存的是后端真实值，翻译时不要改后端枚举值。
- 新增节点后同步更新 `README.md` 节点一览表和 `NODE_CATALOG_AND_LOCALIZATION.md` 节点分布。

## 测试

- 纯 unittest，无 pytest 配置。`test_node_catalog.py` 是纯 AST 静态校验（不依赖 ComfyUI 运行时），任意 Python 即可运行；`test_frame_rate_match.py` 需要 torch（用 ComfyUI 的 Python 环境）。
- 只跑目录校验：`python -m unittest discover -s tests -p test_node_catalog.py`；全量：`python -m unittest discover -s tests`。
- 验证从简：只跑与改动相关的测试，不跑全量。

## 依赖约束

- 禁止擅自引入第三方依赖或重依赖。确需引入时，必须先向用户说明用途并征得同意后方可添加。
- `requirements.txt` 只维护已有依赖，不要静默增删。

## 隔离红线（官方仓库零污染）

- 严禁修改、monkey-patch ComfyUI 主程序的全局对象与模块（`comfy.*`、`nodes`、`folder_paths`、前端文件等），本仓库为完全旁路模式。
- 任何对运行环境的副作用（如环境变量、网络配置）只能发生在节点执行期间；未运行本仓库节点时，主程序行为必须与未安装本仓库完全一致。
- 提交/推送前必须完成相关测试，确认不影响官方仓库正常运行。
