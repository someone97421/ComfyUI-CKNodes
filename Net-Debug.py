import os
import sys
import subprocess

# --- 1. 定义万能类型 (Any Type) ---
# 确保任何类型的连线都能接入
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

# 实例化万能对象
any_type = AnyType("*")

# --- 2. 核心检测逻辑 ---
def get_network_diagnostics():
    lines = []
    lines.append("🌐 --- 网络环境诊断报告 (Diagnostics) ---")
    
    # 1. [系统环境变量代理]
    # 检查大写和小写，以及 ALL_PROXY
    proxy_keys = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
        'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'
    ]
    
    active_proxies = []
    for key in proxy_keys:
        val = os.environ.get(key)
        if val:
            active_proxies.append(f"  - {key}: {val}")
    
    if active_proxies:
        lines.append("[当前生效代理 (Environment)]:\n" + "\n".join(active_proxies))
    else:
        lines.append("[当前生效代理 (Environment)]: 无 (Direct/None)")

    # 2. [特殊加速配置]
    special_lines = []
    
    # PIP
    pip_index = os.environ.get('PIP_INDEX_URL')
    if pip_index:
        special_lines.append(f"  - PIP 源: {pip_index}")
    
    # HuggingFace
    hf_endpoint = os.environ.get('HF_ENDPOINT')
    if hf_endpoint:
        special_lines.append(f"  - HF 镜像: {hf_endpoint}")
    else:
        special_lines.append(f"  - HF 镜像: 默认 (huggingface.co)")

    # GH_PROXY (ComfyUI 常用)
    gh_proxy = os.environ.get('GH_PROXY')
    if gh_proxy:
        special_lines.append(f"  - Git/GH 加速: {gh_proxy}")
        
    lines.append("[镜像/加速源]:\n" + "\n".join(special_lines))

    # 3. [Git 全局配置]
    try:
        git_out = subprocess.check_output(
            ['git', 'config', '--global', '--list'], 
            stderr=subprocess.STDOUT, text=True, timeout=2
        ).strip().split('\n')
        
        relevant_git = []
        for c in git_out:
            c = c.strip()
            if 'url' in c or 'proxy' in c:
                relevant_git.append(f"  - {c}")
                
        if relevant_git:
            lines.append("[Git 全局文件配置 (Global Config)]:\n" + "\n".join(relevant_git))
        else:
            lines.append("[Git 全局文件配置]: 无")
    except Exception:
        pass

    lines.append("------------------------------------------------")
    return "\n".join(lines)


# --- 3. 启动时立即执行打印 (Global Execution) ---
# 【保留功能】这段代码会在 ComfyUI 启动/加载此节点时直接运行
print("\n" + "="*20 + " 👻-网络信息(启动监测)-👻 " + "="*20)
try:
    print(get_network_diagnostics())
except Exception as e:
    print(f"❌ 启动自检失败: {e}")
print("="*62 + "\n")


# --- 4. ComfyUI 节点定义 ---
class NetDebugNodeAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 【修改】只保留这一个输入口，名称统一
                # 使用 any_type 确保可以接任何东西
                "any_input": (any_type, {}), 
            },
        }

    # 输出也是 AnyType
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any_output",)
    
    FUNCTION = "do_debug"
    CATEGORY = "CK Nodes/System/Network"

    DESCRIPTION = """
    在控制台显示当前代理及镜像设置
    """
    
    # 设为 True 确保节点始终运行
    OUTPUT_NODE = True

    def do_debug(self, any_input):
        # 运行时再次获取（显示最新状态）
        report = get_network_diagnostics()
        
        # 控制台打印
        print("\n" + "▼"*20 + " 👻-网络状态快照-👻 " + "▼"*20)
        print(report)
        print("▲"*20 + " [End Report] " + "▲"*20 + "\n")

        # 返回 UI 显示文本，并透传输入数据
        return {"ui": {"text": [report]}, "result": (any_input,)}

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "NetDebugNodeAny": NetDebugNodeAny
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NetDebugNodeAny": "CK Network Diagnostics"
}
