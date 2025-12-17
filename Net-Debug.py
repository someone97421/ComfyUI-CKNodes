import os
import sys
import subprocess
import socket

# --- 1. 定义万能类型 (Any Type) ---
class AnyType(str):
    def __ne__(self, __value):
        return False

any_type = AnyType("*")

# --- 2. 核心检测逻辑 ---
def get_network_diagnostics():
    lines = []
    lines.append("🌐 --- 网络环境诊断报告 (Network Diagnostics) ---")
    
    # 1. [系统代理 System Proxy]
    proxy_keys = ['http_proxy', 'https_proxy', 'all_proxy', 'no_proxy']
    proxies = []
    for key in proxy_keys:
        # 检查大写和小写环境变量
        val = os.environ.get(key) or os.environ.get(key.upper())
        if val:
            proxies.append(f"  - {key.upper()}: {val}")
    
    if proxies:
        lines.append("[系统代理]:\n" + "\n".join(proxies))
    else:
        lines.append("[系统代理]: 无 (Direct)")

    # 2. [PIP 配置] (镜像与代理)
    pip_lines = []
    pip_index = os.environ.get('PIP_INDEX_URL')
    pip_proxy = os.environ.get('PIP_PROXY')
    
    if pip_index:
        pip_lines.append(f"  - 镜像源 (INDEX_URL): {pip_index}")
    else:
        pip_lines.append(f"  - 镜像源: 默认 (PyPI)")
        
    if pip_proxy:
        pip_lines.append(f"  - 独立代理 (PIP_PROXY): {pip_proxy}")
    
    lines.append("[PIP 配置]:\n" + "\n".join(pip_lines))

    # 3. [Hugging Face 镜像]
    hf_endpoint = os.environ.get('HF_ENDPOINT')
    if hf_endpoint:
        lines.append(f"[HF 镜像]: {hf_endpoint}")
    else:
        lines.append("[HF 镜像]: 未设置 (使用官方 hugginface.co)")

    # 4. [Git 配置]
    try:
        # 获取 global 配置
        git_out = subprocess.check_output(
            ['git', 'config', '--global', '--list'], 
            stderr=subprocess.STDOUT, text=True, timeout=2
        ).strip().split('\n')
        
        relevant_git = []
        for c in git_out:
            c = c.strip()
            # 筛选 url替换(insteadOf) 和 http.proxy
            if 'url' in c or 'proxy' in c:
                relevant_git.append(f"  - {c}")
                
        if relevant_git:
            lines.append("[Git 配置]:\n" + "\n".join(relevant_git))
        else:
            lines.append("[Git 配置]: 无全局代理/镜像设置")
    except FileNotFoundError:
        lines.append("[Git 配置]: 未找到 git 命令")
    except Exception as e:
        lines.append(f"[Git 配置]: 检测出错 ({str(e)})")

    # 5. [端口占用 Port Usage]
    try:
        import psutil
        lines.append("[端口占用]:")
        proc = psutil.Process()
        # 获取当前进程(ComfyUI)监听的端口
        listening = [c for c in proc.connections(kind='inet') if c.status == 'LISTEN']
        if listening:
            for c in listening:
                lines.append(f"  - 本地端口: {c.laddr.port} (类型: {c.type})")
        else:
            lines.append("  - 当前进程无监听端口 (可能由父进程管理)")
            
    except ImportError:
        lines.append("[端口占用]: 未安装 psutil 库，无法检测")
    except Exception as e:
        lines.append(f"[端口占用]: 检测失败 ({str(e)})")

    lines.append("------------------------------------------------")
    return "\n".join(lines)


# --- 3. 启动时立即执行打印 (Global Execution) ---
# 这段代码会在 ComfyUI 加载此节点文件时直接运行
print("\n" + "="*20 + " 👻-网络信息-👻 " + "="*20)
try:
    # 获取并打印报告
    start_report = get_network_diagnostics()
    print(start_report)
except Exception as e:
    print(f"❌ 自检脚本运行错误: {e}")
print("="*62 + "\n")


# --- 4. ComfyUI 节点定义 ---
class NetDebugNodeAny:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_data": (any_type,), 
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output_data",)
    
    FUNCTION = "do_debug"
    CATEGORY = "👻CKNodes"
    
    # 设为 True 确保节点始终运行
    OUTPUT_NODE = True

    def do_debug(self, input_data):
        # 运行时再次获取（以防中途修改了环境变量）
        report = get_network_diagnostics()
        
        # 控制台打印
        print("\n" + "▼"*20 + " 👻-网络信息-👻 " + "▼"*20)
        print(report)
        print("▲"*20 + " [End Report] " + "▲"*20 + "\n")

        # 直通数据
        return (input_data,)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "NetDebugNodeAny": NetDebugNodeAny
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NetDebugNodeAny": "👻网络信息-CK👻"
}