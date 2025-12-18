import os

# --- 1. 定义万能类型 ---
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

class TemporaryNetSettings:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": (any_type, {}), 
            },
            "optional": {
                "http_proxy": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "placeholder": "e.g. http://127.0.0.1:7890 or None"
                }),
                "pip_mirror": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "placeholder": "e.g. https://pypi.tuna.tsinghua.edu.cn/simple"
                }),
                "git_mirror": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "placeholder": "e.g. https://ghproxy.com/"
                }),
                "huggingface_mirror": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "placeholder": "e.g. https://hf-mirror.com"
                }),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "apply_settings"
    CATEGORY = "👻CKNodes"
    OUTPUT_NODE = True

    DESCRIPTION = """
    临时修改当前运行环境的网络设置。
    - 输入 'None' = 彻底清除代理 (包含 HTTP, HTTPS, ALL_PROXY)。
    - 留空 = 保持当前系统原有设置。
    - 输入 URL = 设置为该代理或镜像。
    """

    def apply_settings(self, any_input, http_proxy, pip_mirror, git_mirror, huggingface_mirror):
        status_log = []
        
        def update_env(key_list, value, name):
            val = value.strip()
            
            if val == "":
                # 留空：不做修改，只报告当前状态
                current = os.environ.get(key_list[0])
                if current:
                    status_log.append(f"[{name}] Keep: {current}")
                else:
                    status_log.append(f"[{name}] Keep: (Not Set)")
                return

            if val.lower() == "none":
                # 输入 None：清除列表中的所有 Key
                for key in key_list:
                    if key in os.environ:
                        del os.environ[key]
                status_log.append(f"[{name}] Cleared")
            else:
                # 输入值：设置列表中的所有 Key
                # 特殊处理：如果是 Proxy 设置，通常不应把 NO_PROXY 设置为 URL，所以需要分离逻辑
                # 但为了简单起见，这里假设 key_list 都是同类项。
                # 下面主逻辑中我们把 NO_PROXY 单独处理了。
                for key in key_list:
                    os.environ[key] = val
                status_log.append(f"[{name}] Set: {val}")

        # --- 1. 设置代理 (核心修改) ---
        # 包含了 ALL_PROXY，这是很多工具的默认回退代理
        proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
        
        # 如果用户输入 None，我们顺便把 NO_PROXY 也清理掉，确保完全纯净
        if http_proxy.strip().lower() == "none":
            proxy_keys.extend(["NO_PROXY", "no_proxy"])
            update_env(proxy_keys, "None", "Proxy")
        else:
            # 如果是设置代理，只设置 http/https/all，不设置 no_proxy
            update_env(proxy_keys, http_proxy, "Proxy")

        # --- 2. Pip 镜像 ---
        update_env(["PIP_INDEX_URL"], pip_mirror, "Pip Mirror")

        # --- 3. HuggingFace 镜像 ---
        update_env(["HF_ENDPOINT"], huggingface_mirror, "HF Mirror")

        # --- 4. Git 代理 (GH_PROXY 环境变量) ---
        # 注意：这不会改变 git config --global 中的 http.proxy，只改变 ComfyUI 脚本常用的环境变量
        update_env(["GH_PROXY"], git_mirror, "Git/GH Proxy")

        final_text = "\n".join(status_log)
        print(f"\n--- Network Settings Updated ---\n{final_text}\n--------------------------------")

        return {"ui": {"text": [final_text]}, "result": (any_input,)}

NODE_CLASS_MAPPINGS = {
    "TemporaryNetSettings": TemporaryNetSettings
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemporaryNetSettings": "👻网络设置-CK👻"
}