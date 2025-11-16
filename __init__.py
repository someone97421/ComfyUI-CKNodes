# 文件名: __init__.py

import os
import importlib.util
import sys

# 获取当前 __init__.py 文件的目录
NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# 初始化空的映射字典，ComfyUI 将从这里读取
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 遍历目录中的所有文件
for filename in os.listdir(NODE_DIR):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # 移除 .py 后缀
        file_path = os.path.join(NODE_DIR, filename)

        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 检查并合并 NODE_CLASS_MAPPINGS
            if hasattr(module, "NODE_CLASS_MAPPINGS") and isinstance(module.NODE_CLASS_MAPPINGS, dict):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            
            # 检查并合并 NODE_DISPLAY_NAME_MAPPINGS
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and isinstance(module.NODE_DISPLAY_NAME_MAPPINGS, dict):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        except Exception as e:
            print(f"[{os.path.basename(NODE_DIR)}] Error loading node from {filename}: {e}")

# 告诉 ComfyUI 这个包暴露了哪些映射
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# (可选) 打印一条消息到控制台，确认加载成功
print(f"Loaded custom nodes from {os.path.basename(NODE_DIR)}: {list(NODE_CLASS_MAPPINGS.keys())}")