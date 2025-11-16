import os
import io

# 1. 定义 WAS Suite 中使用的 TEXT_TYPE
TEXT_TYPE = "STRING"

class Text_Load_From_File:
    """
    从文件加载文本，过滤掉以'#'开头的行。
    返回完整的文本字符串和按行分割的字典。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": '', "multiline": False}),
                # 4. 修正了拼写错误 ( '[filename]]' -> '[filename]' )
                "dictionary_name": ("STRING", {"default": '[filename]', "multiline": False}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE, "DICT")
    FUNCTION = "load_file"

    CATEGORY = "CKNodes"

    def load_file(self, file_path='', dictionary_name='[filename]'):

        # 确保 os 模块被正确导入
        if not hasattr(os, 'path'):
            print("[Text_Load_From_File] Error: 'os' module not imported correctly.")
            return ('', {"error": []})

        try:
            # 提取文件名（不含扩展名）
            filename = ( os.path.basename(file_path).split('.', 1)[0]
                if '.' in os.path.basename(file_path) else os.path.basename(file_path) )
        except Exception as e:
            print(f"[Text_Load_From_File] Error getting basename: {e}")
            filename = "default" # 提供一个备用名称以防出错

        # 如果用户指定了字典名称，则使用它
        if dictionary_name != '[filename]':
            filename = dictionary_name

        # 检查文件是否存在
        if not os.path.exists(file_path):
            # 2. 用标准 print 替换 cstr
            print(f"[Text_Load_From_File] Warning: The path `{file_path}` specified cannot be found.")
            return ('', {filename: []})

        try:
            # 读取文件内容
            with open(file_path, 'r', encoding="utf-8", newline='\n') as file:
                text = file.read()
        except Exception as e:
            print(f"[Text_Load_From_File] Error reading file {file_path}: {e}")
            return ('', {filename: []})

        # 3. 移除了外部依赖 'update_history_text_files(file_path)'

        lines = []
        try:
            # 逐行处理文本
            for line in io.StringIO(text):
                # 忽略以 '#' 开头的行（注释）
                if not line.strip().startswith('#'):
                    # 移除换行符并添加到列表中
                    lines.append(line.replace("\n",'').replace("\r",''))
            dictionary = {filename: lines}
        except Exception as e:
            print(f"[Text_Load_From_File] Error processing text lines: {e}")
            return ('', {filename: []}) # 出错时返回空

        # 返回处理后的文本（所有行合并）和字典（每行一个元素）
        return ("\n".join(lines), dictionary)

# 3. 添加 ComfyUI 必需的节点映射
NODE_CLASS_MAPPINGS = {
    "Text_Load_From_File": Text_Load_From_File
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Text_Load_From_File": "👻从文件加载文本-CK"
}