class TextLineCount:
    """
    一个简单的ComfyUI节点，用于计算输入文本的行数。
    增加了是否忽略空行的选项。
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                # 新增开关，默认关闭 (False)，即默认统计所有行
                "ignore_empty_lines": ("BOOLEAN", {"default": False}), 
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("line_count",)
    
    FUNCTION = "count_lines"

    # 将节点分类在 utils 下，方便查找
    CATEGORY = "CK Nodes/Text"

    def count_lines(self, text, ignore_empty_lines):
        # 如果文本为空，返回0
        if not text:
            return (0,)
            
        # 使用 splitlines() 方法，它可以自动处理 \n, \r, \r\n 等不同系统的换行符
        lines = text.splitlines()
        
        # 如果开启了忽略空行
        if ignore_empty_lines:
            # line.strip() 去除首尾空格，如果不等于 "" 说明有内容
            # 这意味着纯空格的行也会被视为“空行”被剔除
            lines = [line for line in lines if line.strip() != ""]
        
        count = len(lines)
        
        return (count,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "TextLineCount": TextLineCount
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextLineCount": "CK Text Line Count"
}
