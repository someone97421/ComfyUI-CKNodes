import torch

class AnyNullNode:
    """
    一个万能的空节点。
    它不需要输入，输出类型为 '*' (通配符)，可以连接到任何输入端口。
    实际传递的值为 None。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
        }

    # "*" 代表通配符，允许连接到任何类型的输入插槽
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("Null/Empty",)
    FUNCTION = "do_nothing"
    CATEGORY = "👻CKNodes"

    def do_nothing(self):
        # 返回 None，即“空”
        return (None,)

# 节点映射导出
NODE_CLASS_MAPPINGS = {
    "AnyNullNode": AnyNullNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyNullNode": "空输入InputNone"
}