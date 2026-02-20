import torch

# 定义通配符常量，方便后续使用
ANY = "*"

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

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("Null/Empty",)
    FUNCTION = "do_nothing"
    CATEGORY = "👻CKNodes"

    def do_nothing(self):
        return (None,)

class AnyBooleanSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入": (ANY,),
                "开关": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
            }
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("输出结果",)
    FUNCTION = "process"
    CATEGORY = "👻CKNodes"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def process(self, 输入, 开关): # 注意这里的参数顺序与输入定义一致
        if 开关:
            return (输入,)
        else:
            return (None,)

# 节点映射导出 - 修复了逗号问题
NODE_CLASS_MAPPINGS = {
    "AnyNullNode": AnyNullNode,
    "AnyBooleanSwitch": AnyBooleanSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyNullNode": "👻空输入InputNone👻",
    "AnyBooleanSwitch": "👻任意布尔开关👻",
}
