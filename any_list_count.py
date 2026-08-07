import torch

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
    def __eq__(self, __value: object) -> bool:
        return True

class AnyListCount:
    """
    通用计数节点（修复版）。
    增加了 INPUT_IS_LIST = True，防止 ComfyUI 自动拆解列表。
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": (AnyType("*"),), 
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    
    FUNCTION = "count_any"
    CATEGORY = "CK Nodes/Logic"

    # 关键修改：这行告诉 ComfyUI 不要自动迭代列表，而是把整个列表传给函数
    INPUT_IS_LIST = True

    def count_any(self, any_input):
        # 注意：因为设置了 INPUT_IS_LIST = True，
        # any_input 永远是一个列表（Python List）。
        
        # 1. 如果列表本身有多项（例如你图中的文本列表）
        if len(any_input) > 1:
            return (len(any_input),)
        
        # 2. 如果列表只有1项，我们需要判断它是“单个对象”还是“一个Batch”
        elif len(any_input) == 1:
            item = any_input[0]
            
            # 如果是 PyTorch Tensor (Image Batch / Mask Batch)
            # 这种情况下，虽然列表只有1个对象（Tensor），但Tensor内部包含多个图片
            if isinstance(item, torch.Tensor):
                return (item.shape[0],)

            # 如果是 Latent (Batch)
            if isinstance(item, dict) and "samples" in item:
                if isinstance(item["samples"], torch.Tensor):
                    return (item["samples"].shape[0],)
            
            # 其他情况（普通的单个字符串、单个模型等）
            return (1,)

        # 3. 空列表
        return (0,)

NODE_CLASS_MAPPINGS = {
    "AnyListCount": AnyListCount
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyListCount": "CK Any List Count"
}
