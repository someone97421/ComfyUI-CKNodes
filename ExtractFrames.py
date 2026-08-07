import torch

class ExtractFramesFromBatch:
    """
    一个ComfyUI节点，用于从图像批次中提取指定数量的帧。
    
    输入:
    - image: 图像批次 (B, H, W, C)
    
    参数:
    - start_index: 起始帧的索引 (从0开始)
    - direction: 提取方向 ("forward" 或 "backward")
    - frame_count: 要提取的总帧数
    
    输出:
    - image: 提取出的新图像批次
    """
    
    @classmethod
    def INPUT_TYPES(s):
        """
        定义节点的输入类型和参数
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "start_index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192,  # 允许一个较大的最大值
                    "step": 1
                }),
                "direction": (["forward", "backward"], {
                    "default": "forward"
                }),
                "frame_count": ("INT", {
                    "default": 1, 
                    "min": 1,     # 至少提取1帧
                    "max": 8192,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_frames"
    CATEGORY = "CK Nodes/Video/Batch"

    def extract_frames(self, image, start_index, direction, frame_count):
        """
        主要的执行函数
        """
        # 获取输入的图像批次总帧数
        # image shape is (B, H, W, C)
        total_frames = image.shape[0]

        # 1. 处理空批次或无效输入的边缘情况
        if total_frames == 0:
            print("ExtractFrames: 输入批次为空，返回空批次。")
            return (image,) # 直接返回空批次

        # 2. 确保参数有效
        # 确保 start_index 不会超过总帧数减1 (因为索引从0开始)
        start_index_clamped = max(0, min(start_index, total_frames - 1))
        # 确保 frame_count 至少为 1
        frame_count_clamped = max(1, frame_count)

        # 3. 根据方向计算切片索引
        if direction == "backward":
            # 向前提取：从 start_index 开始，提取 frame_count 帧
            start_slicer = start_index_clamped
            # 结束索引不能超过总帧数
            end_slicer = min(start_slicer + frame_count_clamped, total_frames)
            
        else: # direction == "forward"
            # 向后提取：从 start_index 向前（索引减小）提取 frame_count 帧
            # 结束索引是 start_index + 1 (因为切片不包含end)
            end_slicer = start_index_clamped + 1
            # 开始索引不能小于 0
            start_slicer = max(0, end_slicer - frame_count_clamped)

        # 4. 执行切片
        print(f"ExtractFrames: 原始批次大小: {total_frames} 帧")
        print(f"ExtractFrames: 模式: {direction}, 起始索引: {start_index}, 提取数量: {frame_count}")
        print(f"ExtractFrames: 实际切片范围: [{start_slicer}:{end_slicer}]")
        
        extracted_batch = image[start_slicer:end_slicer]
        
        print(f"ExtractFrames: 提取后批次大小: {extracted_batch.shape[0]} 帧")

        # 5. 返回结果
        # 必须返回一个元组 (tuple)
        return (extracted_batch,)

# 注册节点到 ComfyUI
NODE_CLASS_MAPPINGS = {
    "ExtractFramesFromBatch": ExtractFramesFromBatch
}

# 给节点一个在界面上显示的好看名字
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractFramesFromBatch": "CK Extract Frames from Batch"
}
