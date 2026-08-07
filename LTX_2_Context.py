import torch
import comfy.utils

class LTXVContext_TTP:
    """
    LTX Video Context (Forward)
    视频续接节点：将【上一个视频的结尾】应用到【新视频的开头】
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_video": ("IMAGE",),  # 上一个视频
                "vae": ("VAE",),
                "latent": ("LATENT",),  # 新视频的latent
                "context_latent_frames": ("INT", {
                    "default": 6, 
                    "min": 2, 
                    "max": 20, 
                    "step": 1,
                    "tooltip": "从previous_video结尾提取多少个latent帧作为开头参考 (6 latent帧 ≈ 41原始帧)"
                }),
            },
            "optional": {
                "context_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Context固定强度 (1.0=完全固定，<1.0允许微调)"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply_context"
    CATEGORY = "CK Nodes/Video/LTXV"
    
    def apply_context(self, previous_video, vae, latent, context_latent_frames, context_strength=1.0):
        # 复制 samples 防止修改源数据
        samples = latent["samples"].clone()
        batch, channels, latent_frames, latent_height, latent_width = samples.shape
        
        # --- 1. 处理 Noise Mask ---
        # 如果latent里已经有mask（比如已经被Reverse节点处理过），则继承它
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            # 否则创建全白mask（默认全去噪）
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )

        # --- 2. 获取目标尺寸 ---
        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor
        
        # --- 3. 计算需要提取的原始帧数 (8N+1 逻辑) ---
        required_frames = (context_latent_frames - 1) * 8 + 1
        
        # --- 4. 提取视频结尾 ---
        total_video_frames = previous_video.shape[0]
        start_idx = max(0, total_video_frames - required_frames)
        context_frames = previous_video[start_idx:]
        
        # --- 5. 调整图像尺寸 (如果需要) ---
        if context_frames.shape[1] != target_height or context_frames.shape[2] != target_width:
            pixels = comfy.utils.common_upscale(
                context_frames.movedim(-1, 1), 
                target_width, 
                target_height, 
                "bilinear", 
                "center"
            ).movedim(1, -1)
        else:
            pixels = context_frames
            
        encode_pixels = pixels[:, :, :, :3]
        
        # --- 6. VAE 编码 ---
        context_latent = vae.encode(encode_pixels)
        actual_latent_frames = context_latent.shape[2]
        
        # --- 7. 注入到 Latent 开头 ---
        embed_frames = min(actual_latent_frames, latent_frames)
        samples[:, :, :embed_frames] = context_latent[:, :, :embed_frames]
        
        # --- 8. 设置 Mask (固定开头) ---
        noise_mask[:, :, :embed_frames] = 1.0 - context_strength
        
        return ({"samples": samples, "noise_mask": noise_mask},)


class LTXVContext_Reverse_TTP:
    """
    LTX Video Context (Reverse)
    视频向前延伸节点：将【下一个视频的开头】应用到【新视频的结尾】
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "next_video": ("IMAGE",),  # 下一个视频
                "vae": ("VAE",),
                "latent": ("LATENT",),  # 待处理的latent
                "context_latent_frames": ("INT", {
                    "default": 6, 
                    "min": 2, 
                    "max": 20, 
                    "step": 1,
                    "tooltip": "从next_video开头提取多少个latent帧作为结尾参考"
                }),
            },
            "optional": {
                "context_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Context固定强度"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply_reverse_context"
    CATEGORY = "CK Nodes/Video/LTXV"
    
    def apply_reverse_context(self, next_video, vae, latent, context_latent_frames, context_strength=1.0):
        samples = latent["samples"].clone()
        batch, channels, total_latent_frames, latent_height, latent_width = samples.shape
        
        # --- 1. 处理 Noise Mask ---
        # 继承mask，允许与Forward节点串联
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            noise_mask = torch.ones(
                (batch, 1, total_latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )

        # --- 2. 获取目标尺寸 ---
        _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor
        
        # --- 3. 计算帧数 (8N+1 逻辑) ---
        required_frames = (context_latent_frames - 1) * 8 + 1
        
        # --- 4. 提取视频开头 ---
        available_frames = next_video.shape[0]
        actual_pixel_frames = min(required_frames, available_frames)
        # 取开头 [0 : N]
        context_frames = next_video[:actual_pixel_frames]
        
        # --- 5. 调整图像尺寸 ---
        if context_frames.shape[1] != target_height or context_frames.shape[2] != target_width:
            pixels = comfy.utils.common_upscale(
                context_frames.movedim(-1, 1), 
                target_width, 
                target_height, 
                "bilinear", 
                "center"
            ).movedim(1, -1)
        else:
            pixels = context_frames
            
        encode_pixels = pixels[:, :, :, :3]
        
        # --- 6. VAE 编码 ---
        context_latent = vae.encode(encode_pixels)
        actual_context_len = context_latent.shape[2]
        
        # --- 7. 注入到 Latent 结尾 ---
        embed_frames = min(actual_context_len, total_latent_frames)
        # 使用负索引定位结尾 [-N : ]
        samples[:, :, -embed_frames:] = context_latent[:, :, :embed_frames]
        
        # --- 8. 设置 Mask (固定结尾) ---
        noise_mask[:, :, -embed_frames:] = 1.0 - context_strength
        
        return ({"samples": samples, "noise_mask": noise_mask},)


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "LTXVContext_TTP": LTXVContext_TTP,
    "LTXVContext_Reverse_TTP": LTXVContext_Reverse_TTP
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVContext_TTP": "CK LTXV Context Forward",
    "LTXVContext_Reverse_TTP": "CK LTXV Context Reverse"
}
