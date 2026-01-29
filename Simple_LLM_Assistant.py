import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import json

# 尝试导入openai库，如果没有安装则报错提示
try:
    from openai import OpenAI
except ImportError:
    print("\033[31m[ComfyUI OpenAI Node] Error: 'openai' library not found. Please run: pip install openai\033[0m")

class SimpleOpenAI_LLM:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://api.openai.com/v1", 
                    "multiline": False,
                    "tooltip": "API接入点 (Base URL). 本地模型可用 http://localhost:11434/v1"
                }),
                "api_key": ("STRING", {
                    "default": "sk-...", 
                    "multiline": False,
                    "tooltip": "你的 API Key. 本地模型随便填即可"
                }),
                "model_name": ("STRING", {
                    "default": "gpt-4o", 
                    "multiline": False,
                    "tooltip": "模型名称, 如 gpt-4o, gpt-4o-mini, llama3, deepseek-chat"
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.", 
                    "multiline": True,
                    "dynamicPrompts": True
                }),
                "user_prompt": ("STRING", {
                    "default": "Describe this image in detail.", 
                    "multiline": True,
                    "dynamicPrompts": True
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 128000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "images": ("IMAGE", ), # 支持图片批次（视频帧）
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    FUNCTION = "generate_completion"
    CATEGORY = "👻CKNodes"

    def tensor_to_base64(self, image_tensor):
        """将ComfyUI的Tensor图片转换为Base64字符串"""
        # Tensor形状: [Batch, Height, Width, Channel] -> 这里的输入是单张 [H, W, C]
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffered = io.BytesIO()
        # 默认保存为JPEG以节省Token，质量设为85
        img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

def generate_completion(self, api_url, api_key, model_name, system_prompt, user_prompt, temperature, max_tokens, seed, images=None):
        
        # 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )

        # 构建消息内容
        content_list = [{"type": "text", "text": user_prompt}]

        # 处理图片输入 (支持 Batch/Video)
        if images is not None:
            batch_size = images.shape[0]
            for i in range(batch_size):
                image_data = self.tensor_to_base64(images[i])
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data,
                        "detail": "auto" 
                    }
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_list}
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed
            )
            
            # --- 修复核心：兼容性处理 ---
            
            # 情况1: 如果返回的是字符串（Raw JSON 或 直接文本）
            if isinstance(response, str):
                # 尝试解析 JSON
                try:
                    response = json.loads(response)
                except:
                    # 如果无法解析JSON，假设它就是最终的文本结果（某些非标API的行为）
                    return (response,)

            # 情况2: 如果是字典 (Dict)，通常发生在使用旧版库或代理时
            if isinstance(response, dict):
                # 使用字典方式取值 ['choices']
                if 'choices' in response and len(response['choices']) > 0:
                    choice = response['choices'][0]
                    # choice 本身也可能是字典或对象
                    if isinstance(choice, dict):
                        result = choice.get('message', {}).get('content', '')
                    else:
                        result = choice.message.content
                    return (result,)
                else:
                    return (f"API Error: Invalid dict response {response}",)

            # 情况3: 标准 OpenAI 对象 (Object)
            if hasattr(response, 'choices') and len(response.choices) > 0:
                result = response.choices[0].message.content
                return (result,)
            
            # 未知情况
            return (f"API Error: Unknown response format: {type(response)}",)
            
        except Exception as e:
            error_msg = f"API Error: {str(e)}"
            print(f"\033[31m{error_msg}\033[0m")
            return (error_msg,)
# 节点映射
NODE_CLASS_MAPPINGS = {
    "SimpleOpenAI_LLM": SimpleOpenAI_LLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleOpenAI_LLM": "👻简单LLM助手-API👻"

}
