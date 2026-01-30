import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import urllib.request
import urllib.error

# 2025-01-30 Final Fix: 
# 1. 强制将 seed 限制在 Signed 32-bit Integer (2147483647) 范围内，
#    彻底解决 "seed must be Integer" 的越界报错。
# 2. 增加了 int() 强制类型转换，确保序列化不出错。

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
                    "tooltip": "API地址。例如 https://api.openai.com/v1"
                }),
                "api_key": ("STRING", {
                    "default": "sk-...", 
                    "multiline": False,
                    "tooltip": "Bearer Token / API Key"
                }),
                "model_name": ("STRING", {
                    "default": "gpt-4o", 
                    "multiline": False,
                    "tooltip": "模型名称 (model)"
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
                # 界面上也限制为 Signed 32-bit Max
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}), 
            },
            "optional": {
                "images": ("IMAGE", ), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response_text",)
    FUNCTION = "generate_completion"
    CATEGORY = "👻CKNodes"

    def tensor_to_base64(self, image_tensor):
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def generate_completion(self, api_url, api_key, model_name, system_prompt, user_prompt, temperature, max_tokens, seed, images=None):
        
        # --- 1. URL 智能处理 ---
        endpoint = api_url.strip()
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/chat/completions"

        print(f"\033[36m[API Node] Target URL: {endpoint}\033[0m")

        # --- 2. 构建消息 ---
        content_list = [{"type": "text", "text": user_prompt}]

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

        # --- 关键修复：Seed 安全处理 ---
        # 即使输入很大的数，也强制取模，确保落在 0 - 2147483647 之间
        safe_seed = int(seed) % 2147483647

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False, 
            "seed": safe_seed 
        }
        
        data = json.dumps(payload).encode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "ComfyUI_Client/1.0"
        }

        # --- 3. 发送与调试 ---
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as response:
                response_body = response.read().decode('utf-8')
                
                # 尝试解析 JSON
                try:
                    json_response = json.loads(response_body)
                except json.JSONDecodeError:
                    print("\033[31m[API Node Warning] Response is NOT JSON.\033[0m")
                    return (f"API Error: Response is not JSON. Raw content:\n{response_body}",)
                
                # 解析标准结构
                if "choices" in json_response and len(json_response["choices"]) > 0:
                    choice = json_response["choices"][0]
                    if "message" in choice:
                        return (choice["message"].get("content", ""),)
                    if "delta" in choice:
                        return (choice["delta"].get("content", ""),)
                
                return (f"API Response (Unparsed): {str(json_response)}",)

        except urllib.error.HTTPError as e:
            error_content = e.read().decode('utf-8')
            print(f"\033[31m[API Node HTTP Error] {e.code}: {error_content}\033[0m")
            return (f"HTTP Error {e.code}: {error_content}",)
            
        except Exception as e:
            print(f"\033[31m[API Node Error] {str(e)}\033[0m")
            return (f"Connection Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "SimpleOpenAI_LLM": SimpleOpenAI_LLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleOpenAI_LLM": "👻简单LLM助手👻"
}
