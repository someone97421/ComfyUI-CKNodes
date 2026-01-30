import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import urllib.request
import urllib.error

# 这个节点不再依赖 'openai' 库，直接使用 Python 原生库发送请求，
# 完美适配截图中的标准 REST API 格式。

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
                    "tooltip": "API地址。例如 https://api.openai.com/v1 或 https://api.deepseek.com"
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
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def generate_completion(self, api_url, api_key, model_name, system_prompt, user_prompt, temperature, max_tokens, seed, images=None):
        
        # 1. 构建 Endpoint URL
        # 如果用户输入的 URL 不包含 /chat/completions，我们尝试自动补全
        # 截图中的路径是 /vi/chat/completions (可能是OCR识别错误或特殊API)，标准是 /v1/chat/completions
        endpoint = api_url.strip()
        if not endpoint.endswith("/chat/completions"):
            # 处理结尾的斜杠
            if endpoint.endswith("/"):
                endpoint = endpoint + "chat/completions"
            else:
                endpoint = endpoint + "/chat/completions"

        # 2. 构建消息体 (Messages)
        content_list = [{"type": "text", "text": user_prompt}]

        # 处理图片 (Vision)
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

        # 3. 构建 Payload (参考截图格式)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False, # ComfyUI 节点必须等待完整响应，不能流式传输
            "seed": seed
            # 可以在这里添加截图中的其他参数，如 top_p, frequency_penalty 等
        }
        
        data = json.dumps(payload).encode('utf-8')

        # 4. 构建 Headers (参考截图格式)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "ComfyUI_Simple_Client/1.0"
        }

        # 5. 发送请求 (使用 urllib，不依赖 openai 库)
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as response:
                response_body = response.read().decode('utf-8')
                
                # 6. 解析响应 (参考截图 Response 部分)
                # 截图显示的标准响应: {"choices": [{"message": {"content": "..."}}]}
                json_response = json.loads(response_body)
                
                if "choices" in json_response and len(json_response["choices"]) > 0:
                    choice = json_response["choices"][0]
                    # 兼容部分 API 返回 message 或 delta
                    if "message" in choice:
                        content = choice["message"].get("content", "")
                        return (content,)
                    elif "delta" in choice:
                        content = choice["delta"].get("content", "")
                        return (content,)
                    else:
                        return (f"API Error: No 'message' in choice. raw: {str(choice)}",)
                
                # 错误处理：如果 API 返回了错误信息
                if "error" in json_response:
                    return (f"API returned error: {json_response['error']}",)
                    
                return (f"API Error: Unexpected format. Keys found: {list(json_response.keys())}",)

        except urllib.error.HTTPError as e:
            # 读取错误正文
            error_content = e.read().decode('utf-8')
            print(f"\033[31m[API Node Error] Status: {e.code}, Reason: {error_content}\033[0m")
            return (f"HTTP Error {e.code}: {error_content}",)
            
        except Exception as e:
            print(f"\033[31m[API Node Error] {str(e)}\033[0m")
            return (f"Connection Error: {str(e)}",)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SimpleOpenAI_LLM": SimpleOpenAI_LLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleOpenAI_LLM": "👻简单LLM助手👻"
}
