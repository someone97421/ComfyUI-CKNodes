import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import urllib.request
import urllib.error
import re  # 新增正则库用于提取文本中的 think 标签

# 2025-01-30 Final Fix: 
# 1. 强制将 seed 限制在 Signed 32-bit Integer 范围内。
# 2. 增加思考机制开关与智能回退。
# 3. 更新默认参数，并将输出拆分为正文、思考部分、原始完整API返回三路。

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
                
                # --- 根据要求修改的默认值 ---
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 128000}),
                
                "enable_thinking": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "是否开启思考过程 (主要针对支持思考机制的模型)"
                }),
                "thinking_length": ("INT", {
                    "default": 1024, 
                    "min": 1, 
                    "max": 128000, 
                    "tooltip": "思考长度预算 (budget_tokens)"
                }),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}), 
            },
            "optional": {
                "images": ("IMAGE", ), 
            }
        }

    # --- 修改为三路输出 ---
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("content", "reasoning", "raw_response")
    FUNCTION = "generate_completion"
    CATEGORY = "👻CKNodes"

    def tensor_to_base64(self, image_tensor):
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def generate_completion(self, api_url, api_key, model_name, system_prompt, user_prompt, temperature, max_tokens, enable_thinking, thinking_length, seed, images=None):
        
        endpoint = api_url.strip()
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/chat/completions"

        print(f"\033[36m[API Node] Target URL: {endpoint}\033[0m")

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

        safe_seed = int(seed) % 2147483647

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False, 
            "seed": safe_seed 
        }

        if enable_thinking:
            safe_thinking_length = min(thinking_length, max(1, max_tokens - 1))
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": safe_thinking_length
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "ComfyUI_Client/1.0"
        }

        def send_request(current_payload):
            data = json.dumps(current_payload).encode('utf-8')
            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req) as response:
                return response.read().decode('utf-8')

        response_body = ""
        try:
            response_body = send_request(payload)
            
        except urllib.error.HTTPError as e:
            error_content = e.read().decode('utf-8')
            
            if e.code in [400, 422] and enable_thinking:
                print(f"\033[33m[API Node Warning] Model rejected 'thinking' parameters. Retrying without them...\033[0m")
                payload.pop("thinking", None)
                try:
                    response_body = send_request(payload)
                except urllib.error.HTTPError as e2:
                    error_content2 = e2.read().decode('utf-8')
                    print(f"\033[31m[API Node HTTP Error] {e2.code}: {error_content2}\033[0m")
                    return ("", "", f"HTTP Error {e2.code}: {error_content2}")
                except Exception as e2:
                    return ("", "", f"Connection Error: {str(e2)}")
            else:
                print(f"\033[31m[API Node HTTP Error] {e.code}: {error_content}\033[0m")
                return ("", "", f"HTTP Error {e.code}: {error_content}")
            
        except Exception as e:
            print(f"\033[31m[API Node Error] {str(e)}\033[0m")
            return ("", "", f"Connection Error: {str(e)}")

        # --- 解析三路输出内容 ---
        try:
            json_response = json.loads(response_body)
        except json.JSONDecodeError:
            print("\033[31m[API Node Warning] Response is NOT JSON.\033[0m")
            return ("", "", f"API Error: Response is not JSON. Raw content:\n{response_body}")
        
        final_content = ""
        final_reasoning = ""
        
        if "choices" in json_response and len(json_response["choices"]) > 0:
            choice = json_response["choices"][0]
            
            # 从 message 或 delta 提取内容
            source = choice.get("message", choice.get("delta", {}))
            
            # 使用 or "" 确保即使返回 None 也能转成空字符串进行操作
            final_content = source.get("content", "") or ""
            final_reasoning = source.get("reasoning_content", "") or ""

            # 兼容性处理：如果 API 没给 reasoning_content 字段，而是直接包在了正文里的 <think> 标签中
            if not final_reasoning and "<think>" in final_content:
                think_match = re.search(r"<think>(.*?)</think>", final_content, flags=re.DOTALL)
                if think_match:
                    # 提取思考内容
                    final_reasoning = think_match.group(1).strip()
                    # 从正文中剥离思考标签
                    final_content = re.sub(r"<think>.*?</think>", "", final_content, flags=re.DOTALL).strip()
                    
            # 去除正文开头可能因为换行残留的空余字符
            final_content = final_content.strip()

        return (final_content, final_reasoning, response_body)

NODE_CLASS_MAPPINGS = {
    "SimpleOpenAI_LLM": SimpleOpenAI_LLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleOpenAI_LLM": "👻简单LLM助手👻"
}
