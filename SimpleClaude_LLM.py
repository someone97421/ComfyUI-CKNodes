import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import urllib.request
import urllib.error
import re  # 新增正则库用于提取文本中的 think 标签

# 2026-03-27 Claude API 版本
# 基于 SimpleOpenAI_LLM 修改，适配 Anthropic Claude API 标准格式
# 保留完整功能：思考机制、种子、温度、max_tokens 等

class SimpleClaude_LLM:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://api.anthropic.com/v1", 
                    "multiline": False,
                    "tooltip": "Claude API地址。例如 https://api.anthropic.com/v1"
                }),
                "api_key": ("STRING", {
                    "default": "sk-ant-...", 
                    "multiline": False,
                    "tooltip": "Anthropic API Key (x-api-key)"
                }),
                "model_name": ("STRING", {
                    "default": "claude-3-5-sonnet-20241022", 
                    "multiline": False,
                    "tooltip": "Claude 模型名称，如 claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-sonnet-4-20250514"
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
                
                # --- 完整的默认值 ---
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 128000}),
                
                "enable_thinking": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "是否开启 Claude 思考过程 (thinking 扩展)"
                }),
                "thinking_length": ("INT", {
                    "default": 1024, 
                    "min": 1, 
                    "max": 128000, 
                    "tooltip": "思考长度预算 (budget_tokens)，Claude 模型最大 32000"
                }),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}), 
            },
            "optional": {
                "images": ("IMAGE", ), 
            }
        }

    # --- 三路输出 ---
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
        return img_str

    def generate_completion(self, api_url, api_key, model_name, system_prompt, user_prompt, temperature, max_tokens, enable_thinking, thinking_length, seed, images=None):
        
        endpoint = api_url.strip()
        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]
        if not endpoint.endswith("/messages"):
            endpoint = endpoint + "/messages"

        print(f"\033[36m[Claude API Node] Target URL: {endpoint}\033[0m")

        # 构建 Claude 格式的消息内容
        content_list = [{"type": "text", "text": user_prompt}]

        if images is not None:
            batch_size = images.shape[0]
            for i in range(batch_size):
                image_base64 = self.tensor_to_base64(images[i])
                # 插入到开头，Claude 图片在前，文本在后
                content_list.insert(i, {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                })

        messages = [
            {"role": "user", "content": content_list}
        ]

        safe_seed = int(seed) % 2147483647

        # Claude API 请求体格式
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "system": system_prompt
        }

        # Claude 3.7+ 支持 thinking 扩展
        if enable_thinking:
            safe_thinking_length = min(thinking_length, max(1, max_tokens - 1))
            # Claude thinking 参数格式
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": safe_thinking_length
            }

        # Claude API 请求头格式
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
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
            
            # 智能回退：如果模型不支持 thinking 参数
            if e.code in [400, 422] and enable_thinking:
                print(f"\033[33m[Claude API Node Warning] Model rejected 'thinking' parameters. Retrying without them...\033[0m")
                payload.pop("thinking", None)
                try:
                    response_body = send_request(payload)
                except urllib.error.HTTPError as e2:
                    error_content2 = e2.read().decode('utf-8')
                    print(f"\033[31m[Claude API Node HTTP Error] {e2.code}: {error_content2}\033[0m")
                    return ("", "", f"HTTP Error {e2.code}: {error_content2}")
                except Exception as e2:
                    return ("", "", f"Connection Error: {str(e2)}")
            else:
                print(f"\033[31m[Claude API Node HTTP Error] {e.code}: {error_content}\033[0m")
                return ("", "", f"HTTP Error {e.code}: {error_content}")
            
        except Exception as e:
            print(f"\033[31m[Claude API Node Error] {str(e)}\033[0m")
            return ("", "", f"Connection Error: {str(e)}")

        # --- 解析三路输出内容 ---
        try:
            json_response = json.loads(response_body)
        except json.JSONDecodeError:
            print("\033[31m[Claude API Node Warning] Response is NOT JSON.\033[0m")
            return ("", "", f"API Error: Response is not JSON. Raw content:\n{response_body}")
        
        final_content = ""
        final_reasoning = ""
        
        # Claude API 响应格式中的 content 是一个数组
        if "content" in json_response and len(json_response["content"]) > 0:
            content_parts = []
            reasoning_parts = []
            
            for item in json_response["content"]:
                if item.get("type") == "text":
                    content_parts.append(item.get("text", ""))
                elif item.get("type") == "thinking":
                    # Claude 的 thinking 类型
                    reasoning_parts.append(item.get("thinking", ""))
            
            final_content = "\n".join(content_parts)
            final_reasoning = "\n".join(reasoning_parts)
            
            # 兼容性处理：如果 API 没给 thinking 类型，而是直接包在了正文里的 <think> 标签中
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
    "SimpleClaude_LLM": SimpleClaude_LLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleClaude_LLM": "👻Claude LLM助手👻"
}
