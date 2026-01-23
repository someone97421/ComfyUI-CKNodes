import os
import json
import requests
import time
import base64
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import cv2
import shutil
import comfy.utils
import folder_paths
import torchaudio # 引入音频处理库

# 基础配置
baseurl = "https://ai.t8star.cn"

# 辅助函数：读取和保存配置
def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        if not os.path.exists(config_path):
            return {}
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# 辅助函数：图像转换
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Google Veo3 节点
class CK_Googel_Veo3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ([
                    "veo3", 
                    "veo3-fast", 
                    "veo3-pro", 
                    "veo3-fast-frames", 
                    "veo3-pro-frames", 
                    "veo3.1",
                    "veo3.1-fast", 
                    "veo3.1-pro", 
                    "veo3.1-components", 
                    "veo3.1-4k", 
                    "veo3.1-pro-4k", 
                    "veo3.1-components-4k"
                ], {"default": "veo3-fast"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "enable_upsample": ("BOOLEAN", {"default": False}),
            }
        }
    
    # --- 关键修改：增加 AUDIO 输出类型 ---
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path", "video_url", "response")
    FUNCTION = "generate_video"
    CATEGORY = "👻CKNodes"

    def __init__(self):
        self.api_key = get_config().get('api_key', '')
        self.timeout = 300
        self.output_dir = folder_paths.get_output_directory()

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def image_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
        pil_image = tensor2pil(image_tensor)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 读取视频画面
    def load_video_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        finally:
            cap.release()
        
        if len(frames) > 0:
            return torch.from_numpy(np.stack(frames))
        return None

    # 新增：读取音频
    def load_audio_track(self, video_path):
        try:
            # 使用 torchaudio 读取音频
            waveform, sample_rate = torchaudio.load(video_path)
            # ComfyUI 标准音频格式
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            return audio
        except Exception as e:
            print(f"Warning: Could not extract audio: {e}")
            return None

    def generate_video(self, prompt, model="veo3", enhance_prompt=False, aspect_ratio="16:9", apikey="", image1=None, image2=None, image3=None, seed=0, enable_upsample=False):
        # 默认空返回值
        empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        empty_audio = None 

        if apikey.strip():
            self.api_key = apikey
            config = get_config()
            config['api_key'] = apikey
            save_config(config)
            
        if not self.api_key:
            return (empty_image, empty_audio, "", "", json.dumps({"code": "error", "message": "API key missing"}))
            
        try:
            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # 构建请求 payload
            has_images = any(img is not None for img in [image1, image2, image3])
            payload = {
                "prompt": prompt,
                "model": model,
                "enhance_prompt": enhance_prompt
            }
            if seed > 0: payload["seed"] = seed
            
            supported_models = [
                "veo3", "veo3-fast", "veo3-pro", 
                "veo3.1", "veo3.1-fast", "veo3.1-pro", "veo3.1-components", 
                "veo3.1-4k", "veo3.1-pro-4k", "veo3.1-components-4k"
            ]
            if model in supported_models:
                if aspect_ratio: payload["aspect_ratio"] = aspect_ratio
                if enable_upsample: payload["enable_upsample"] = enable_upsample

            if has_images:
                images_base64 = []
                for img in [image1, image2, image3]:
                    if img is not None:
                        batch_size = img.shape[0]
                        for i in range(batch_size):
                            single_image = img[i:i+1]
                            image_base64 = self.image_to_base64(single_image)
                            if image_base64:
                                images_base64.append(f"data:image/png;base64,{image_base64}")
                if images_base64: payload["images"] = images_base64
            
            # 发送生成请求
            response = requests.post(f"{baseurl}/v2/videos/generations", headers=self.get_headers(), json=payload, timeout=self.timeout)
            
            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return (empty_image, empty_audio, "", "", json.dumps({"code": "error", "message": error_msg}))
                
            task_id = response.json().get("task_id")
            if not task_id: return (empty_image, empty_audio, "", "", json.dumps({"code": "error", "message": "No task ID"}))
            
            print(f"[Veo3] Task ID: {task_id}")
            pbar.update_absolute(30)
            
            # 轮询状态
            max_attempts = 150
            video_url = None
            for _ in range(max_attempts):
                time.sleep(2)
                try:
                    status_res = requests.get(f"{baseurl}/v2/videos/generations/{task_id}", headers=self.get_headers(), timeout=self.timeout)
                    if status_res.status_code != 200: continue
                    status_data = status_res.json()
                    status = status_data.get("status", "")
                    
                    if status == "SUCCESS":
                        video_url = status_data.get("data", {}).get("output")
                        break
                    elif status == "FAILURE":
                        return (empty_image, empty_audio, "", "", json.dumps({"code": "error", "message": status_data.get("fail_reason")}))
                except: pass
            
            if not video_url:
                return (empty_image, empty_audio, "", "", json.dumps({"code": "error", "message": "Timeout"}))

            pbar.update_absolute(90)
            
            # 下载文件
            filename = f"veo3_{task_id}_{int(time.time())}.mp4"
            local_filepath = os.path.join(self.output_dir, filename)
            
            print(f"Downloading to {local_filepath}...")
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                with open(local_filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # 解码画面
            print("Decoding video frames...")
            out_images = self.load_video_frames(local_filepath)
            if out_images is None: out_images = empty_image

            # 解码音频 (新增)
            print("Extracting audio...")
            out_audio = self.load_audio_track(local_filepath)

            response_data = {"code": "success", "task_id": task_id, "video_url": video_url, "local_path": local_filepath}
            pbar.update_absolute(100)
            
            # 返回：画面, 音频, 路径, URL, 信息
            return (out_images, out_audio, local_filepath, video_url, json.dumps(response_data))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (empty_image, empty_audio, "", "", json.dumps({"code": "error", "message": str(e)}))

NODE_CLASS_MAPPINGS = {
    "CK_Googel_Veo3": CK_Googel_Veo3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CK_Googel_Veo3": "👻Google Veo3 (Plus)👻"
}