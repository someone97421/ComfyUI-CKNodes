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
import torchaudio

# ================= 基础配置 =================
baseurl = "https://ai.t8star.cn"

# ================= 配置读写 =================
def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        if not os.path.exists(config_path):
            return {}
        with open(config_path, 'r', encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
    with open(config_path, 'w', encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# ================= 图像工具 =================
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )

# ================= 节点定义 =================
class CK_Googel_Veo3:

    OUTPUT_NODE = True
    CATEGORY = "👻CKNodes"
    FUNCTION = "generate_video"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path", "video_url", "response")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": ([
                    "veo3", "veo3-fast", "veo3-pro",
                    "veo3.1", "veo3.1-fast", "veo3.1-pro",
                    "veo3.1-components", "veo3.1-4k",
                    "veo3.1-pro-4k", "veo3.1-components-4k"
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
                "save_path": ("STRING", {
                    "default": "",
                    "placeholder": "Example: D:\\VeoVideos"
                }),
            }
        }

    def __init__(self):
        self.api_key = get_config().get("api_key", "")
        self.timeout = 300
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    # ================= 工具方法 =================
    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def image_to_base64(self, image_tensor):
        pil = tensor2pil(image_tensor)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def load_video_frames(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
        finally:
            cap.release()

        if not frames:
            return torch.zeros((1, 64, 64, 3))
        return torch.from_numpy(np.stack(frames))

    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sr}
        except:
            return None

    # ================= 核心逻辑 =================
    def generate_video(
        self, prompt, model, enhance_prompt, aspect_ratio,
        apikey="", image1=None, image2=None, image3=None,
        seed=0, enable_upsample=False, save_path=""
    ):
        empty_img = torch.zeros((1, 64, 64, 3))
        empty_audio = None

        def error(msg):
            return {
                "ui": {"text": [msg]},
                "result": (empty_img, empty_audio, "", "", msg)
            }

        if apikey.strip():
            self.api_key = apikey
            cfg = get_config()
            cfg["api_key"] = apikey
            save_config(cfg)

        if not self.api_key:
            return error("❌ API Key missing")

        pbar = comfy.utils.ProgressBar(100)
        pbar.update_absolute(10)

        payload = {
            "prompt": prompt,
            "model": model,
            "enhance_prompt": enhance_prompt,
            "aspect_ratio": aspect_ratio
        }
        if seed > 0:
            payload["seed"] = seed
        if enable_upsample:
            payload["enable_upsample"] = True

        images = []
        for img in (image1, image2, image3):
            if img is not None:
                for i in range(img.shape[0]):
                    images.append("data:image/png;base64," + self.image_to_base64(img[i:i+1]))
        if images:
            payload["images"] = images

        res = requests.post(
            f"{baseurl}/v2/videos/generations",
            headers=self.get_headers(),
            json=payload,
            timeout=self.timeout
        )
        if res.status_code != 200:
            return error(res.text)

        task_id = res.json().get("task_id")
        if not task_id:
            return error("No task_id returned")

        pbar.update_absolute(30)

        video_url = None
        for _ in range(150):
            time.sleep(2)
            s = requests.get(
                f"{baseurl}/v2/videos/generations/{task_id}",
                headers=self.get_headers()
            ).json()
            if s.get("status") == "SUCCESS":
                video_url = s["data"]["output"]
                break
            if s.get("status") == "FAILURE":
                return error(s.get("fail_reason", "Generation failed"))

        if not video_url:
            return error("Generation timeout")

        filename = f"veo3_{task_id}_{int(time.time())}.mp4"
        comfy_path = os.path.join(self.output_dir, filename)

        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(comfy_path, "wb") as f:
                for c in r.iter_content(8192):
                    f.write(c)

        final_path = comfy_path
        if save_path.strip():
            os.makedirs(save_path, exist_ok=True)
            user_path = os.path.join(save_path, filename)
            shutil.copy(comfy_path, user_path)
            final_path = user_path

        images_out = self.load_video_frames(comfy_path)
        audio_out = self.load_audio(comfy_path)

        pbar.update_absolute(100)

        return {
            "ui": {
                "video": [{
                    "filename": filename,
                    "subfolder": "",
                    "type": self.type
                }]
            },
            "result": (
                images_out,
                audio_out,
                final_path,
                video_url,
                json.dumps({
                    "task_id": task_id,
                    "video_url": video_url,
                    "local_path": final_path
                }, ensure_ascii=False)
            )
        }

# ================= 注册 =================
NODE_CLASS_MAPPINGS = {
    "CK_Googel_Veo3": CK_Googel_Veo3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CK_Googel_Veo3": "👻 Google Veo3 (Video Preview)"
}
