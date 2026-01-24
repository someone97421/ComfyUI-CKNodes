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
    # 输出定义：图像、音频、帧率、视频路径、视频URL、调试信息
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "video_path", "video_url", "response")

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
                ], {"default": "veo3.1-fast"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "apikey": ("STRING", {"default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "enable_upsample": ("BOOLEAN", {"default": True}),
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
        fps = 24.0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 24.0
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32) / 255.0)
        finally:
            cap.release()

        if not frames:
            return torch.zeros((1, 64, 64, 3)), fps
            
        return torch.from_numpy(np.stack(frames)), float(fps)

    # 修改点：增强版音频加载，解决 MP4 无声问题
    def load_audio(self, path):
        # 1. 尝试 PyAV (最稳妥，支持 MP4)
        try:
            import av
            container = av.open(path)
            stream = container.streams.audio[0] # 获取第一个音频流
            
            resampler = av.audio.resampler.AudioResampler(format='flt', layout='stereo')
            
            audio_data = []
            for frame in container.decode(stream):
                # 重采样并转换为 numpy
                frame.pts = None
                for new_frame in resampler.resample(frame):
                    audio_data.append(new_frame.to_ndarray())

            if audio_data:
                # 拼接所有帧
                audio_np = np.concatenate(audio_data, axis=1)
                return {
                    "waveform": torch.from_numpy(audio_np).unsqueeze(0),
                    "sample_rate": stream.rate
                }
        except Exception as e:
            print(f"👻 [CK_Node] PyAV loading failed: {e}")

        # 2. 回退到 Torchaudio (有些环境可能没有 pyav)
        try:
            waveform, sr = torchaudio.load(path)
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sr}
        except Exception as e:
            print(f"👻 [CK_Node] Torchaudio also failed: {e}")
            
        return None

    # ================= 核心逻辑 =================
    def generate_video(
        self, prompt, model, enhance_prompt, aspect_ratio,
        apikey="", image1=None, image2=None, image3=None,
        seed=0, enable_upsample=False, save_path=""
    ):
        empty_img = torch.zeros((1, 64, 64, 3))
        empty_audio = None
        empty_fps = 0.0

        def error(msg):
            return {
                "ui": {"text": [msg]},
                "result": (empty_img, empty_audio, empty_fps, "", "", msg)
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

        try:
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
        except Exception as e:
            return error(f"Request failed: {str(e)}")

        pbar.update_absolute(30)

        video_url = None
        for _ in range(150):
            time.sleep(2)
            try:
                s = requests.get(
                    f"{baseurl}/v2/videos/generations/{task_id}",
                    headers=self.get_headers()
                ).json()
                if s.get("status") == "SUCCESS":
                    video_url = s["data"]["output"]
                    break
                if s.get("status") == "FAILURE":
                    return error(s.get("fail_reason", "Generation failed"))
            except:
                continue

        if not video_url:
            return error("Generation timeout")

        filename = f"veo3_{task_id}_{int(time.time())}.mp4"
        comfy_path = os.path.join(self.output_dir, filename)

        try:
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                with open(comfy_path, "wb") as f:
                    for c in r.iter_content(8192):
                        f.write(c)
        except Exception as e:
             return error(f"Download failed: {str(e)}")

        final_path = comfy_path
        if save_path.strip():
            try:
                os.makedirs(save_path, exist_ok=True)
                user_path = os.path.join(save_path, filename)
                shutil.copy(comfy_path, user_path)
                final_path = user_path
            except Exception as e:
                print(f"Warning: Failed to copy to custom path: {e}")

        # 解码所有帧和 FPS
        images_out, fps_out = self.load_video_frames(comfy_path)
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
                fps_out,
                final_path,
                video_url,
                json.dumps({
                    "task_id": task_id,
                    "video_url": video_url,
                    "local_path": final_path,
                    "fps": fps_out
                }, ensure_ascii=False)
            )
        }

# ================= 注册 =================
NODE_CLASS_MAPPINGS = {
    "CK_Googel_Veo3": CK_Googel_Veo3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CK_Googel_Veo3": "👻 Google Veo3 (Plus)"
}

