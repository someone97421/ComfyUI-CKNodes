import gc
import os
from enum import Enum
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

import folder_paths

# --- Path Setup ---
# 注册 models/prompt_generator 路径，以便 ComfyUI 能找到
folder_paths.add_model_folder_path("prompt_generator", os.path.join(folder_paths.models_dir, "prompt_generator"))

TOOLTIPS = {
    "model_name": "Select a model folder located in 'models/prompt_generator'.",
    "quantization": "Precision vs VRAM. 4-bit is best for low VRAM, FP16 for best quality.",
    "system_prompt": "System instructions defining how the AI should behave (e.g., 'You are a helpful assistant').",
    "user_prompt": "The specific query or instruction for the image/video.",
    "keep_model_loaded": "Keeps the model in VRAM for faster subsequent generations.",
    "seed": "Control randomness.",
}

class Quantization(str, Enum):
    Q4 = "4-bit (VRAM-friendly)"
    Q8 = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Unsupported quantization: {value}")

ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa"]

def get_installed_models():
    """Scan the prompt_generator directory for model folders."""
    model_names = []
    # 获取所有可能的路径（包括用户自定义路径）
    paths = folder_paths.get_folder_paths("prompt_generator")
    
    if not paths:
        # Fallback if folder doesn't exist yet
        base_path = os.path.join(folder_paths.models_dir, "prompt_generator")
        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path)
            except:
                pass
        paths = [base_path]

    for path in paths:
        if not os.path.exists(path):
            continue
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            # 简单的检查：如果是目录且包含 config.json，视为模型
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                model_names.append(item)
    
    if not model_names:
        return ["No models found in models/prompt_generator"]
    return sorted(model_names)

def get_device_info():
    device_type = "cpu"
    recommended = "cpu"
    if torch.cuda.is_available():
        device_type = "nvidia_gpu"
        recommended = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
    return {
        "device_type": device_type,
        "recommended_device": recommended,
    }

def flash_attn_available():
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False

def resolve_attention_mode(mode):
    if mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[QwenVL Local] Flash-Attn requested but unavailable, falling back to SDPA")
        return "sdpa"
    if flash_attn_available():
        return "flash_attention_2"
    return "sdpa"

def quantization_config(quantization):
    if quantization == Quantization.Q4:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return cfg, None
    if quantization == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None
    return None, torch.float16 if torch.cuda.is_available() else torch.float32

class QwenVL_Local_Base:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        print(f"[QwenVL Local] Node initialized on {self.device_info['device_type']}")

    def clear(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_path(self, model_name):
        # Locate the absolute path of the model
        paths = folder_paths.get_folder_paths("prompt_generator")
        for path in paths:
            target = os.path.join(path, model_name)
            if os.path.exists(target):
                return target
        raise FileNotFoundError(f"Model '{model_name}' not found in prompt_generator paths.")

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        device_choice,
        keep_model_loaded,
    ):
        quant = Quantization.from_value(quant_value)
        attn_impl = resolve_attention_mode(attention_mode)
        device = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        
        signature = (model_name, quant.value, attn_impl, device)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            return

        self.clear()
        model_path = self.get_model_path(model_name)
        
        quant_config, dtype = quantization_config(quant)
        
        load_kwargs = {
            "device_map": {"": 0} if device == "cuda" and torch.cuda.is_available() else device,
            "dtype": dtype,
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "trust_remote_code": True, 
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        print(f"[QwenVL Local] Loading from: {model_path}")
        print(f"[QwenVL Local] Settings: {quant.value}, attn={attn_impl}")

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model. Ensure transformers is up to date. Error: {e}")

        # Ensure cache is enabled
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_signature = signature

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @torch.no_grad()
    def generate(
        self,
        system_prompt,
        user_prompt,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
    ):
        # Prepare Conversation
        conversation = []
        
        # Add System Prompt if provided (Text only)
        if system_prompt and system_prompt.strip():
            conversation.append({"role": "system", "content": system_prompt.strip()})
        
        # Build User Content (Multimodal list)
        user_content = []
        
        if image is not None:
            user_content.append({"type": "image", "image": self.tensor_to_pil(image)})
        
        if video is not None:
            frames = [self.tensor_to_pil(frame) for frame in video]
            # Simple frame sampling
            if len(frames) > frame_count:
                idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                frames = [frames[i] for i in idx]
            if frames:
                user_content.append({"type": "video", "video": frames})
        
        # Add User Text
        user_txt = user_prompt.strip() if user_prompt else "Describe this."
        user_content.append({"type": "text", "text": user_txt})
        
        conversation.append({"role": "user", "content": user_content})

        # Apply Template
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Extract Media for Processor manually (FIXED: Added checks for content type)
        images = []
        video_frames = []
        
        for msg in conversation:
            content = msg["content"]
            # Only iterate if content is a list (multimodal message)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            images.append(item["image"])
                        elif item.get("type") == "video":
                            video_frames.extend(item["video"])
        
        videos = [video_frames] if video_frames else None

        # Process Inputs
        processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
        
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }

        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)

        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False

        # Seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        outputs = self.model.generate(**model_inputs, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

class QwenVL_Local_Loader(QwenVL_Local_Base):
    @classmethod
    def INPUT_TYPES(cls):
        model_list = get_installed_models()
        
        return {
            "required": {
                "model_name": (model_list, {"default": model_list[0] if model_list else "", "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant. Describe the content accurately.", 
                    "multiline": True, 
                    "tooltip": TOOLTIPS["system_prompt"]
                }),
                "user_prompt": ("STRING", {
                    "default": "Describe this image in detail.", 
                    "multiline": True, 
                    "tooltip": TOOLTIPS["user_prompt"]
                }),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "attention_mode": (ATTENTION_MODES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/Local"

    def process(
        self,
        model_name,
        quantization,
        system_prompt,
        user_prompt,
        max_tokens,
        temperature,
        top_p,
        keep_model_loaded,
        seed,
        image=None,
        video=None,
        attention_mode="auto"
    ):
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            "auto",
            keep_model_loaded,
        )
        try:
            response = self.generate(
                system_prompt,
                user_prompt,
                image,
                video,
                16, # Default frame count
                max_tokens,
                temperature,
                top_p,
                1,  # Default num_beams
                1.1, # Default repetition_penalty
                seed,
            )
            return (response,)
        finally:
            if not keep_model_loaded:
                self.clear()

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "QwenVL_Local_Loader": QwenVL_Local_Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL_Local_Loader": "👻Qwen3VL (Local Loader)👻",
}