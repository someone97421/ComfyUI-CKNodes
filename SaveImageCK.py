import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.cli_args import args

class SaveImageCK:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save."}),
                "output_folder": ("STRING", {"default": "output", "tooltip": "The folder to save the images to."}),
            },
            "optional": {
                "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
                "caption": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}),
                # 新增：编码选择下拉菜单
                "encoding": (
                    ["utf-8", "gbk", "utf-16", "ascii", "shift_jis", "latin-1"], 
                    {"default": "utf-8", "tooltip": "The encoding to use for the caption file. Use 'gbk' for legacy Windows software in China."}
                ),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "CK Nodes/Image/Output"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, output_folder, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, caption=None, caption_file_extension=".txt", encoding="utf-8"):
        filename_prefix += self.prefix_append

        # 处理输出路径
        if os.path.isabs(output_folder):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            full_output_folder = output_folder
            _, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_folder, images[0].shape[1], images[0].shape[0])
        else:
            self.output_dir = folder_paths.get_output_directory()
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            base_file_name = f"{filename_with_batch_num}_{counter:05}_"
            file = f"{base_file_name}.png"
            
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            
            # 保存 Caption 文本
            if caption is not None:
                txt_file = base_file_name + caption_file_extension
                file_path = os.path.join(full_output_folder, txt_file)
                try:
                    # 使用传入的 encoding 参数
                    with open(file_path, 'w', encoding=encoding) as f:
                        f.write(caption)
                except UnicodeEncodeError:
                    print(f"[Warning] Failed to encode caption using {encoding}. Falling back to utf-8.")
                    # 如果用户选了 ascii 这种存不了中文的格式导致报错，回退到 utf-8 避免节点崩溃
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

            counter += 1

        return (file,)

NODE_CLASS_MAPPINGS = {
    "SaveImageCK": SaveImageCK
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageCK": "CK Save Image with Caption"

}
