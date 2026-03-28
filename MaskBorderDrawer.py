import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2


class MaskBorderDrawer:
    """
    根据遮罩范围在图像上绘制线框边框描边
    支持方框和圆框，可调节粗细、颜色、范围扩展
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入原图像"}),
                "mask": ("MASK", {"tooltip": "输入遮罩，用于确定绘制边框的范围"}),
                "stroke_width": ("INT", {
                    "default": 3, 
                    "min": 1, 
                    "max": 50, 
                    "step": 1,
                    "tooltip": "边框粗细（像素）"
                }),
                "box_type": (["rectangle", "rounded"], {
                    "default": "rectangle",
                    "tooltip": "边框类型：rectangle=方框，rounded=圆框"
                }),
                "expand_pixels": ("INT", {
                    "default": 0, 
                    "min": -100, 
                    "max": 500, 
                    "step": 1,
                    "tooltip": "范围扩展像素数（可为负数缩小范围）"
                }),
                "red": ("INT", {
                    "default": 255, 
                    "min": 0, 
                    "max": 255, 
                    "step": 1,
                    "tooltip": "边框颜色 - 红色通道"
                }),
                "green": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 255, 
                    "step": 1,
                    "tooltip": "边框颜色 - 绿色通道"
                }),
                "blue": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 255, 
                    "step": 1,
                    "tooltip": "边框颜色 - 蓝色通道"
                }),
                "corner_radius": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 200, 
                    "step": 1,
                    "tooltip": "圆框的圆角半径（仅在box_type为rounded时有效）"
                }),
                "draw_mode": (["separate", "combined"], {
                    "default": "separate",
                    "tooltip": "绘制模式：separate=分别为每个不连通区域绘制边框，combined=合并所有区域绘制一个整体边框"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_border",)
    FUNCTION = "draw_border"
    CATEGORY = "👻CKNodes"
    DESCRIPTION = "根据遮罩范围在图像上绘制线框边框描边"

    def get_mask_bounds(self, mask, expand_pixels=0, mode="separate"):
        """
        获取遮罩的边界框，并支持扩展
        mode: "separate" 返回多个边界框列表，"combined" 返回合并后的单个边界框
        返回: [(x1, y1, x2, y2), ...] 列表 或 None（如果没有有效区域）
        """
        # 确保mask是numpy数组
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # 处理batch维度，取第一个
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # 二值化mask
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        height, width = binary_mask.shape
        bounds_list = []
        
        if mode == "combined":
            # 合并所有轮廓，找到整体边界框
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            
            # 应用扩展
            x1 = max(0, x - expand_pixels)
            y1 = max(0, y - expand_pixels)
            x2 = min(width, x + w + expand_pixels)
            y2 = min(height, y + h + expand_pixels)
            
            bounds_list.append((x1, y1, x2, y2))
        else:
            # 分别为每个轮廓计算边界框
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 应用扩展
                x1 = max(0, x - expand_pixels)
                y1 = max(0, y - expand_pixels)
                x2 = min(width, x + w + expand_pixels)
                y2 = min(height, y + h + expand_pixels)
                
                # 过滤掉太小的区域
                if x2 > x1 and y2 > y1:
                    bounds_list.append((x1, y1, x2, y2))
        
        return bounds_list if bounds_list else None

    def draw_border(self, image, mask, stroke_width, box_type, expand_pixels, red, green, blue, corner_radius, draw_mode):
        """
        在图像上绘制边框
        """
        # 转换tensor到numpy
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # 处理batch，假设batch size为1
        if len(image_np.shape) == 4:
            image_np = image_np[0]
        
        # 转换为PIL Image (0-255范围)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 获取图像尺寸
        height, width = image_np.shape[:2]
        
        # 创建PIL Image
        pil_image = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_image)
        
        # 获取边界框列表
        bounds_list = self.get_mask_bounds(mask, expand_pixels, draw_mode)
        
        if bounds_list is None:
            # 如果没有找到遮罩区域，返回原图
            print("[MaskBorderDrawer] Warning: No mask region found, returning original image")
            result = np.array(pil_image).astype(np.float32) / 255.0
            result = torch.from_numpy(result).unsqueeze(0)
            return (result,)
        
        # 边框颜色
        stroke_color = (red, green, blue)
        
        # 遍历所有边界框并绘制
        for bounds in bounds_list:
            x1, y1, x2, y2 = bounds
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # 确保x1 < x2, y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 跳过太小的区域
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            
            # 绘制边框
            if box_type == "rectangle":
                # 绘制方框
                # 使用多线方式绘制指定粗细的边框
                for i in range(stroke_width):
                    offset = i - stroke_width // 2
                    draw.rectangle(
                        [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                        outline=stroke_color
                    )
            else:
                # 绘制圆框（圆角矩形）
                # 计算圆角半径，不能超过宽高的一半
                max_radius = min((x2 - x1) // 2, (y2 - y1) // 2)
                radius = min(corner_radius, max_radius)
                
                # PIL的rounded_rectangle方法需要较新版本
                try:
                    draw.rounded_rectangle(
                        [x1, y1, x2, y2],
                        radius=radius,
                        outline=stroke_color,
                        width=stroke_width
                    )
                except AttributeError:
                    # 如果PIL版本不支持rounded_rectangle，手动绘制
                    self._draw_rounded_rectangle(draw, x1, y1, x2, y2, radius, stroke_color, stroke_width)
        
        # 转换回tensor格式
        result_np = np.array(pil_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        
        return (result_tensor,)

    def _draw_rounded_rectangle(self, draw, x1, y1, x2, y2, radius, color, width):
        """
        手动绘制圆角矩形（兼容旧版本PIL）
        """
        # 绘制四条直线边
        for i in range(width):
            offset = i - width // 2
            # 上边
            draw.line([(x1 + radius, y1 + offset), (x2 - radius, y1 + offset)], fill=color, width=1)
            # 下边
            draw.line([(x1 + radius, y2 - offset), (x2 - radius, y2 - offset)], fill=color, width=1)
            # 左边
            draw.line([(x1 + offset, y1 + radius), (x1 + offset, y2 - radius)], fill=color, width=1)
            # 右边
            draw.line([(x2 - offset, y1 + radius), (x2 - offset, y2 - radius)], fill=color, width=1)
        
        # 绘制四个圆角
        # 左上角
        draw.arc([x1, y1, x1 + radius * 2, y1 + radius * 2], 180, 270, fill=color, width=width)
        # 右上角
        draw.arc([x2 - radius * 2, y1, x2, y1 + radius * 2], 270, 360, fill=color, width=width)
        # 左下角
        draw.arc([x1, y2 - radius * 2, x1 + radius * 2, y2], 90, 180, fill=color, width=width)
        # 右下角
        draw.arc([x2 - radius * 2, y2 - radius * 2, x2, y2], 0, 90, fill=color, width=width)


NODE_CLASS_MAPPINGS = {
    "MaskBorderDrawer": MaskBorderDrawer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBorderDrawer": "👻Mask Border Drawer👻"
}
