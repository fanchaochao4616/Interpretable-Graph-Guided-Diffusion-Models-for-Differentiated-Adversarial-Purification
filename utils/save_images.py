import numpy as np
from PIL import Image
import torch
import os

def save_images(image, save_dir, start_index=0, max_index=np.inf):

    # 检查并转换 Tensor 为 NumPy
    if isinstance(image, torch.Tensor):
        # 确保图像在 CPU 上并转换为 NHWC 格式
        image = image.detach().cpu()
        if image.shape[1] in [1, 3]:  # NCHW 格式 (N, C, H, W)
            image = image.permute(0, 2, 3, 1)  # 转换为 NHWC
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"预期输入为 Tensor 或 NumPy 数组，实际为 {type(image)}")

    # 如果不是 NumPy 数组，转换为 NumPy
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # 验证图像形状和通道数
    if image.ndim != 4:
        raise ValueError(f"预期 4D 图像数组 (N, H, W, C)，实际形状为 {image.shape}")
    if image.shape[3] not in [1, 3]:
        raise ValueError(f"预期 1 或 3 个通道，实际为 {image.shape[3]}")

    # 从 [-1, 1] 归一化到 [0, 255] 并转换为 uint8
    image = (image * 255).clip(0, 255).astype(np.uint8)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存每张图像
    for i in range(image.shape[0]):
        index = start_index + i
        if index < max_index:
            save_single_image(image[i], index, save_dir)

def save_single_image(image, index, save_dir):
    """
    将单张图像保存到指定目录为 PNG 文件。

    参数:
        image: NumPy 数组，形状 (H, W, C)，值范围 [0, 255]，类型 uint8。
        index: 文件命名的索引。
        save_dir: 保存图像的目录。
    """
    try:
        save_name = f"{index}_.png"
        # 处理单通道图像
        if image.shape[2] == 1:
            image = image.squeeze(-1)  # 移除通道维度以支持灰度图像
        Image.fromarray(image).save(os.path.join(save_dir, save_name))
    except Exception as e:
        print(f"保存图像 {index} 失败: {e}")