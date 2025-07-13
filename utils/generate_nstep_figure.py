import numpy as np
from PIL import Image
import torch, os
from scipy.fftpack import diff
from utils import *
from attacks import *
import pandas as pd
from purification.diff_purify import *
from pytorch_diffusion.diffusion import Diffusion
import tqdm

def normalize_and_quantize(xt):
    """将图像从 [-1, 1] 归一化并量化为 [0, 255] 的 uint8 格式"""
    if xt.max() > 1 or xt.min() < -1:
        center = (xt.max() + xt.min()) / 2
        xt = xt - center
        xt = xt / (xt.max() - xt.min())
    xt = xt.transpose(1, 2).transpose(2, 3)  # [1, C, H, W] -> [H, W, C]
    xt = xt[0].cpu().detach().numpy()  # [H, W, C]
    xt = (xt + 1) / 2 * 255
    xt = xt.astype(np.uint8)
    return xt

def save_single_xt_reverse(xt_reverse, diff_step, reverse_step, index, save_dir):
    """保存单张去噪图像为 PNG 文件"""
    xt_reverse = normalize_and_quantize(xt_reverse)
    save_name = f"{index}_diff_{diff_step}_rev_{reverse_step}.png"
    Image.fromarray(xt_reverse).save(os.path.join(save_dir, save_name))
    print(f"保存图像: {save_name}")

def save_xt_reverse(xt_reverse, diff_step, reverse_step, start_index, save_dir, y):
    """保存一批去噪图像，按标签组织到子目录"""
    for i in range(xt_reverse.shape[0]):
        index = start_index + i
        xt_reverse_i = xt_reverse[i:i+1]
        label_i = y[i].item() if torch.is_tensor(y) else y[i]
        save_root = os.path.join(save_dir, f"{label_i}")
        os.makedirs(save_root, exist_ok=True)
        save_single_xt_reverse(xt_reverse_i, diff_step, reverse_step, index, save_root)

def diff_reverse_gen(x, diffusion, diff_step, reverse_step, sample_number, start_index, save_dir, y):
    """对输入图像进行扩散和去噪，生成并保存结果"""
    print(f"sample_number: {sample_number}, start_index: {start_index}")
    for j in range(sample_number):
        print(f"生成第 {j+1}/{sample_number} 组")
        xt = diffusion.diffuse_t_steps(x, diff_step)
        xt_reverse = diffusion.denoise(
            xt.shape[0], n_steps=reverse_step, x=xt.to("cuda:0"), 
            curr_step=diff_step, progress_bar=tqdm.tqdm
        )
        print(f"第 {j+1} 次去噪，xt_reverse 均值: {xt_reverse.mean().item()}")
        save_xt_reverse(
            xt_reverse, diff_step, reverse_step, 
            start_index + j * xt_reverse.shape[0], save_dir, y
        )

if __name__ == "__main__":
    diff_step = 60
    reverse_step = 50
    sample_number = 3  # 增加到 3，生成 3 组结果
    batch_size = 100

    # 加载数据 (CIFAR-10)
    testLoader = import_data(dataset="CIFAR10", train=True, shuffle=False, bsize=batch_size)

    # 加载预训练扩散模型
    model_name = 'ema_cifar10'
    diffusion = Diffusion.from_pretrained(model_name, device="cuda:0")

    # 设置保存目录
    save_dir = os.path.join("/home/fcc/codefile/HGDM/generated_samples", f"diff_{diff_step}_rev_{reverse_step}")
    transform_raw_to_diff = raw_to_diff("CIFAR10")

    # 处理每批次
    for i, (x, y) in enumerate(testLoader):
        print(f"Progress [{i}/{len(testLoader)}], 批次大小: {x.shape[0]}")
        x = transform_raw_to_diff(x).to("cuda:0")
        diff_reverse_gen(x, diffusion, diff_step, reverse_step, sample_number, 
                        i * batch_size * sample_number, save_dir, y)