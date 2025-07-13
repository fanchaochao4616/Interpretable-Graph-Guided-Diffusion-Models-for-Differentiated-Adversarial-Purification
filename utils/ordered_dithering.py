import torch
from fastai.vision.all import Normalize

#半色调方法
normer = Normalize.from_stats(0.5, 0.5)
class DifferentiableDither(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, dither_matrix):
        binary_output = (image > dither_matrix).float()
        ctx.save_for_backward(image, dither_matrix)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        image, dither_matrix = ctx.saved_tensors

        temperature = 10.0  # 控制梯度陡峭程度
        diff = image - dither_matrix
        sigmoid_approx = torch.sigmoid(diff * temperature)
        grad_approx = temperature * sigmoid_approx * (1 - sigmoid_approx)

        return grad_output * grad_approx, None

def generate_bayer_matrix(n):
    matrix = torch.tensor([[0.0]])
    for _ in range(n):
        size = matrix.shape[0]
        new_size = size * 2
        new_matrix = torch.zeros((new_size, new_size), dtype=matrix.dtype)
        new_matrix[:size, :size] = 4 * matrix
        new_matrix[:size, size:] = 4 * matrix + 2
        new_matrix[size:, :size] = 4 * matrix + 3
        new_matrix[size:, size:] = 4 * matrix + 1
        matrix = new_matrix
    max_val = 4 ** n - 1
    matrix = matrix / max_val
    return matrix

def ordered_dithering(image, n=3, norm=True):
    if norm: image = normer.decodes(image)
    b, c, h, w = image.shape
    dither_matrix = generate_bayer_matrix(n)
    m = dither_matrix.shape[0]

    repeat_h = (h + m - 1) // m
    repeat_w = (w + m - 1) // m
    tiled_dither = dither_matrix.repeat(repeat_h, repeat_w)

    tiled_dither = tiled_dither[:h, :w].to(image.device).type_as(image)

    tiled_dither = tiled_dither.view(1, 1, h, w)

    halftone = DifferentiableDither.apply(image, tiled_dither)
    if norm: halftone = normer.encodes(halftone)
    return halftone

from torchvision.transforms import GaussianBlur
import torch.nn.functional as F

def apply_wavelet_denoise(image, level=2, threshold=0.1):
    """
    使用 PyTorch 实现小波去噪，保留梯度信息
    Args:
        image: 输入张量 (B,C,H,W)
        level: 分解层数
        threshold: 软阈值系数
    """
    def soft_threshold_torch(coeffs, threshold):
        """软阈值处理"""
        return torch.sign(coeffs) * F.relu(torch.abs(coeffs) - threshold)

    # 初始化结果
    denoised = image.clone()

    # 批量处理
    for _ in range(level):
        # 小波分解 (使用简单的 Haar 小波)
        low = F.avg_pool2d(denoised, kernel_size=2, stride=2)
        high = denoised - F.interpolate(low, scale_factor=2, mode='nearest')

        # 对高频分量应用软阈值
        high = soft_threshold_torch(high, threshold)

        # 小波重构
        denoised = F.interpolate(low, scale_factor=2, mode='nearest') + high

    return denoised


def apply_gaussian_blur(image, kernel_size=5, sigma=1):
    """
    使用 TorchVision 实现高斯模糊
    Args:
        image: 输入张量 (B,C,H,W)
        kernel_size: 高斯核大小
        sigma: 标准差
    """
    blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blur(image)

def apply_freq_filter(image, cutoff_ratio=0.9):
    """
    Suppress high-frequency components in the frequency domain (differentiable).
    Args:
        image: Input tensor (B, C, H, W).
        cutoff_ratio: Cutoff frequency ratio (0-1).
    """
    # Perform 2D FFT
    fft_image = torch.fft.fft2(image, dim=(-2, -1))
    fft_image = torch.fft.fftshift(fft_image)

    # Create a circular low-pass filter mask
    b, c, h, w = image.shape
    radius = int(min(h, w) * cutoff_ratio / 2)
    y = torch.arange(h, device=image.device).view(h, 1).expand(h, w)
    x = torch.arange(w, device=image.device).view(1, w).expand(h, w)
    center = (h // 2, w // 2)
    distance_squared = (x - center[1])**2 + (y - center[0])**2
    mask = (distance_squared <= radius**2).float()  # Differentiable mask

    # Apply the mask to the FFT image
    fft_image = fft_image * mask.unsqueeze(0).unsqueeze(0)

    # Perform inverse FFT
    fft_image = torch.fft.ifftshift(fft_image)
    filtered_image = torch.fft.ifft2(fft_image, dim=(-2, -1)).real

    return filtered_image

