import torch
import torch.nn.functional as F
import pytorch_ssim
import numpy as np
from tqdm import tqdm
from utils import ordered_dithering, raw_to_diff, diff_to_raw
from pytorch_diffusion.diffusion import denoising_step
def purify_cifar(config, model, images, diffusion, interpretations, mask):#接收raw图，返回raw图
    """
    使用 DDIM 采样对 ImageNet 图像进行纯化，移除对抗性噪声。

    Args:
        config: 配置对象，包含数据集、纯化参数和攻击参数等。
        model: 预训练的扩散模型。
        images: 输入对抗样本，形状为 (batch_size, C, H, W)。
        diffusion: 扩散模型实例，包含 compute_scale 和 ddim_sample_loop 方法。
        interpretations: 解释性掩码，用于动态步进掩码生成。
        mask: 静态掩码，控制纯化区域。

    Returns:
        torch.Tensor: 纯化后的图像，形状为 (batch_size, C, 224, 224)。
    """
    # 1. 图像预处理
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(images).to(config.device)  # 转换为扩散域
    x_adv = F.interpolate(x_adv, size=(256, 256), mode="bilinear")  # 调整到 256x256
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().to(config.device)
    measurement = ordered_dithering(x_adv).to(config.device)#生成指导图
    # 3. 动态步进掩码生成
    def get_step_mask(input, percentile):
        """
        根据输入张量的分位数生成动态掩码。

        Args:
            input (torch.Tensor): 输入张量，形状为 [B, ...] 或 [N, B, ...]。
            percentile (float): 分位数阈值，范围 [0, 1]。

        Returns:
            torch.Tensor: 布尔掩码，形状与输入匹配。
        """
        if input.dim() == 4:
            q_value = torch.quantile(input.flatten(), percentile)
            return input > q_value
        mask = torch.zeros_like(input[0], dtype=torch.bool)
        for i in range(input.size(0)):
            q_value = torch.quantile(input[i].flatten(), percentile)
            mask |= (input[i] > q_value * input[i].max())
        return mask

    # 4. 应用掩码
    def generate_mask(config, step, mask=None, interpretations=None):#mask中为true的需要纯化，为false的不用纯化
        """
        根据时间步和掩码条件生成步进掩码。

        Args:
            t (int): 当前时间步。
            st (int): 强度控制时间步阈值。
            purify_step (int): 纯化起始时间步。
            mask (torch.Tensor, optional): 静态掩码。
            interpretations (torch.Tensor, optional): 解释性掩码。

        Returns:
            torch.Tensor: 步进掩码。
        """

        if mask is not None and interpretations is not None:
            threshold_adv = max(
                (torch.sqrt(1 - alphas_cumprod[step]) - torch.sqrt(1 - alphas_cumprod[config.purification.mask_step])) /
                (torch.sqrt(1 - alphas_cumprod[config.purification.start_step]) - torch.sqrt(1 - alphas_cumprod[config.purification.mask_step])),
                config.purification.threshold_adv_min
            )
            mask_adv = get_step_mask(interpretations, threshold_adv)
            threshold_focus = max(threshold_adv, config.purification.threshold_focus_min)
            mask_focus = get_step_mask(interpretations, threshold_focus)
            step_mask = torch.where(mask, mask_adv, mask_focus)
        elif interpretations is not None:
            threshold_adv = max(
                (torch.sqrt(1 - alphas_cumprod[step]) - torch.sqrt(1 - alphas_cumprod[config.purification.mask_step])) /
                (torch.sqrt(1 - alphas_cumprod[config.purification.start_step]) - torch.sqrt(1 - alphas_cumprod[config.purification.mask_step])),
                config.purification.threshold_adv_min
            )
            step_mask = get_step_mask(interpretations, threshold_adv)
        else:
            step_mask = mask
        return step_mask

    # 5. 条件引导函数
    def cond_fn(x_reverse_t, t):
        """
        计算引导条件的梯度。

        Args:
            x_reverse_t (torch.Tensor): 当前时间步的图像张量，形状为 (batch_size, C, H, W)。
            t (torch.Tensor): 当前时间步，形状为 (batch_size,)。

        Returns:
            torch.Tensor: 引导梯度张量。
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)

            guide_mode = config.purification.guide_mode
            if guide_mode == 'MSE':
                selecte = -1 * F.mse_loss(ordered_dithering(x_in), measurement)
                scale = diffusion.compute_scale(x_in, t, config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif guide_mode == 'SSIM':
                selecte = pytorch_ssim.ssim(ordered_dithering(x_in), measurement)
                scale = diffusion.compute_scale(x_in, t, config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif guide_mode == 'L2':
                selecte = -1 * torch.linalg.norm(ordered_dithering(x_in) - measurement)
                scale = diffusion.compute_scale(x_in, t, config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif guide_mode == 'L2_CONSTANT':
                selecte = -1 * torch.linalg.norm(ordered_dithering(x_in) - measurement)
                scale = config.purification.guide_scale
            else:
                raise ValueError(f"Unsupported guide_mode: {guide_mode}")
            return torch.autograd.grad(selecte.sum(), x_in)[0] * scale

    # 6. 纯化过程
    img = diffusion.q_sample(x_adv, config.purification.start_step)  # 初始化加噪图像

    for step in tqdm(reversed(list(range(config.purification.start_step))), desc="Purification Progress"):
        t_ = torch.tensor([step] * img.shape[0], device=config.device, dtype=torch.long)
        with torch.enable_grad():
            # DDPM 采样
            img = denoising_step(
                x=img,
                t=t_,
                model=model,
                logvar=diffusion.logvar,
                sqrt_recip_alphas_cumprod=diffusion.sqrt_recip_alphas_cumprod,
                sqrt_recipm1_alphas_cumprod=diffusion.sqrt_recipm1_alphas_cumprod,
                posterior_mean_coef1=diffusion.posterior_mean_coef1,
                posterior_mean_coef2=diffusion.posterior_mean_coef2,
                return_pred_xstart=False,
                cond_num=cond_fn if config.purification.cond else None
            )

        # 应用掩码
        if step > config.purification.mask_step and (mask is not None or interpretations is not None):
            noised_image = diffusion.q_sample(x_adv, step)
            step_mask = generate_mask(config, step, mask, interpretations)
            img = torch.where(step_mask, img, noised_image)#mask中为true的需要纯化，为false的不用纯化

    # 7. 后处理
    x_pur = torch.clamp(transform_diff_to_raw(img), 0.0, 1.0)
    return x_pur