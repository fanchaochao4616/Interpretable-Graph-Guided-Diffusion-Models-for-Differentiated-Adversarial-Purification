import torch
import torch.nn.functional as F
import pytorch_ssim
from utils import ordered_dithering, raw_to_diff, diff_to_raw,show_images,apply_freq_filter,apply_gaussian_blur,apply_wavelet_denoise

def purify_imagenet(config, model, images, diffusion, interpretations=None, mask=None):#接收raw图，返回raw图
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
    if config.purification.guide_image =="gaussian_blur":
        apply_func = apply_gaussian_blur
    elif config.purification.guide_image == "wavelet_denoise":
        apply_func = apply_wavelet_denoise
    elif config.purification.guide_image == "freq_filter":
        apply_func = apply_freq_filter
    elif config.purification.guide_image == "ordered_dithering":
        apply_func = ordered_dithering

    # show_images(images.clone().detach())
    # show_images(apply_func(images.clone().detach()))
    # 1. 图像预处理
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(images).to(config.device)  # 转换为扩散域
    x_adv = F.interpolate(x_adv, size=(256, 256), mode="bilinear")  # 调整到 256x256
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).float().to(config.device)
    measurement = apply_func(x_adv).to(config.device)

    # 3. 动态步进掩码生成
    def get_step_mask(input: torch.Tensor, percentile: float = 0.8) -> torch.Tensor:
        positive = input[input > 0]
        # 降序排序以获取前 percentile 大的值
        sorted_vals, _ = torch.sort(positive.flatten(), descending=False)
        # 3. 计算索引
        k = int(percentile * len(sorted_vals))
        # 4. 取第k大的值作为阈值
        q_value = sorted_vals[k]
        return input > q_value

    def get_step_masks(input: torch.Tensor, percentile: float = 0.8) -> torch.Tensor:
        """
        对输入张量中的每一个 (B, 3, 224, 224) 块生成动态掩码。

        Args:
            input (torch.Tensor): 输入张量，形状为 (N, B, 3, 224, 224)。
            percentile (float): 分位数阈值，范围 [0, 1]。

        Returns:
            torch.Tensor: 布尔掩码，形状为 (N, B, 3, 224, 224)。
        """
        masks = torch.zeros_like(input[0],dtype=torch.bool)
        for i in range(input.size(0)):  # 遍历 N 维度
            mask = get_step_mask(input[i], percentile)
            masks=masks|mask
        return masks

    # 4. 应用掩码
    def generate_mask(config, step, mask=None, interpretations=None):  # mask中为true的需要纯化，为false的不用纯化
        """
        根据时间步和掩码条件生成步进掩码。
        """
        # 生成对抗区域掩码
        def make_mask(interp, thresh):
            m = get_step_masks(interp, thresh)
            return F.interpolate(m.float(), size=(256, 256), mode="bilinear") > 0

        mask_step = config.purification.mask_step
        purify_step = config.purification.purify_step
        relative_step = torch.tensor(
            ((step - (mask_step - 1)) / (purify_step - (mask_step - 1))),
            device=config.device
        )
        # 统一计算 threshold_adv
        if config.purification.mask_root == "linear":
            threshold_adv = relative_step
        elif config.purification.mask_root == "exponential":
            # 设置默认参数
            exponential_k = config.purification.exponential_k  # 控制衰减速度的系数
            threshold_adv = max(
                (torch.sqrt(1 - alphas_cumprod[step]) - torch.sqrt(1 - alphas_cumprod[mask_step])) /
                (torch.sqrt(1 - alphas_cumprod[purify_step]) - torch.sqrt(1 - alphas_cumprod[mask_step])),
                config.purification.threshold_adv_min
            )
            threshold_adv = 1.0 - torch.exp(-exponential_k * threshold_adv)
        elif config.purification.mask_root == "beta_t":
            threshold_adv = max(
                (torch.sqrt(1 - alphas_cumprod[step]) - torch.sqrt(1 - alphas_cumprod[mask_step])) /
                (torch.sqrt(1 - alphas_cumprod[purify_step]) - torch.sqrt(1 - alphas_cumprod[mask_step])),
                config.purification.threshold_adv_min
            )
        else:
            raise ValueError(f"Unsupported mask_root: {config.purification.mask_root}")

        mask_adv = make_mask(interpretations, threshold_adv)
        threshold_focus = max(threshold_adv, config.purification.threshold_focus_min)
        mask_focus = make_mask(interpretations, threshold_focus)

        # 区分不同输入情况
        if mask is not None and interpretations is not None:#对主体区域进行较小的纯化，非主体区域进行较大纯化
            mask=F.interpolate(mask.float(), size=(256, 256), mode="bilinear") > 0
            step_mask = torch.where(mask, mask_adv, mask_focus)
        elif interpretations is not None:#不区分主体区域和非主体区域
            step_mask = mask_adv
        else:#区分主体区域和非主体区域，不进行步进掩码纯化
            step_mask = F.interpolate(mask.float(), size=(256, 256), mode="bilinear") > 0
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
                selecte = -1 * F.mse_loss(apply_func(x_in), measurement)
                scale = diffusion.compute_scale(x_in, t, config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif guide_mode == 'SSIM':
                selecte = pytorch_ssim.ssim(apply_func(x_in), measurement)
                # selecte = pytorch_ssim.ssim(x_in, x_adv)
                scale = diffusion.compute_scale(x_in, t, config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif guide_mode == 'L2':
                selecte = -1 * torch.linalg.norm(apply_func(x_in) - measurement)
                scale = diffusion.compute_scale(x_in, t, config.attack.ptb * 2 / 255. / 3. / config.purification.guide_scale)
            elif guide_mode == 'L2_CONSTANT':
                selecte = -1 * torch.linalg.norm(apply_func(x_in) - measurement)
                scale = config.purification.guide_scale
            else:
                raise ValueError(f"Unsupported guide_mode: {guide_mode}")
            if config.purification.guide_scale == 0:
                scale = 0
            return torch.autograd.grad(selecte.sum(), x_in)[0] * scale

    purify_step_=torch.ones(x_adv.size(0), dtype=torch.long) * config.purification.purify_step
    purify_step_=purify_step_.to(config.device)
    # 6. 纯化过程
    with torch.set_grad_enabled(config.grad_enabled):# 根据外部需求启用梯度  true  启用梯度  false 不启用梯度
        images = []
        xt_reverse = x_adv # 初始化逆向扩散图像
        for i in range(config.purification.max_iter):
            img = diffusion.q_sample(xt_reverse, purify_step_) # 初始化加噪图像
            sample_fn = diffusion.ddim_sample if config.net.use_ddim else diffusion.p_sample
            for step in reversed(range(config.purification.purify_step)):
                t_ = torch.tensor([step] * img.shape[0], device=config.device, dtype=torch.long)
                out = sample_fn(
                    model=model,
                    x=img,
                    t=t_,
                    cond_fn=cond_fn if config.purification.cond else None
                )
                img = out["sample"]
                # 应用掩码
                if step >= config.purification.mask_step and (mask is not None or interpretations is not None):
                    step_=torch.ones(x_adv.size(0), dtype=torch.long) * step
                    step_=step_.to(config.device)
                    noised_image = diffusion.q_sample(xt_reverse, step_)
                    step_mask = generate_mask(config, step, mask, interpretations)
                    img = torch.where(step_mask, img, noised_image)#mask中为true的需要纯化，为false的不用纯化
            xt_reverse = img
            # 7. 后处理
            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            x_pur = F.interpolate(x_pur, size=(224, 224), mode="bilinear")
            # show_images(x_pur.clone().detach())
            images.append(x_pur)
    return images