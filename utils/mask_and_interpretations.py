import torch
from utils.Explanation import generate_interpretations, generate_robust_cam
from utils.transforms import *

def generate_mask_and_interpretations(config, inputs, network_clf, model, diffusion, purify_func):
    """生成掩码和解释张量"""
    # inputs: 输入图像张量
    # 返回: mask（掩码张量或 None），interpretations（解释张量或 None）
    mask = None
    interpretations = None
    if config.structure.mask_type in ["rcam", "steps_rcam_interpretation"]:
        denoised_images = purify_func(
            config=config,
            model=model,
            images=inputs,
            diffusion=diffusion,
            interpretations=None,
            mask=None
        )
        rcam = generate_robust_cam(
            network_clf, denoised_images[-1], network_clf.model.layer4,
            normalize_fn=raw_to_clf(config.structure.dataset)
        )
        mask = rcam < config.purification.threshold_focus_min
    if config.structure.mask_type in ["steps_interpretation", "steps_rcam_interpretation"]:
        interpretations = generate_interpretations(
            network_clf, inputs, config.structure.interpreter_names,
            raw_to_clf(config.structure.dataset), config.device
        )
    return mask, interpretations