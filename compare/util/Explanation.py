from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from captum.attr import IntegratedGradients, GradientShap, Saliency
import torch.nn.functional as F
import torch

#生成robust cam
def generate_robust_cam(
    model,
    input_tensor,
    target_layer,
    k=5,
    normalize_fn=None,
    device=None,
):
    normalized_images = normalize_fn(input_tensor) if normalize_fn else input_tensor
    device = device or next(model.parameters()).device
    # 初始化GradCAM
    grad_cam = GradCAM(
        model=model.eval(),
        target_layers=[target_layer],
    )

    # 获取top-k预测
    with torch.no_grad():
        logits = model(normalized_images.to(device))
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_indices = probs.topk(k, dim=1)

    # 生成多类别CAM并加权平均
    batch_size, _, h, w = normalized_images.shape
    rcam = torch.zeros(batch_size, 1, h, w, device=device)

    for i in range(k):
        targets = [ClassifierOutputTarget(idx.item()) for idx in topk_indices[:, i]]
        cam = grad_cam(input_tensor=normalized_images, targets=targets)
        cam = torch.tensor(cam, device=device).unsqueeze(1)  # [B,1,H,W]
        rcam += F.relu(cam)  # 只保留正向激活
    # 归一化到[0,1]
    rcam = rcam / rcam.flatten(2).max(dim=2)[0].view(-1,1,1,1)
    return rcam

class Interpreter:
    def __init__(self, model):
        self.model = model
        self.IntegratedGradients = IntegratedGradients(model)
        self.GradientShap = GradientShap(model)
        self.Saliency = Saliency(model)
    @staticmethod
    def interpret(type, model, inputs, targets):
        # 创建解释器实例
        interpreter = Interpreter(model)
        if type == 'IntegratedGradients':
            interpretation = interpreter.IntegratedGradients.attribute(inputs=inputs, target=targets, n_steps=20)
        elif type == 'GradientShap':
            baselines = torch.zeros_like(inputs)  # 基准为全零张量
            interpretation = interpreter.GradientShap.attribute(inputs=inputs, baselines=baselines, n_samples=5, target=targets)
        elif type == 'Saliency':
            interpretation = interpreter.Saliency.attribute(inputs=inputs, target=targets, abs=False)
        return interpretation

# 生成解释图
def generate_interpretations(model, images, interpreter_names, normalize_fn=None, device=None):
    # 设备初始化
    model.eval()
    device = device or next(model.parameters()).device
    images = images.to(device)
    # 归一化处理
    normalized_images = normalize_fn(images) if normalize_fn else images
    # 获取预测类别
    with torch.no_grad():
        outputs = model(normalized_images)
        _, predicted = torch.max(outputs, 1)
    # 生成解释图
    interpretations = []
    for name in interpreter_names:
        # 核心解释逻辑
        interpretation = Interpreter.interpret(
            name, model, normalized_images, predicted
        ).float()
        # 非负截断处理
        # interpretation = torch.where(
        #     interpretation > 0,
        #     interpretation,
        #     torch.tensor(0.0, device=device)
        # )
        interpretations.append(interpretation)
    return torch.stack(interpretations, dim=0).to(device)  # [N,B,C,H,W]

