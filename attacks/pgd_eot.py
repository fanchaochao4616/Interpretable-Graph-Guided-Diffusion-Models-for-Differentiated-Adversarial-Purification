from tqdm import trange
from purification import *
from utils import *
def pgd_eot(x, y, diffusion, network_clf, config, model=None):
    attack_steps = config.attack.attack_steps  # 攻击迭代次数
    n_eot = config.attack.n_eot # EOT 采样次数
    step_size = config.attack.attack_lambda # 步长
    eps = config.attack.ptb/ 255.0  # 扰动幅度
    clamp = (0.0, 1.0)  # 输入范围

    # 预处理和设备设置
    transform_raw_to_clf = raw_to_clf(config.structure.dataset)
    x = x.to(config.device)
    y = y.to(config.device)
    x_adv = x.clone().detach()

    def get_logit(inputs):
        # 等价于原代码的净化和分类逻辑
        x_eot = mask_purify(
            config=config,
            network_clf=network_clf,
            model=model,
            diffusion=diffusion,
            inputs=inputs,
        ).to(config.device)
        x_clf = transform_raw_to_clf(x_eot).to(config.device)
        logits = network_clf(x_clf)
        return logits

    # PGD 攻击循环
    for _ in trange(attack_steps):
        grad = torch.zeros_like(x_adv).to(config.device)

        # EOT：多次采样净化过程的梯度
        for _ in range(n_eot):
            x_adv.requires_grad = True
            logits = get_logit(x_adv)
            loss = F.cross_entropy(logits, y, reduction="sum")
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            x_adv = x_adv.detach()

        # 平均梯度并取符号
        grad /= n_eot
        grad = grad.sign()

        # PGD 更新
        x_adv = x_adv + step_size * grad
        x_adv = x + torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, *clamp)

    # 检查攻击是否成功
    x_clf = transform_raw_to_clf(x_adv.clone().detach()).to(config.device)
    success = torch.eq(torch.argmax(network_clf(x_clf), dim=1), y)
    acc = success.float().mean()

    return x_adv, success, acc