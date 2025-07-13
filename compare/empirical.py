from compare.util import elapsed_seconds, raw_to_clf,import_data,get_imagenet_ddpm
import torch
import os, sys
from datetime import datetime
import tqdm
import pandas as pd
import torchvision
from pytorch_diffusion.diffusion import Diffusion
from networks.wrn_cifar10 import WideResNet
import warnings
from compare.purification import PurificationForward
from compare.attacks.pgd_eot import PGD
from compare.attacks.pgd_eot_l2 import PGDL2
from compare.attacks.pgd_eot_bpda import BPDA
from compare.attacks.aa_eot_l2 import AutoAttackL2
from compare.attacks.aa_eot_linf import AutoAttackLinf
from compare.attacks.pgd_eot_bpda import BPDA

warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ['Empirical']


class Empirical:
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def run(self, log_progress=sys.stdout):
        """
        主运行流程：
        数据加载 -> 模型构建 -> 攻击生成 -> 净化 -> 评估 -> 结果记录
        """
        # 初始化日志输出
        sys.stdout = log_progress

        # ------------------------- 数据加载 -------------------------
        testLoader = import_data(
            dataset=self.config.compare.dataset,
            train=False,
            shuffle=False,
            bsize=self.config.compare.bsize
        )
        print(f"[{datetime.now()}] 数据集加载完成 {self.config.compare.dataset}")

        # ------------------------- 分类模型加载 -------------------------
        print(f"[{datetime.now()}] 开始加载分类器网络")
        transform_raw_to_clf = raw_to_clf(self.config.compare.dataset)

        if self.config.compare.dataset in ["ImageNet", "ImageNet-5k", "ImageNet-Mini"]:
            if self.config.compare.classifier == 'ResNet152':
                clf = torchvision.models.resnet152(pretrained=True)
            elif self.config.compare.classifier == 'ResNet50':
                clf = torchvision.models.resnet50(pretrained=True)
        elif self.config.compare.dataset == "CIFAR10":
            clf = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
            checkpoint = torch.load(self.config.compare.clf_path, map_location='cpu')
            clf.load_state_dict(checkpoint, strict=True)
        clf= clf.to(self.config.device).eval()

        # ------------------------- 扩散模型加载 -------------------------
        print("创建模型和扩散过程...")
        if self.config.compare.dataset == "CIFAR10":
            model_name = 'ema_cifar10'
            diffusion = Diffusion.from_pretrained(model_name, device=self.config.device)
            model = diffusion.model.to(self.config.device).eval()
        else:
            model, diffusion = get_imagenet_ddpm(self.config)
            if self.config.compare.use_fp16:
                model.convert_to_fp16()
            model = model.to(self.config.device).eval()

        # ------------------------- 扩散模型参数设置 -------------------------
        attack_forward = PurificationForward(
            clf=clf, diffusion=model, strength_a=self.config.compare.strength_l, strength_b=self.config.compare.strength_s,
            classifier_name=self.config.compare.classifier, is_imagenet=self.config.compare.is_imagenet,
            threshold=self.config.compare.threshold, threshold_percent=self.config.compare.threshold_percent,
            ddim_steps=self.config.compare.attack_ddim_steps, forward_noise_steps=self.config.compare.forward_noise_steps, device=self.config.device)
        defense_forward = PurificationForward(
            clf=clf, diffusion=model, strength_a=self.config.compare.strength_s, strength_b=self.config.compare.strength_s,
            classifier_name=self.config.compare.classifier, is_imagenet=self.config.compare.is_imagenet,
            threshold=self.config.compare.threshold, threshold_percent=self.config.compare.threshold_percent,
            ddim_steps=self.config.compare.defense_ddim_steps, forward_noise_steps=self.config.compare.forward_noise_steps, device=self.config.device)

        total_samples=0
        nat_correct=0
        att_correct=0

        # ------------------------- 主运行循环 -------------------------
        for i, (x, y) in enumerate(tqdm.tqdm(testLoader)):
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            total_samples += x.size(0)
            x=transform_raw_to_clf(x)
            if i < self.config.start_epoch:
                continue
            if i > self.config.end_epoch:
                break

            start_time = datetime.now()
            x = x.to(self.config.device)
            y = y.long().to(self.config.device)

            # ===================== 对抗攻击生成 =====================
            if self.config.compare.if_attack:
                if self.config.compare.dataset == 'cifar10':
                    print('[Dataset] CIFAR-10')
                    if self.config.compare.attack_method == 'pgd':  # PGD Linf
                        eps = 8. / 255.
                        attack = PGD(attack_forward, attack_steps=self.config.compare.n_iter,
                                     eps=eps, step_size=0.007, eot=self.config.compare.eot)
                        print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                            self.config.compare.n_iter, eps, self.config.compare.eot))
                    elif self.config.compare.attack_method == 'pgd_l2':  # PGD L2
                        eps = 0.5
                        attack = PGDL2(attack_forward, attack_steps=self.config.compare.n_iter,
                                       eps=eps, step_size=0.007, eot=self.config.compare.eot)
                        print('[Attack] PGD L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                            self.config.compare.n_iter, eps, self.config.compare.eot))
                    elif self.config.compare.attack_method == 'bpda':  # BPDA
                        eps = 8. / 255.
                        attack = BPDA(attack_forward, attack_steps=self.config.compare.n_iter,
                                      eps=eps, step_size=0.007, eot=self.config.compare.eot)
                        print('[Attack] BPDA Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                            self.config.compare.n_iter, eps, self.config.compare.eot))
                    if self.config.compare.attack_method == 'aa':
                        eps = 8. / 255.
                        attack = AutoAttackLinf(attack_forward, attack_steps=self.config.compare.n_iter,
                                                eps=eps, step_size=0.007, eot=self.config.compare.eot)
                        print('[Attack] CIFAR10 | AutoAttack Linf | attack_steps: {} | eps: {} | eot: {}'.format(
                            self.config.compare.n_iter, eps, self.config.compare.eot))
                    if self.config.compare.attack_method == 'aa_l2':
                        eps = 8. / 255.
                        attack = AutoAttackL2(attack_forward, attack_steps=self.config.compare.n_iter,
                                              eps=eps, step_size=0.007, eot=self.config.compare.eot)
                        print('[Attack] CIFAR10 | AutoAttack L2 | attack_steps: {} | eps: {} | eot: {}'.format(
                            self.config.compare.n_iter, eps, self.config.compare.eot))
                elif self.config.compare.dataset == 'ImageNet-Mini':
                    print('[Dataset] ImageNet')
                    if self.config.compare.attack_method == 'pgd':  # PGD Linf
                        eps = 8. / 255.
                        attack = PGD(attack_forward, attack_steps=self.config.compare.n_iter,
                                     eps=eps, step_size=0.007, eot=self.config.compare.eot)
                        print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                            self.config.compare.n_iter, eps, self.config.compare.eot))
                elif self.config.compare.dataset == 'svhn':
                    print('[Dataset] SVHN')
                    eps = 8. / 255.
                    attack = PGD(attack_forward, attack_steps=self.config.compare.n_iter,
                                 eps=eps, step_size=0.007, eot=self.config.compare.eot)
                    print('[Attack] PGD Linf | attack_steps: {} | eps: {:.3f} | eot: {}'.format(
                        self.config.compare.n_iter, eps, self.config.compare.eot))

                x_adv = attack(x, y)

                attack_time = elapsed_seconds(start_time, datetime.now())
                print(f"[{datetime.now()}] 第 {i} 轮次: 攻击 {self.config.compare.bsize} 个数据耗时 {attack_time:.2f} 秒")
            else:
                x_adv = x
                attack_time = 0.0

            # ===================== 对抗样本净化 =====================
            start_time = datetime.now()
            print(f"[{datetime.now()}] 第 {i} 轮次: 开始净化 {self.config.compare.bsize} 个对抗样本")

            logits = defense_forward(x)

            purify_time = elapsed_seconds(start_time, datetime.now())
            print(f"[{datetime.now()}] 第 {i} 轮次: 净化耗时 {purify_time:.2f} 秒")

            # ===================== 分类性能评估 =====================
            # 对抗样本分类
            with torch.no_grad():
                y_t = clf(x.clone().detach())
                nat_correct += torch.eq(torch.argmax(y_t, dim=1), y).float().sum()
                att_correct += torch.eq(torch.argmax(logits, dim=1), y).float().sum()
        acc_nat = nat_correct / total_samples
        acc_att = att_correct / total_samples
        print(f"自然样本准确率: {acc_nat:.4f}, 对抗样本准确率: {acc_att:.4f}")

