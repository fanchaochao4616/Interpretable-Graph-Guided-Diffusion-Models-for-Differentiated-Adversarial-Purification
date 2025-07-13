from attacks import *
from utils import elapsed_seconds, gen_ll, acc_final_step, raw_to_clf,NormalizedModel,import_data,get_imagenet_ddpm
import torch
import os, sys
from datetime import datetime
import tqdm
import pandas as pd
import torchvision
from pytorch_diffusion.diffusion import Diffusion
from purification.DiffusionDenoise import mask_purify
from networks.wrn_cifar10 import WideResNet
import warnings

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
            dataset=self.config.structure.dataset,
            train=False,
            shuffle=False,
            bsize=self.config.structure.bsize
        )
        print(f"[{datetime.now()}] 数据集加载完成 {self.config.structure.dataset}")

        # ------------------------- 分类模型加载 -------------------------
        print(f"[{datetime.now()}] 开始加载分类器网络")
        transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)

        if self.config.structure.dataset in ["ImageNet", "ImageNet-5k", "ImageNet-Mini"]:
            if self.config.structure.classifier == 'ResNet152':
                network_clf = torchvision.models.resnet152(pretrained=True)
            elif self.config.structure.classifier == 'ResNet50':
                network_clf = torchvision.models.resnet50(pretrained=True)
        elif self.config.structure.dataset == "CIFAR10":
            network_clf = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
            checkpoint = torch.load(self.config.net.clf_path, map_location='cpu')
            network_clf.load_state_dict(checkpoint, strict=True)

        network_clf = NormalizedModel(network_clf, transform_raw_to_clf).to(self.config.device).eval()

        # ------------------------- 扩散模型加载 -------------------------
        print("创建模型和扩散过程...")
        if self.config.structure.dataset == "CIFAR10":
            model_name = 'ema_cifar10'
            diffusion = Diffusion.from_pretrained(model_name, device=self.config.device)
            model = diffusion.model.to(self.config.device).eval()
        else:
            model, diffusion = get_imagenet_ddpm(self.config)
            if self.config.net.use_fp16:
                model.convert_to_fp16()
            model = model.to(self.config.device).eval()

        # ------------------------- 结果记录初始化 -------------------------
        df_columns = ["Epoch", "nData", "att_time", "pur_time", "clf_time",
                      "std_acc", "att_acc", "pur_acc_l", "pur_acc_s", "pur_acc_o",
                      "pur_acc_list_l", "pur_acc_list_s", "pur_acc_list_o",
                      "count_att", "count_diff"]
        df = pd.DataFrame(columns=df_columns)

        # ------------------------- 主运行循环 -------------------------
        for i, (x, y) in enumerate(tqdm.tqdm(testLoader)):
            if i < self.config.structure.start_epoch:
                continue
            if i > self.config.structure.end_epoch:
                break

            start_time = datetime.now()
            x = x.to(self.config.device)
            y = y.long().to(self.config.device)

            # ===================== 对抗攻击生成 =====================
            if self.config.attack.if_attack:
                attack_method = eval(self.config.attack.attack_method)
                x_adv, success, acc = attack_method(
                    x, y, diffusion, network_clf, self.config, model=model
                )
                attack_time = elapsed_seconds(start_time, datetime.now())
                print(f"[{datetime.now()}] 第 {i} 轮次: 攻击 {self.config.structure.bsize} 个数据耗时 {attack_time:.2f} 秒")
            else:
                x_adv = x
                attack_time = 0.0

            # ===================== 对抗样本净化 =====================
            start_time = datetime.now()
            print(f"[{datetime.now()}] 第 {i} 轮次: 开始净化 {self.config.structure.bsize} 个对抗样本")

            x_pur_list_list = [
                mask_purify(
                    config=self.config,
                    network_clf=network_clf,
                    model=model,
                    diffusion=diffusion,
                    inputs=x_adv
                )
                for _ in range(self.config.purification.path_number)
            ]
            purify_time = elapsed_seconds(start_time, datetime.now())
            print(f"[{datetime.now()}] 第 {i} 轮次: 净化耗时 {purify_time:.2f} 秒")

            # ===================== 分类性能评估 =====================
            # 对抗样本分类
            with torch.no_grad():
                y_t = network_clf(x.clone().detach())
                y_adv_t = network_clf(x_adv.clone().detach())
                nat_correct = torch.eq(torch.argmax(y_t, dim=1), y).float().sum()
                att_correct = torch.eq(torch.argmax(y_adv_t, dim=1), y).float().sum()
                att_label = torch.argmax(y_adv_t, dim=1).to('cpu').numpy()

            # 净化后对抗样本的分类
            with torch.no_grad():
                start_time = datetime.now()
                print(f"[{datetime.now()}] 第 {i} 轮次:\t开始对 {self.config.structure.bsize} 个净化后的对抗样本进行分类预测")
                att_list_list_dict = gen_ll(x_pur_list_list, network_clf, self.config)
                classify_time = elapsed_seconds(start_time, datetime.now())
                print(f"[{datetime.now()}] 第 {i} 轮次:\t耗时 {classify_time:.2f} 秒完成 {self.config.structure.bsize} 个净化对抗样本的分类预测")

            # ===================== 误分类统计 =====================
            att_acc, att_acc_iter, cls_label = acc_final_step(att_list_list_dict, y)

            # ===================== 记录结果 =====================
            count_att, count_diff = 0, 0
            for pred, label, adv_label in zip(cls_label, y, att_label):
                if pred != label:
                    if pred == adv_label:
                        count_att += 1
                    else:
                        count_diff += 1

            new_row = {
                "Epoch": i + 1,
                "nData": self.config.structure.bsize,
                "att_time": attack_time,
                "pur_time": purify_time,
                "clf_time": classify_time,
                "std_acc": nat_correct.to('cpu').numpy(),
                "att_acc": att_correct.to('cpu').numpy(),
                "pur_acc_l": att_acc["logit"],
                "pur_acc_s": att_acc["softmax"],
                "pur_acc_o": att_acc["onehot"],
                "pur_acc_list_l": att_acc_iter["logit"],
                "pur_acc_list_s": att_acc_iter["softmax"],
                "pur_acc_list_o": att_acc_iter["onehot"],
                "count_att": count_att,
                "count_diff": count_diff
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # ------------------------- 打印总体结果 -------------------------
        total_samples = df["nData"].sum()
        print(f"\n总体结果:")
        print(f"干净样本准确率: {df['std_acc'].sum() / total_samples:.4f}")
        print(f"攻击成功率: {(total_samples - df['att_acc'].sum()) / total_samples:.4f}")
        print(f"净化后准确率logit: {df['pur_acc_l'].sum() / total_samples:.4f}")
        print(f"净化后准确率softmax: {df['pur_acc_s'].sum() / total_samples:.4f}")
        print(f"净化后准确率onehot: {df['pur_acc_o'].sum() / total_samples:.4f}")
        print(f"净化扰动残留率: {df['count_att'].sum() / total_samples:.4f}")
        print(f"净化失真率: {df['count_diff'].sum() / total_samples:.4f}")

        # ------------------------- 保存结果 -------------------------
        df.to_csv(os.path.join(self.args.logs, f"paper_{self.config.attack.paper}_result_{self.config.structure.dataset}"
                                               f"_{self.config.attack.attack_method}_scale{self.config.purification.guide_scale}"
                                               f"_mask_step{self.config.purification.mask_step}"
                                               f"_purify_step{self.config.purification.purify_step}"
                                               f"_{self.config.purification.guide_mode}_{self.config.rank}.csv"))