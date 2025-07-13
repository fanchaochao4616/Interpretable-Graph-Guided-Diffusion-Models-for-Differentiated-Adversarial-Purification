import os
import yaml
import copy
from types import SimpleNamespace
from runners import Empirical  # 假设 Empirical 类已定义
import sys
from utils import *
def setup_logging(logs_dir, config):
    """创建带参数标识的日志文件"""
    os.makedirs(logs_dir, exist_ok=True)
    log_name = (f"{config.structure.dataset}_{config.attack.attack_method}_scale{config.purification.guide_scale}"
                f"_purify_step{config.purification.purify_step}_{config.rank}_{config.purification.guide_mode}.log")
    log_path = os.path.join(logs_dir, log_name)

    log_file = open(log_path, 'a', encoding='utf-8')
    print(f"[INFO] 日志文件创建于: {log_path}", file=sys.stdout)  # 打印到标准输出
    return log_file, log_path

def main():
    # 初始化参数
    # scales = [750,1000, 1250, 1500]
    # purify_steps = [40,50,60]
    #
    # for scale in scales:
    #     for i, purify_step in enumerate(purify_steps):
    #         print(f"\n=== 开始实验: purify_step={purify_step} ===", file=sys.stdout)  # 同时打印到标准输出
    #         args = None
    #         args = SimpleNamespace(logs='/home/fcc/codefile/HGDM/logs')
    #
    #         current_config = load_config('/home/fcc/codefile/HGDM/configs/ImageNet.yml')
    #
    #         current_config.rank = 2  # 0:干净样本无掩码；1:干净样本有掩码 2:对抗样本无掩码；3:对抗样本有掩码
    #         current_config.attack.if_attack = True
    #         current_config.purification.purify_step = purify_step
    #         current_config.purification.st = purify_step
    #         current_config.purification.guide_scale = scale
    #         current_config.purification.guide_mode = "MSE"
    #
    #         log_file, log_path = setup_logging(args.logs, current_config)
    #         print(f"\n=== 开始实验: purify_step={purify_step} ===", file=log_file)
    #
    #         # 运行实验
    #         empirical = Empirical(args, current_config)
    #         empirical.run(log_file)  # 输出到日志文件
    #         # empirical.run()#输出到控制台
    #
    # scales = [ 0.5, 0.75, 1,]
    # purify_steps = [40,50,60]
    # for scale in scales:
    #     for i, purify_step in enumerate(purify_steps):
    #         print(f"\n=== 开始实验: purify_step={purify_step} ===", file=sys.stdout)  # 同时打印到标准输出
    #         args =None
    #         args = SimpleNamespace(logs='/home/fcc/codefile/HGDM/logs')
    #
    #         current_config = load_config('/home/fcc/codefile/HGDM/configs/ImageNet.yml')
    #
    #         current_config.rank = 2 #0:干净样本无掩码；1:干净样本有掩码 2:对抗样本无掩码；3:对抗样本有掩码
    #         current_config.attack.if_attack = True
    #         current_config.purification.purify_step = purify_step
    #         current_config.purification.st = purify_step
    #         current_config.purification.guide_scale = scale
    #         current_config.purification.guide_mode = "L2"
    #
    #         log_file, log_path = setup_logging(args.logs, current_config)
    #         print(f"\n=== 开始实验: purify_step={purify_step} ===", file=log_file)
    #
    #         # 运行实验
    #         empirical = Empirical(args, current_config)
    #         empirical.run(log_file)#输出到日志文件
    #         # empirical.run()#输出到控制台
    #
    scales = [1]
    purify_steps = [55,45,65,75,140]
    for scale in scales:
        for i, purify_step in enumerate(purify_steps):
            print(f"\n=== 开始实验: purify_step={purify_step} ===", file=sys.stdout)  # 同时打印到标准输出
            args = None
            args = SimpleNamespace(logs='/home/fcc/codefile/HGDM/logs')

            current_config = load_config('/home/fcc/codefile/HGDM/configs/ImageNet_Res50_pgd_eot_linf.yml')

            current_config.rank = 2  # 0:干净样本无掩码；1:干净样本有掩码 2:对抗样本无掩码；3:对抗样本有掩码
            current_config.attack.if_attack = True
            current_config.purification.purify_step = purify_step
            current_config.purification.st = purify_step
            current_config.purification.guide_scale = scale
            current_config.purification.guide_mode = "L2_CONSTANT"

            log_file, log_path = setup_logging(args.logs, current_config)
            print(f"\n=== 开始实验: purify_step={purify_step} ===", file=log_file)

            # 运行实验
            empirical = Empirical(args, current_config)
            empirical.run(log_file)  # 输出到日志文件
            # empirical.run()#输出到控制台

    # scales = [1.25, 1.5, 1.75]
    # purify_steps = [35, 45, 55, 65]
    # for scale in scales:
    #     for i, purify_step in enumerate(purify_steps):
    #         print(f"\n=== 开始实验: purify_step={purify_step} ===", file=sys.stdout)  # 同时打印到标准输出
    #         args = None
    #         args = SimpleNamespace(logs='/home/fcc/codefile/HGDM/logs')
    #
    #         current_config = load_config('/home/fcc/codefile/HGDM/configs/ImageNet.yml')
    #
    #         current_config.rank = 2  # 0:干净样本无掩码；1:干净样本有掩码 2:对抗样本无掩码；3:对抗样本有掩码
    #         current_config.attack.if_attack = True
    #         current_config.purification.purify_step = purify_step
    #         current_config.purification.st = purify_step
    #         current_config.purification.guide_scale = scale
    #         current_config.purification.guide_mode = "L2_CONSTANT"
    #
    #         log_file, log_path = setup_logging(args.logs, current_config)
    #         print(f"\n=== 开始实验: purify_step={purify_step} ===", file=log_file)
    #
    #         # 运行实验
    #         empirical = Empirical(args, current_config)
    #         empirical.run(log_file)  # 输出到日志文件
    #         # empirical.run()#输出到控制台




if __name__ == "__main__":
    main()