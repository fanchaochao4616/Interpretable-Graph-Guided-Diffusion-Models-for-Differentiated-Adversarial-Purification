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
    log_name = (f"{config.attack.paper}_{config.structure.dataset}_{config.attack.attack_method}_scale{config.purification.guide_scale}"
                f"_purify_step{config.purification.purify_step}_mask_step{config.purification.mask_step}_{config.purification.guide_mode}_{config.rank}.log")
    log_path = os.path.join(logs_dir, log_name)

    log_file = open(log_path, 'a', encoding='utf-8')
    print(f"[INFO] 日志文件创建于: {log_path}", file=sys.stdout)  # 打印到标准输出
    return log_file, log_path

def main():
    print(f"\n=== 开始实验:6.4 ===", file=sys.stdout)  # 同时打印到标准输出

    # 初始化参数
    args = SimpleNamespace(logs='/home/fcc/codefile/HGDM/logs')
    #  加载配置文件
    current_config = load_config('/home/fcc/codefile/HGDM/configs/ImageNet_Res50_pgd_GDMAP.yml')
    current_config.rank = 4  # 0:干净样本无掩码；1:干净样本有掩码 2:对抗样本无掩码；3:对抗样本有掩码；4:测试

    current_config.purification.guide_mode = "MSE"
    scales = [1000]
    steps = [85, 100, 115, 130]
    for scale in scales:
        for step in steps:
            print(f"\n=== scale={scale} step={step} ===", file=sys.stdout)  # 同时打印到标准输出
            current_config.purification.guide_scale = scale
            current_config.purification.purify_step = step
            #  创建带参数标识的日志文件
            log_file, log_path = setup_logging(args.logs, current_config)
            print(f"\n=== 开始实验: ===", file=log_file)
            # 运行实验
            empirical = Empirical(args, current_config)
            empirical.run(log_file)  # 输出到日志文件
            # empirical.run()#输出到控制台
    current_config.purification.guide_mode = "SSIM"
    for scale in scales:
        for step in steps:
            print(f"\n=== scale={scale} step={step} ===", file=sys.stdout)  # 同时打印到标准输出
            current_config.purification.guide_scale = scale
            current_config.purification.purify_step = step
            #  创建带参数标识的日志文件
            log_file, log_path = setup_logging(args.logs, current_config)
            print(f"\n=== 开始实验: ===", file=log_file)
            # 运行实验
            empirical = Empirical(args, current_config)
            empirical.run(log_file)  # 输出到日志文件
            # empirical.run()#输出到控制台
    current_config.purification.guide_mode = "L2_CONSTANT"
    scales = [0.5, 1, 1.5, 2,]
    for scale in scales:
        for step in steps:
            print(f"\n=== scale={scale} step={step} ===", file=sys.stdout)  # 同时打印到标准输出
            current_config.purification.guide_scale = scale
            current_config.purification.purify_step = step
            #  创建带参数标识的日志文件
            log_file, log_path = setup_logging(args.logs, current_config)
            print(f"\n=== 开始实验: ===", file=log_file)
            # 运行实验
            empirical = Empirical(args, current_config)
            empirical.run(log_file)  # 输出到日志文件
            # empirical.run()#输出到控制台
    current_config.purification.guide_mode = "L2"
    scales = [0.5, 1, 1.5, 2,]
    for scale in scales:
        for step in steps:
            print(f"\n=== scale={scale} step={step} ===", file=sys.stdout)  # 同时打印到标准输出
            current_config.purification.guide_scale = scale
            current_config.purification.purify_step = step
            #  创建带参数标识的日志文件
            log_file, log_path = setup_logging(args.logs, current_config)
            print(f"\n=== 开始实验: ===", file=log_file)
            # 运行实验
            empirical = Empirical(args, current_config)
            empirical.run(log_file)  # 输出到日志文件
            # empirical.run()#输出到控制台

if __name__ == "__main__":
    main()