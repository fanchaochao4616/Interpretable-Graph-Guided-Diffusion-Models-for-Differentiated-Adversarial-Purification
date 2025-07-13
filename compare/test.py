import os
from types import SimpleNamespace
from compare.empirical import Empirical
import sys
from compare.util import *
def setup_logging(logs_dir, config):
    """创建带参数标识的日志文件"""
    os.makedirs(logs_dir, exist_ok=True)
    log_name = (f"_{config.compare.dataset}_{config.compare.attack_method}.log")
    log_path = os.path.join(logs_dir, log_name)

    log_file = open(log_path, 'a', encoding='utf-8')
    print(f"[INFO] 日志文件创建于: {log_path}", file=sys.stdout)  # 打印到标准输出
    return log_file, log_path

def main():
    print(f"\n=== 开始实验:6.17===", file=sys.stdout)  # 同时打印到标准输出

    # 初始化参数
    args = SimpleNamespace(logs='/home/fcc/codefile/HGDM/compare/logs')
    #  加载配置文件
    current_config = load_config('/home/fcc/codefile/HGDM/compare/purify.yml')

    #  创建带参数标识的日志文件
    log_file, log_path = setup_logging(args.logs, current_config)
    print(f"\n=== 开始实验: ===", file=log_file)
    # 运行实验
    empirical = Empirical(args, current_config)
    # empirical.run(log_file)  # 输出到日志文件
    empirical.run()#输出到控制台

if __name__ == "__main__":
    main()