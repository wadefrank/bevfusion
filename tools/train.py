"""
本代码是一个完整的训练过程，用于训练3D物体检测模型。主要包括:
    1.解析配置文件
    2.设置运行环境
    3.构建数据集和模型
    4.进行训练，并在过程中记录日志
"""

# Python标准库
import argparse     # argparse      命令行选项、参数和子命令解析器，该模块可以让人轻松编写用户友好的命令行接口
import copy         # copy          浅层和深层复制操作
import os           # os            多种操作系统接口，该模块提供了一种使用与操作系统相关的功能的便捷式途径（如创建目录、获取系统环境变量等）
import random       # random        生成伪随机数，该模块实现了各种分布的伪随机数生成器
import time         # time          时间的访问和转换，该模块提供了各种与时间相关的函数

import numpy as np
import torch
from mmcv import Config

# torchpack 主要用于分布式计算
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs


from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


# For LiDAR-only detector, please run:
# torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml

def main():
    dist.init()

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    # 加载配置文件
    # 使用mmcv.Config加载配置文件，并通过recursive=True参数递归地加载嵌套的配置。
    # 然后使用opts更新配置，opts存储的是命令行中未被解析的参数。
    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
