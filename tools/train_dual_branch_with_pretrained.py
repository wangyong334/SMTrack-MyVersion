#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import argparse
import torch
import torch.distributed as dist

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_module, configure_omp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", type=int, default=8, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    # 预训练模型相关参数
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="path to pretrained model"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="freeze backbone layers"
    )
    parser.add_argument(
        "--freeze-head",
        action="store_true",
        help="freeze head layers"
    )
    
    return parser


def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        torch.cuda.manual_seed_all(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # 设置预训练模型路径
    if args.pretrained is not None:
        exp.pretrained_model_path = args.pretrained
        exp.load_pretrained = True
        print(f"将使用预训练模型: {args.pretrained}")
    
    # 设置冻结选项
    if args.freeze_backbone:
        exp.freeze_backbone = True
        print("将冻结backbone层")
    
    if args.freeze_head:
        exp.freeze_head = True
        print("将冻结检测头层")

    configure_nccl()
    configure_omp()
    configure_module(args.fp16)

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    import random
    import warnings
    import torch.backends.cudnn as cudnn

    args = make_parser().parse_args()
    exp = get_exp(exp_file=args.exp_file, exp_name=args.name)

    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args=(exp, args),
        dist_url=args.dist_url,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        backend=args.dist_backend,
    ) 