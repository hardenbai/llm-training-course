"""估算数据并行与张量并行为什么依赖不同的网络路径。"""

from __future__ import annotations

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--params", type=float, default=70, help="总参数量，单位 B")
parser.add_argument("--gpus", type=int, default=8)
parser.add_argument("--mode", choices=("dp", "tp"), default="dp")
parser.add_argument("--tokens", type=int, default=8192, help="TP 时的 batch×sequence")
parser.add_argument("--hidden", type=int, default=8192)
args = parser.parse_args()

if args.mode == "dp":
    gradient_gb = args.params * 2  # BF16 梯度，1B × 2 bytes ≈ 2GB
    ring_factor = 2 * (args.gpus - 1) / args.gpus
    traffic_gb = gradient_gb * ring_factor
    print(f"DP：每卡保存完整模型；每步近似 All-Reduce {traffic_gb:.1f} GB 梯度数据。")
    print("通信集中在反向传播后，可通过 bucket 与反向计算重叠。")
else:
    activation_gb = args.tokens * args.hidden * 2 / 1e9
    traffic_gb = activation_gb * 2 * (args.gpus - 1) / args.gpus
    print(f"TP：单个激活张量约 {activation_gb:.2f} GB。")
    print(f"一次近似集合通信约 {traffic_gb:.2f} GB，而且会在很多层重复发生。")
    print("因此 TP 更偏好节点内 NVLink/高速 ICI，而 DP 更容易跨节点扩展。")
