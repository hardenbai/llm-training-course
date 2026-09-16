"""模拟 MoE Top-k 路由：观察热点专家、容量溢出和 All-to-All 数据量。"""

from __future__ import annotations

import argparse
import math
import random


def weighted_sample_without_replacement(rng: random.Random, weights: list[float], count: int) -> list[int]:
    remaining = list(range(len(weights)))
    chosen = []
    for _ in range(count):
        pick = rng.choices(remaining, weights=[weights[i] for i in remaining], k=1)[0]
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--bytes", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--skew", type=float, default=0.35, help="0=均匀，1=全部偏向 expert 0")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not 0 <= args.skew <= 1:
        parser.error("skew 必须位于 0 到 1")
    if args.capacity_factor <= 0 or args.top_k > args.experts or min(args.tokens, args.experts, args.top_k, args.hidden, args.world_size) < 1:
        parser.error("规模参数无效，且 top-k 不能大于 experts")

    hot = 1 / args.experts + args.skew * (1 - 1 / args.experts)
    weights = [hot] + [(1 - hot) / (args.experts - 1)] * (args.experts - 1) if args.experts > 1 else [1.0]
    rng = random.Random(args.seed)
    loads = [0] * args.experts
    for _ in range(args.tokens):
        for expert in weighted_sample_without_replacement(rng, weights, args.top_k):
            loads[expert] += 1

    assignments = args.tokens * args.top_k
    average = assignments / args.experts
    capacity = math.ceil(args.capacity_factor * average)
    dropped = sum(max(0, load - capacity) for load in loads)
    dispatch_and_combine = 2 * assignments * args.hidden * args.bytes
    remote_fraction = 0 if args.world_size == 1 else (args.world_size - 1) / args.world_size

    print(f"Top-{args.top_k} assignments={assignments:,}；平均/专家={average:.1f}；容量/专家={capacity}")
    scale = max(loads) or 1
    for index, load in enumerate(loads):
        bar = "█" * max(1, round(load / scale * 32))
        overflow = max(0, load - capacity)
        print(f"E{index:02d} {load:6d} {bar:<32s} overflow={overflow}")

    print(f"\n最大/平均负载             : {max(loads) / average:.2f}x")
    print(f"容量溢出 assignments      : {dropped:,} ({dropped / assignments:.1%})")
    print(f"dispatch+combine 载荷      : {dispatch_and_combine / 1e9:.3f} GB / MoE layer")
    print(f"粗略跨设备部分             : {dispatch_and_combine * remote_fraction / 1e9:.3f} GB")
    print("\n现场实验：分别运行 --skew 0 和 --skew 0.7；辅助负载均衡损失解决的是长尾，不是数学装饰。")
    print("边界：模拟器省略路由 logits、拓扑、token 丢弃策略和通信重叠。")


if __name__ == "__main__":
    main()
