"""比较 DP、TP、PP、EP 的通信对象、频率与带宽时间下界。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    mode: str
    payload_gb: float
    events: int
    description: str

    @property
    def total_gb(self) -> float:
        return self.payload_gb * self.events


def ring_factor(world: int) -> float:
    return 0.0 if world == 1 else 2 * (world - 1) / world


def estimates(args: argparse.Namespace) -> list[Estimate]:
    activation_gb = args.tokens * args.hidden * args.bytes / 1e9
    return [
        Estimate(
            "DP",
            args.params * args.bytes * ring_factor(args.gpus),
            1,
            "每步同步梯度；实现通常拆成多个 bucket 并与反向重叠",
        ),
        Estimate(
            "TP",
            activation_gb * ring_factor(args.gpus),
            2 * args.layers,
            "每层近似两次激活集合通信；高频，偏好节点内高速互联",
        ),
        Estimate(
            "PP",
            activation_gb,
            2 * args.microbatches,
            "每个微批在 stage 边界发送前向激活和反向梯度",
        ),
        Estimate(
            "EP",
            activation_gb * args.top_k * (args.gpus - 1) / args.gpus,
            2 * args.layers,
            "每个 MoE 层 dispatch + combine；真实代价受热点专家和 All-to-All 拓扑影响",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "dp", "tp", "pp", "ep"), default="all")
    parser.add_argument("--params", type=float, default=70, help="参数量，单位 B")
    parser.add_argument("--gpus", type=int, default=8, help="参与该并行组的设备数")
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--tokens", type=int, default=8192, help="一次微批的 batch×sequence")
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--bytes", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--bandwidth", type=float, default=400, help="有效单向带宽，GB/s")
    args = parser.parse_args()

    if min(args.params, args.gpus, args.layers, args.tokens, args.hidden, args.microbatches, args.top_k, args.bandwidth) <= 0:
        parser.error("规模参数必须大于 0")

    rows = estimates(args)
    if args.mode != "all":
        rows = [row for row in rows if row.mode.lower() == args.mode]

    print(f"假设：{args.gpus} 卡并行组，{args.bandwidth:g} GB/s 有效带宽；不计延迟与计算通信重叠。")
    print("策略  单次载荷 GB  事件/步  每步流量 GB  串行时间下界")
    for row in rows:
        lower_bound_ms = row.total_gb / args.bandwidth * 1000
        print(f"{row.mode:>3}   {row.payload_gb:10.3f}  {row.events:7d}  {row.total_gb:11.3f}  {lower_bound_ms:10.2f} ms")
        print(f"      {row.description}")

    print("\n读表方法：先看『每次传多大』，再看『每步发生几次』；总字节相同，很多小消息也可能更慢。")
    print("边界：这些是算法级上界/下界教学估算，不代表具体框架的实测通信量。")


if __name__ == "__main__":
    main()
