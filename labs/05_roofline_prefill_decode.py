"""用 Roofline 数量级解释：为什么 Prefill 偏计算、低 batch Decode 偏带宽。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    name: str
    flops: float
    bytes_moved: float
    compute_s: float
    memory_s: float
    ridge: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes_moved

    @property
    def bottleneck(self) -> str:
        return "COMPUTE" if self.compute_s >= self.memory_s else "MEMORY"


def evaluate(name: str, flops: float, bytes_moved: float, peak_flops: float, bandwidth: float) -> Result:
    return Result(name, flops, bytes_moved, flops / peak_flops, bytes_moved / bandwidth, peak_flops / bandwidth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=float, default=70, help="活跃参数，单位 B")
    parser.add_argument("--weight-bytes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-bytes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--peak-tflops", type=float, default=1000)
    parser.add_argument("--hbm-tbps", type=float, default=3.35)
    args = parser.parse_args()

    positive = (
        args.params, args.prompt_tokens, args.decode_batch, args.context, args.layers,
        args.hidden, args.kv_heads, args.head_dim, args.peak_tflops, args.hbm_tbps,
    )
    if min(positive) <= 0:
        parser.error("规模和硬件参数必须大于 0")

    params = args.params * 1e9
    weight_bytes = params * args.weight_bytes
    peak_flops = args.peak_tflops * 1e12
    bandwidth = args.hbm_tbps * 1e12
    kv_per_position = 2 * args.layers * args.kv_heads * args.head_dim * args.kv_bytes

    prefill_flops = 2 * params * args.prompt_tokens
    prefill_bytes = weight_bytes
    decode_flops = 2 * params * args.decode_batch
    decode_flops += 4 * args.layers * args.hidden * args.context * args.decode_batch
    decode_bytes = weight_bytes + kv_per_position * args.context * args.decode_batch

    rows = [
        evaluate("Prefill", prefill_flops, prefill_bytes, peak_flops, bandwidth),
        evaluate("Decode", decode_flops, decode_bytes, peak_flops, bandwidth),
    ]

    print(f"硬件 Roofline 拐点：{rows[0].ridge:.1f} FLOP/byte（峰值 {args.peak_tflops:g} TFLOPS，HBM {args.hbm_tbps:g} TB/s）")
    print("阶段      FLOPs       搬移下界     算术强度   计算时间   访存时间   判断")
    for row in rows:
        print(
            f"{row.name:7s} {row.flops / 1e12:9.1f} T  {row.bytes_moved / 1e9:9.1f} GB  "
            f"{row.intensity:9.1f}  {row.compute_s * 1e3:8.1f}ms  {row.memory_s * 1e3:8.1f}ms  {row.bottleneck}"
        )

    print("\n现场实验：把 --decode-batch 从 1 改为 32，观察权重被更多 token 复用后算术强度如何上升。")
    print("边界：这是权重+KV 的下界模型，未计激活、算子效率、跨卡通信、调度和重叠。")


if __name__ == "__main__":
    main()
