"""容量优先：估算推理权重、KV Cache 与 ZeRO 各阶段的训练状态。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    total_params_b: float
    active_params_b: float
    layers: int
    kv_heads: int
    head_dim: int


PRESETS = {
    "7b": ModelPreset(7, 7, 32, 8, 128),
    "70b": ModelPreset(70, 70, 80, 8, 128),
    "moe200": ModelPreset(200, 22, 48, 8, 128),
}

TRAINING_COMPONENTS = (
    ("BF16 参数", 2, 3),
    ("BF16 梯度", 2, 2),
    ("FP32 主参数", 4, 1),
    ("Adam 一阶矩", 4, 1),
    ("Adam 二阶矩", 4, 1),
)


def kv_cache_gb(model: ModelPreset, context: int, batch: int, bytes_per_value: int) -> float:
    cache_bytes = 2 * model.layers * model.kv_heads * model.head_dim
    cache_bytes *= context * batch * bytes_per_value
    return cache_bytes / 1e9


def training_state_gb(params_b: float, zero_stage: int, shards: int) -> tuple[float, list[str]]:
    total = 0.0
    rows = []
    for name, bytes_per_param, sharded_from_stage in TRAINING_COMPONENTS:
        divisor = shards if zero_stage >= sharded_from_stage else 1
        size = params_b * bytes_per_param / divisor
        total += size
        rows.append(f"{name} {size:.1f} GB" + (f" ÷{shards}" if divisor > 1 else ""))
    return total, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="70b")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1, help="并发序列数")
    parser.add_argument("--kv-bytes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--weight-bytes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tp", type=int, default=1, help="推理权重张量并行份数")
    parser.add_argument("--shards", type=int, default=8, help="ZeRO/FSDP 状态分片份数")
    args = parser.parse_args()

    if min(args.context, args.batch, args.tp, args.shards) < 1:
        parser.error("context、batch、tp 和 shards 必须大于 0")

    model = PRESETS[args.preset]
    kv_total = kv_cache_gb(model, args.context, args.batch, args.kv_bytes)
    kv_shards = min(args.tp, model.kv_heads)
    weight_total = model.total_params_b * args.weight_bytes
    active_read = model.active_params_b * args.weight_bytes

    print(f"模型预设：{args.preset}（总参数 {model.total_params_b:g}B；每 token 激活约 {model.active_params_b:g}B）")
    print("\n[推理静态容量]")
    print(f"权重总量                   : {weight_total:9.2f} GB")
    print(f"TP={args.tp} 每卡权重        : {weight_total / args.tp:9.2f} GB")
    print(f"每步活跃权重读取下界        : {active_read:9.2f} GB / batch")
    print(f"KV Cache 总量              : {kv_total:9.2f} GB")
    print(f"KV 每卡（最多按 KV heads 切）: {kv_total / kv_shards:9.2f} GB")

    print(f"\n[训练模型状态：{args.shards} 路分片，未计激活/临时 buffer]")
    print("阶段    每卡 GB    分片范围")
    for stage in range(4):
        total, rows = training_state_gb(model.total_params_b, stage, args.shards)
        print(f"ZeRO-{stage}  {total:8.1f}    " + "；".join(rows))

    print("\n结论：容量是硬约束；MoE 的总参数决定能否装下，活跃参数更接近每 token 计算与权重读取量。")
    print("边界：结果未计激活、通信 buffer、内存碎片、量化元数据和框架工作区。")


if __name__ == "__main__":
    main()
