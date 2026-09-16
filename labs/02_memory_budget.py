"""用数量级估算解释训练状态和 KV Cache 为什么会吃掉显存。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int


PRESETS = {
    "7b": ModelPreset(7, 32, 8, 128),
    "70b": ModelPreset(70, 80, 8, 128),
    "moe": ModelPreset(200, 48, 8, 128),
}


def kv_cache_gb(model: ModelPreset, context: int, batch: int, bytes_per_value: int) -> float:
    cache_bytes = 2 * model.layers * model.kv_heads * model.head_dim
    cache_bytes *= context * batch * bytes_per_value
    return cache_bytes / 1e9


parser = argparse.ArgumentParser()
parser.add_argument("--preset", choices=PRESETS, default="70b")
parser.add_argument("--context", type=int, default=8192)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--kv-bytes", type=int, choices=(1, 2), default=2)
args = parser.parse_args()

model = PRESETS[args.preset]
print(f"模型预设                 : {args.preset}")
print(f"BF16 推理权重            : {model.params_b * 2:8.1f} GB")
print(f"INT8 推理权重            : {model.params_b:8.1f} GB")
print(f"混合精度 AdamW 训练状态  : {model.params_b * 16:8.1f} GB（未含激活）")
print(
    f"KV Cache                 : "
    f"{kv_cache_gb(model, args.context, args.batch, args.kv_bytes):8.2f} GB"
)
print("提示：GQA/MQA 减少的是 KV heads，因此会直接压缩 KV Cache。")
