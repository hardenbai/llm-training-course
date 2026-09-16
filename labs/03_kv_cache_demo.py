"""验证 KV Cache 与完整重算结果一致，并比较两种方法的工作量。"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width 必须能被 heads 整除")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda t: t.view(t.size(0), t.size(1), self.heads, self.head_dim).transpose(1, 2)
        return split(q), split(k), split(v)

    def full(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project(x)
        values = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        merged = values.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.out(merged)

    def decode(
        self,
        x_new: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q, k_new, v_new = self.project(x_new)
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
        values = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        merged = values.transpose(1, 2).contiguous().view(x_new.size(0), 1, -1)
        return self.out(merged), (k, v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args()

    if min(args.prompt, args.generate, args.width, args.heads) < 1:
        parser.error("所有规模参数必须大于 0")

    torch.manual_seed(3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = CausalSelfAttention(args.width, args.heads).to(device).eval()
    prompt = torch.randn(1, args.prompt, args.width, device=device)
    future = torch.randn(1, args.generate, args.width, device=device)

    with torch.inference_mode():
        _, cache_k, cache_v = attention.project(prompt)
        cache = (cache_k, cache_v)
        sequence = prompt
        max_error = 0.0
        for index in range(args.generate):
            new_token = future[:, index:index + 1]
            sequence = torch.cat((sequence, new_token), dim=1)
            reference = attention.full(sequence)[:, -1:]
            cached, cache = attention.decode(new_token, cache)
            max_error = max(max_error, (reference - cached).abs().max().item())

    lengths = [args.prompt + step for step in range(1, args.generate + 1)]
    projected_without_cache = sum(lengths)
    projected_with_cache = args.prompt + args.generate
    scores_without_cache = sum(length * length for length in lengths)
    scores_with_cache = args.prompt * args.prompt + sum(lengths)
    kv_bytes = 2 * args.heads * (args.width // args.heads) * (args.prompt + args.generate) * 4

    print(f"device / 最大输出误差          : {device} / {max_error:.3e}")
    print(f"K/V 投影 token 数（完整重算） : {projected_without_cache:,}")
    print(f"K/V 投影 token 数（使用缓存） : {projected_with_cache:,}")
    print(f"投影工作量缩减                : {projected_without_cache / projected_with_cache:.1f}x")
    print(f"注意力 score 元素（完整重算） : {scores_without_cache:,}")
    print(f"注意力 score 元素（使用缓存） : {scores_with_cache:,}")
    print(f"单层 FP32 KV Cache            : {kv_bytes / 1e6:.2f} MB")
    print("\n结论：Cache 避免历史 token 的 K/V 重投影，但新 Q 仍需读取全部历史 K/V；计算减少，容量与带宽压力上升。")


if __name__ == "__main__":
    main()
