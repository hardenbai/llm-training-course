"""用一个注意力层演示增量解码只计算新 token 的 K/V。"""

import time

import torch
from torch import nn


torch.manual_seed(3)
device = "cuda" if torch.cuda.is_available() else "cpu"
width, heads, prompt_length, generated = 256, 8, 256, 64
attention = nn.MultiheadAttention(width, heads, batch_first=True).to(device).eval()
prompt = torch.randn(1, prompt_length, width, device=device)


@torch.inference_mode()
def replay_everything() -> float:
    sequence = prompt
    started = time.perf_counter()
    for _ in range(generated):
        attention(sequence, sequence, sequence, need_weights=False)
        sequence = torch.cat((sequence, torch.randn(1, 1, width, device=device)), dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


@torch.inference_mode()
def project_only_new_tokens() -> float:
    # 教学近似：真实 KV Cache 会保存每层投影后的 K/V，并让新 Q 读取全部缓存。
    attention(prompt, prompt, prompt, need_weights=False)
    started = time.perf_counter()
    for _ in range(generated):
        new_token = torch.randn(1, 1, width, device=device)
        attention(new_token, new_token, new_token, need_weights=False)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


full = replay_everything()
incremental = project_only_new_tokens()
print(f"device                  : {device}")
print(f"重复处理完整历史        : {full:.4f}s")
print(f"只投影新增 token（近似）: {incremental:.4f}s")
print(f"时间比                  : {full / max(incremental, 1e-9):.1f}x")
print("注意：这是解释重复计算的微实验，不是生产 KV Cache 的性能基准。")
