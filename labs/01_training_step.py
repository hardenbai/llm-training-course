"""把一次训练拆成前向、反向、优化器更新，并观察张量与状态占用。"""

from __future__ import annotations

import argparse

import torch
from torch import nn


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 128, width: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 4,
            batch_first=True,
        )
        self.lm_head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=token_ids.device)
        hidden = self.block(self.embedding(token_ids), src_mask=mask)
        return self.lm_head(hidden)


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用 CUDA")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"

    torch.manual_seed(7)
    model = TinyCausalLM(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    observed_activations = []

    def capture(name: str):
        def hook(_module, _inputs, output) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                observed_activations.append((name, tuple(tensor.shape), tensor.numel() * tensor.element_size()))
        return hook

    hooks = [
        model.embedding.register_forward_hook(capture("embedding")),
        model.block.register_forward_hook(capture("transformer block")),
        model.lm_head.register_forward_hook(capture("logits")),
    ]

    batch = torch.randint(0, 128, (args.batch, args.seq + 1), device=device)
    inputs, labels = batch[:, :-1], batch[:, 1:]
    before = model.embedding.weight.detach().clone()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logits = model(inputs)  # 1. 前向：读参数，产生并保存反向所需激活
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # 2. 反向：读激活，写梯度
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()  # 3. 更新：读写参数、梯度和 Adam 状态

    for handle in hooks:
        handle.remove()

    parameters = list(model.parameters())
    parameter_bytes = tensor_bytes(parameters)
    gradient_bytes = tensor_bytes([p.grad for p in parameters])
    optimizer_tensors = [value for state in optimizer.state.values() for value in state.values()]
    optimizer_bytes = tensor_bytes(optimizer_tensors)
    changed = (model.embedding.weight.detach() - before).abs().mean()

    print(f"device / input / logits : {device} / {tuple(inputs.shape)} / {tuple(logits.shape)}")
    print(f"loss / grad norm        : {loss.item():.4f} / {gradient_norm.item():.4f}")
    print(f"mean |Δ embedding|      : {changed.item():.8f}")
    print("\n[本次微实验实际张量]")
    print(f"参数                     : {parameter_bytes / 1e6:8.2f} MB")
    print(f"梯度                     : {gradient_bytes / 1e6:8.2f} MB")
    print(f"Adam 状态（step 后创建） : {optimizer_bytes / 1e6:8.2f} MB")
    print("前向可见张量（不等于 autograd 保存激活的精确峰值）：")
    for name, shape, size in observed_activations:
        print(f"  {name:18s} {str(shape):20s} {size / 1e6:7.2f} MB")
    if device == "cuda":
        print(f"CUDA allocator 峰值      : {torch.cuda.max_memory_allocated() / 1e6:8.2f} MB")
    print("\n放大规律：参数/梯度/优化器随参数量增长；激活主要随 batch×sequence×hidden×layers 增长。")


if __name__ == "__main__":
    main()
