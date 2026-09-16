"""最小 Transformer 训练步骤：让观众看到数据、前向、损失、反向和更新。"""

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
        sequence_length = token_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sequence_length, device=token_ids.device
        )
        hidden = self.block(self.embedding(token_ids), src_mask=causal_mask)
        return self.lm_head(hidden)


torch.manual_seed(7)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyCausalLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 输入和标签错开一位：看见 [12, 9, 31]，学习预测 [9, 31, 6]。
batch = torch.randint(0, 128, (8, 33), device=device)
inputs, labels = batch[:, :-1], batch[:, 1:]

before = model.embedding.weight.detach().clone()
logits = model(inputs)                                      # 1. 前向
loss = nn.functional.cross_entropy(                         # 2. 损失
    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
)
optimizer.zero_grad(set_to_none=True)
loss.backward()                                             # 3. 反向
gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()                                            # 4. 更新

changed = (model.embedding.weight.detach() - before).abs().mean()
print(f"device          : {device}")
print(f"input shape     : {tuple(inputs.shape)}")
print(f"logits shape    : {tuple(logits.shape)}")
print(f"loss            : {loss.item():.4f}")
print(f"gradient norm   : {gradient_norm.item():.4f}")
print(f"mean |Δweight|  : {changed.item():.8f}")
