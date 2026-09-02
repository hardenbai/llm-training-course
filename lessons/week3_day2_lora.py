# 周3 Day2：LoRA —— 用 1% 的参数微调大模型
# 对照源码: repos/minimind/model/model_lora.py + trainer/train_lora.py
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)

print("=" * 60)
print("第 1 步：LoRA 的全部数学 —— W' = W + BA")
print("=" * 60)
d_in, d_out, r = 512, 512, 8
W = nn.Linear(d_in, d_out, bias=False)          # 原模型权重（要冻结）
A = nn.Parameter(torch.randn(r, d_in) * 0.01)   # 降维: 512 -> 8
B = nn.Parameter(torch.zeros(d_out, r))         # 升维: 8 -> 512  (B初始化为0!)

full = d_in * d_out
lora = A.numel() + B.numel()
print(f"原始权重 W : {full:,} 个参数")
print(f"LoRA A+B   : {lora:,} 个参数（r={r}）")
print(f"占比       : {lora/full:.2%}  <- 只训练这 1.6%")

print()
print("=" * 60)
print("第 2 步：为什么 B 必须初始化为 0")
print("=" * 60)
x = torch.randn(1, d_in)
with torch.no_grad():
    base_out = W(x)
    lora_out = B @ A @ x.T     # B=0 -> BA=0
print(f"初始时 W(x)  = {base_out[0,:3].tolist()}")
print(f"初始时 +BA(x) = {(base_out + lora_out.T)[0,:3].tolist()}")
print(f"完全相等: {torch.allclose(base_out, base_out + lora_out.T)}")
print("""
B=0 保证训练开始时 LoRA 是'无操作'——模型行为和原来一模一样，
然后从零慢慢长出改动。如果 A、B 都随机初始化，一上来就把
原模型的行为打乱，训练会不稳定。""")

print()
print("=" * 60)
print("第 3 步：动手训一个 LoRA（目标任务：让输出整体偏移一个固定向量）")
print("=" * 60)
target_delta = torch.randn(1, d_out) * 0.5      # 假装'新技能'
W.weight.requires_grad_(False)                   # 冻结原权重！
opt = torch.optim.AdamW([A, B], lr=1e-2)         # 只优化 LoRA 参数

xs = torch.randn(64, d_in)
ys = W(xs) + target_delta                        # 目标行为 = 原行为 + 新技能

for step in range(300):
    pred = W(xs) + (xs @ A.T @ B.T)              # W(x) + B(A(x))
    loss = ((pred - ys) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == 299:
        print(f"  step {step:3d} | loss {loss.item():.6f}")

print(f"\n最终 loss: {loss.item():.2e} —— 只靠 {lora} 个参数学会了新行为")

print()
print("=" * 60)
print("第 4 步：合并 —— LoRA 的优雅谢幕")
print("=" * 60)
with torch.no_grad():
    W.weight += B @ A                            # 把 BA 烘焙回原权重
x_test = torch.randn(1, d_in)
with torch.no_grad():
    merged = W(x_test)
    lora_path = W(x_test)                        # 合并后再看（已含BA）
print("合并操作: W <- W + BA（一次矩阵加法）")
print(f"合并后推理不再需要 A、B —— 零额外开销，和普通模型一样快")

print()
print("=" * 60)
print("第 5 步：什么时候用 LoRA vs 全参微调")
print("=" * 60)
print("""
全参微调: 效果上限高 | 显存=参数量x(训练态约16字节/参数) | 每个任务一份完整模型
LoRA:    效果略低   | 显存骤降(冻结主体)                 | 每任务只存几十MB的AB
7B模型全参微调要 100GB+ 显存，LoRA 一张消费级显卡就行 —— 这也是
'家里有显卡就能定制模型'这个时代开启的原因。
MiniMind 里: model_lora.py 给 Attention/FFN 的 Linear 加了 lora 分支，
train_lora.py 冻结其余参数只训它们。""")

print()
print("作业：把 r 从 8 改成 2 和 64 各训一次，观察最终 loss ——")
print("感受 r（秩）= '给模型多少容量去学新东西'")
