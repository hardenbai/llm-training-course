# 周1 Day4：把模型剩下的零件一次讲完 —— RMSNorm / SwiGLU / 残差 / MoE / 采样
# 对照源码: model_minimind.py 第 50-60(RMSNorm) 136-146(SwiGLU) 148-176(MoE) 186-194(残差)
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)

print("=" * 60)
print("实验 1：RMSNorm —— LayerNorm 的省钱版（源码 50-60 行）")
print("=" * 60)
x = torch.randn(3, 8)
# LayerNorm: 减均值再除标准差
ln = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)
# RMSNorm: 不减均值，直接除均方根
rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
print(f"LayerNorm 后第一行: {[round(v,2) for v in ln[0].tolist()]}")
print(f"RMSNorm   后第一行: {[round(v,2) for v in rms[0].tolist()]}")
print("两者作用相同（把向量拉回稳定幅度），RMSNorm 少算一次均值，")
print("在隐藏维度上千、每层两次归一化的场景下省出可观算力。Llama/Qwen/MiniMind 都用它。")

print()
print("=" * 60)
print("实验 2：SwiGLU —— 带闸门的前馈网络（源码 136-146 行）")
print("=" * 60)
# 源码: down( act(gate(x)) * up(x) )   <- 注意是乘法，不是加法！
gate_proj = nn.Linear(8, 16, bias=False); up_proj = nn.Linear(8, 16, bias=False)
g = gate_proj(x[0]); u = up_proj(x[0])
silu = g * torch.sigmoid(g)          # SiLU 激活
gated = silu * u                     # 闸门：up 的信息被 gate 控制
print(f"gate 分支(激活后): {[round(v,2) for v in silu[:6].tolist()]}")
print(f"up   分支(原始)  : {[round(v,2) for v in u[:6].tolist()]}")
print(f"相乘后           : {[round(v,2) for v in gated[:6].tolist()]}")
print("gate 接近 0 的维度被'关掉'，信息过不来 —— 网络985万个参数里，")
print("一半在学'放什么进来'(up)，一半在学'让什么通过'(gate)。")

print()
print("=" * 60)
print("实验 3：残差连接 —— 深层网络的生命线（源码 186-194 行）")
print("=" * 60)
# 对比 16 层小网络，有/无残差时输入梯度的规模
def make_grad(depth, residual):
    # 尺度取 0.8/sqrt(d)：无残差时每层把信号缩小到 0.8 倍（模拟深层衰减）
    Ws = [torch.randn(64, 64) * 0.8 / 8.0 for _ in range(depth)]
    x = torch.randn(64, requires_grad=True)
    h = x
    for W in Ws:
        h = h @ W + (h if residual else 0)
    h.sum().backward()
    return x.grad.norm().item()

g_no, g_yes = make_grad(16, False), make_grad(16, True)
print(f"16 层网络，无残差: 输入梯度范数 = {g_no:.2e}")
print(f"16 层网络，有残差: 输入梯度范数 = {g_yes:.2e}   <- 相差约 {g_yes/g_no:.0f} 倍")
print("残差让梯度有'高速公路'直达浅层。没有它，层数一深梯度就传不回去。")
print("源码里就是 hidden_states += residual 和 hidden = hidden + mlp(...) 这两个 += ")

print()
print("=" * 60)
print("实验 4：MoE —— 稀疏专家路由（源码 148-176 行，MiniMind 可选开启）")
print("=" * 60)
n_experts, d = 4, 32
gate = nn.Linear(d, n_experts, bias=False)
experts = [nn.Linear(d, d, bias=False) for _ in range(n_experts)]
tokens = torch.randn(100, d)
routing = gate(tokens).argmax(-1)
print(f"100 个 token 各自选一个专家（top-1 路由）:")
for e in range(n_experts):
    bar = "█" * int((routing == e).sum())
    print(f"  专家{e}: {bar} {int((routing==e).sum())} 个token")
print("每个 token 只经过 1 个专家的计算 -> 参数多了，计算量没多。")
print("风险：大家都挤同一个专家 -> 引入 aux_loss 鼓励负载均衡（源码 171-173 行）")

print()
print("=" * 60)
print("实验 5：采样 —— 模型输出的概率怎么变成文字（源码 267-278 行）")
print("=" * 60)
logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5, 0.0])
def show(name, probs):
    top = probs.argsort(descending=True)[:4]
    print(f"  {name:22}: " + "  ".join(f"token{i}:{probs[i]:.2f}" for i in top.tolist()))
print("原始分布 softmax(logits):"); show("T=1.0", torch.softmax(logits, -1))
print("低温 T=0.5（更确定/保守）:"); show("T=0.5", torch.softmax(logits/0.5, -1))
print("高温 T=2.0（更随机/有创意）:"); show("T=2.0", torch.softmax(logits/2.0, -1))
p = torch.softmax(logits, -1)
sorted_p, _ = torch.sort(p, descending=True)
keep = torch.cumsum(sorted_p, -1) <= 0.8
print(f"top_p=0.8: 只保留累计概率80%的token，其余概率清零后重新归一化")
print("""
temperature：除进 logits 再 softmax，温度越低分布越尖
top_p / top_k：先砍掉长尾再采样，防止'抽风'选到离谱token
写代码助手要低温(准)，写诗要高温(活)""")

print()
print("至此 MiniMind 的所有零件都见过了。完整一层 = 注意力(昨天) + 前馈 + 两个残差 + 两个RMSNorm")
print("明日作业：画出一层的结构图（手画拍照给自己看就行）")
