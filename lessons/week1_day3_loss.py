# 周1 Day3：预训练 loss —— "预测下一个token"怎么变成一个数字
# 对照源码: repos/minimind/model/model_minimind.py 第 251-252 行
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import math
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ROOT / "repos" / "minimind" / "model")

print("=" * 60)
print("实验 1：错位 —— 源码里最精妙的两行")
print("=" * 60)
# 源码 251-252 行:
#   x, y = logits[..., :-1, :], labels[..., 1:]   # 关键的"错位"
#   loss = F.cross_entropy(x.view(-1, V), y.view(-1), ignore_index=-100)
text = "我爱深度学习"
ids = tok.encode(text)
print(f"句子: {text}")
print(f"token: {ids}")
print()
print("错位对齐（输入位置 i 的输出，要预测位置 i+1 的 token）:")
for i in range(len(ids) - 1):
    a = tok.decode([ids[i]]); b = tok.decode([ids[i + 1]])
    print(f"  看到 {a!r:8} -> 应预测出 {b!r:8}   (即 logits[{i}] 对齐 labels[{i+1}])")
print("""
[:-1] 去掉最后一个位置（它没有"下一个"可预测）
[1:]  去掉第一个位置（它是第一个输入，不作为预测目标）
这一行错位，就是"下一token预测"的全部实现。""")

print("=" * 60)
print("实验 2：交叉熵 —— 手算一遍，不再神秘")
print("=" * 60)
# 假设词表只有 4 个 token，模型对某位置输出 4 个分数(logits)
V = 4
logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
label = 0   # 正确答案是 token 0
probs = torch.softmax(logits, dim=-1)
manual_loss = -math.log(probs[label].item())
auto_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label])).item()
print(f"模型分数      : {logits.tolist()}")
print(f"softmax 后   : {[round(p,3) for p in probs.tolist()]}")
print(f"正确答案     : token {label}（概率 {probs[label]:.3f}）")
print(f"手算 loss    : -log({probs[label]:.3f}) = {manual_loss:.4f}")
print(f"框架算的 loss: {auto_loss:.4f}   <- 完全一致")
print("解读：模型给正确答案的概率越高，loss 越小。训练=让 loss 变小=让正确答案概率变大。")

print()
print("=" * 60)
print("实验 3：-100 —— 让 padding 和不该学的位置闭嘴")
print("=" * 60)
# 玩具词表演示（V=4）：一句话 3 个 token，补 padding 到长度 8
real = [1, 2, 3]
padded = real + [0] * (8 - len(real))   # 0 是 pad_token
labels = real + [-100] * (8 - len(real))
print(f"input_ids: {padded}")
print(f"labels  : {labels}   <- padding 位置是 -100")
logits8 = torch.randn(8, V)
logits8[torch.arange(8), padded] = 3.0  # 假装模型偏向"正确"答案
loss_ignore = F.cross_entropy(logits8, torch.tensor(padded), ignore_index=-100).item()
print(f"带 ignore_index 的 loss: {loss_ignore:.4f}")
print("ignore_index=-100 的位置完全不参与 loss —— 周3 SFT 的'只学 assistant 回答'")
print("用的就是同一个机制：把不该学的 token 全标成 -100。")

print()
print("=" * 60)
print("实验 4：困惑度 PPL —— loss 的'人话'版本")
print("=" * 60)
print("PPL = exp(loss)，含义：模型平均在'多少个候选里犹豫'")
for loss, note in [
    (math.log(6400), "均匀乱猜（MiniMind 刚初始化时 loss≈%.2f）" % math.log(6400)),
    (6.0, "训练早期"),
    (4.0, "训练中期"),
    (3.0, "MiniMind 全量训练后水平"),
]:
    print(f"  loss = {loss:.2f}  ->  PPL = {math.exp(loss):8.0f}   {note}")
print("""
PPL=6400：和闭眼乱猜没区别（词表6400选1）
PPL=20  ：相当于每次在20个词里挑对的那个
loss每降1，PPL缩小为原来的 1/e ≈ 37%""")

print()
print("今日作业：")
print("1. 把实验1的 text 换成任意英文句子，重跑看错位对齐")
print("2. 思考：为什么最后一个位置的 logits 没有对应的标签？（答案就在 [:-1]）")
