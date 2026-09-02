# 周1 Day2：注意力机制 —— Transformer 的心脏
# 对照源码: repos/minimind/model/model_minimind.py 第 91-134 行
import sys
sys.stdout.reconfigure(encoding="utf-8")
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)

print("=" * 60)
print("实验 1：Embedding —— token ID 变成向量")
print("=" * 60)
# 源码第 201 行: self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
# 就是一张查找表：第 i 行 = token i 的向量
vocab, dim = 10, 4
embed = torch.nn.Embedding(vocab, dim)
sentence = torch.tensor([[2, 5, 2, 8]])   # 4 个 token 的句子
x = embed(sentence)
print(f"句子 token ID: {sentence[0].tolist()}")
print(f"查表后形状  : {tuple(x.shape)}  (1句, 4个token, 每个4维向量)")
print(f"token 2 的向量: {[round(v,2) for v in x[0,0].tolist()]}")
print(f"两个'2'的向量完全相同（同一行查了两次）: {torch.equal(x[0,0], x[0,2])}")
print("-> 到这一步为止模型还'不知道' token 2 出现了两次（无位置信息），")
print("   这就是需要 RoPE 的原因（实验 4）")

print()
print("=" * 60)
print("实验 2：注意力 —— 每个 token 决定'看谁'")
print("=" * 60)
# 复刻源码第 113/128/131 行的公式（去掉工程细节）：
#   scores = Q @ K^T / sqrt(d)     (第128行)
#   weights = softmax(scores)      (第131行)
#   output = weights @ V           (第131行)
seq_len, d = 4, 4
Wq = torch.randn(d, d); Wk = torch.randn(d, d); Wv = torch.randn(d, d)
Q, K, V = x[0] @ Wq, x[0] @ Wk, x[0] @ Wv     # 每个token各自生成 Q、K、V
scores = (Q @ K.T) / math.sqrt(d)
weights = torch.softmax(scores, dim=-1)
print("注意力权重矩阵（每行=一个token分配给各token的注意力，行和=1）:")
for i in range(seq_len):
    row = " ".join(f"{v:.2f}" for v in weights[i].tolist())
    print(f"  token{i} 看: [{row}]  合计={weights[i].sum():.2f}")
print("""
解读（拿 token3 那行举例）：
  Q(token3) 和每个 K 算相似度 -> softmax 变成权重 -> 加权求和 V
  权重大的 = token3 '关注' 的对象。
  Q=查询(我想找什么)  K=键(我有什么可被找到)  V=值(我的实际内容)
  类比图书馆：Q=你的检索词，K=每本书的标签，V=书的内容""")

print("=" * 60)
print("实验 3：因果掩码 —— 为什么 GPT 不能'偷看未来'")
print("=" * 60)
# 复刻源码第 129 行: scores += torch.full(..., -inf).triu(1)
causal = torch.full((seq_len, seq_len), float("-inf")).triu(1)
masked = torch.softmax(scores + causal, dim=-1)
print("加上因果掩码后的注意力（上三角=0，只能看自己和过去）:")
for i in range(seq_len):
    row = " ".join(f"{v:.2f}" for v in masked[i].tolist())
    print(f"  token{i} 看: [{row}]")
print("""
token3 只能给 token0-3 分配注意力，token4-7 一律为 0。
为什么？训练目标是'预测下一个token'——
如果 token3 能看到 token4（答案），就等于抄答案，模型学不到东西。
BERT 是双向的（完形填空），GPT 是因果的（接龙），本质区别就在这个三角。""")

print("=" * 60)
print("实验 4：RoPE —— 用旋转编码'相对位置'")
print("=" * 60)
# 源码第 62-84 行。核心思想：把向量两两配对，按位置乘不同频率的旋转角
# 关键性质：旋转后 Q_m·K_n 只依赖相对距离 m-n，不依赖绝对位置
def rot(vec2, angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([vec2[0]*c - vec2[1]*s, vec2[0]*s + vec2[1]*c])

q = torch.randn(2); k = torch.randn(2)
theta = 0.3   # 某一维的旋转频率
m, n = 2, 7   # 相对距离 5
s_off = 100   # 两个位置同时平移 100，相对距离仍是 5
dot1 = torch.dot(rot(q, m*theta), rot(k, n*theta))
dot2 = torch.dot(rot(q, (m+s_off)*theta), rot(k, (n+s_off)*theta))
print(f"q 在位置{m}、k 在位置{n}       -> 注意力得分 {dot1:.6f}")
print(f"q 在位置{m+100}、k 在位置{n+100} -> 注意力得分 {dot2:.6f}")
print(f"两者相等: {torch.allclose(dot1, dot2)}")
print("""
绝对位置变了，得分不变 —— 模型学到的是'你在我前面第5个'，
而不是'你在第107个'。这样的规律可以泛化到训练时没见过的长度。""")

print("=" * 60)
print("实验 5：KV-Cache —— 推理加速的关键技巧")
print("=" * 60)
# 源码第 120-123 行: 新token的K、V 与缓存过的 K、V 拼接
# 生成第100个token时，前99个token的K、V完全不用重算，直接用缓存
sys.path.insert(0, "repos/minimind")
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4, vocab_size=6400,
                     num_attention_heads=8, num_key_value_heads=4)
model = MiniMindForCausalLM(cfg).eval()
prompt = torch.randint(3, 6400, (1, 512))
mask = torch.ones_like(prompt)

import time
for use_cache in [True, False]:
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    out = model.generate(inputs=prompt, attention_mask=mask,
                         max_new_tokens=60, use_cache=use_cache,
                         do_sample=True, temperature=0.8, top_p=0.9)
    dt = time.perf_counter() - t0
    print(f"use_cache={str(use_cache):5} | 生成60个token耗时 {dt:6.2f} 秒")

print("""
无缓存：每生成1个token，都要把整个序列的所有K、V从头算一遍
有缓存：只算新token的1份K、V，旧的直接复用
序列越长差距越大 —— 这就是ChatGPT'越写越快'的秘密之一。""")

print()
print("今日源码阅读作业（对照实验看真实实现）:")
print("  model_minimind.py:201  embed_tokens      (实验1)")
print("  model_minimind.py:113  q/k/v_proj        (实验2)")
print("  model_minimind.py:128  Q@K^T/sqrt(d)     (实验2)")
print("  model_minimind.py:129  causal mask       (实验3)")
print("  model_minimind.py:62-84 RoPE             (实验4)")
print("  model_minimind.py:120-123 KV-Cache       (实验5)")
print("  model_minimind.py:251-252 训练loss的错位 (明天Day3的主题!)")
