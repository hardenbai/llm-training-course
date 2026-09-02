# 周4 Day1：DPO —— 不用强化学习的'偏好对齐'
# 对照源码: repos/minimind/trainer/train_dpo.py
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)

print("=" * 60)
print("第 1 步：对齐问题 —— SFT 之后的模型还缺什么")
print("=" * 60)
print("""
SFT 教会模型'怎么回答'，但训练数据里好答案坏答案混在一起，
模型学到的是'平均风格'。对齐 = 明确告诉它'这个回答比那个好'。

经典路线 RLHF 三步: SFT -> 训练奖励模型 -> PPO强化学习(复杂、不稳定)
DPO 的洞察: 偏好数据可以直接推出最优策略，跳过后两步，
用一个小学生都能看懂的损失函数直接优化。""")

print("=" * 60)
print("第 2 步：DPO 损失 —— 一步步推")
print("=" * 60)
# loss = -log sigmoid( beta * [ (logp_chosen - logp_ref_chosen)
#                              - (logp_rejected - logp_ref_rejected) ] )
print("""
对同一个问题，有两个回答：
  chosen   = 人类偏好的回答（✓）
  rejected = 被嫌弃的回答（✗）

让模型(policy)相对参考模型(ref，通常是SFT刚训完的自己)：
  提高 chosen 的概率：   logp_✓ - logp_ref_✓  变大
  压低 rejected 的概率： logp_✗ - logp_ref_✗  变小
两者的'差'越大，sigmoid 越接近 1，-log sigmoid 越接近 0。
beta 控制步子：beta 越大越保守（不敢偏离参考模型太远）。""")

print("=" * 60)
print("第 3 步：从零实现一个 DPO（玩具但完整）")
print("=" * 60)
# 玩具世界：词表10个token，'policy'是可学习的logits，'问题'固定
V = 10
CHOSEN, REJECTED = 3, 7        # 好答案=token3, 坏答案=token7

policy = nn.Parameter(torch.zeros(V))       # 待训练的策略 logits
with torch.no_grad():
    ref = torch.zeros(V)                     # 参考模型（冻结，和初始policy相同）

def seq_logp(logits, token):                 # 单token的log概率
    return F.log_softmax(logits, dim=-1)[token]

BETA = 0.5
opt = torch.optim.AdamW([policy], lr=0.1)
print(f"{'step':>4} | {'logp(✓3)':>9} | {'logp(✗7)':>9} | {'隐式奖励差':>10} | {'DPO loss':>9}")
for step in range(60):
    lp_c = seq_logp(policy, CHOSEN)
    lp_r = seq_logp(policy, REJECTED)
    margin = (lp_c - seq_logp(ref, CHOSEN)) - (lp_r - seq_logp(ref, REJECTED))
    loss = -F.logsigmoid(BETA * margin)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0 or step == 59:
        print(f"{step:4d} | {lp_c.item():9.3f} | {lp_r.item():9.3f} | {margin.item():10.3f} | {loss.item():9.4f}")

print("""
看结果：chosen(token3) 的 logp 一路上升，rejected(token7) 一路下降，
loss 逼近 0 —— 模型'学会了偏好'。
和 trainer/train_dpo.py 的区别只是：seq_logp 换成了整个回答序列的
逐token log概率求和（带长度归一化），其他一模一样。""")

print("=" * 60)
print("第 4 步：beta 的作用 —— 对齐的'缰绳'")
print("=" * 60)
for beta in [0.05, 0.5, 5.0]:
    p = nn.Parameter(torch.zeros(V))
    o = torch.optim.AdamW([p], lr=0.1)
    for _ in range(60):
        m = (seq_logp(p, CHOSEN) - seq_logp(ref, CHOSEN)) - (seq_logp(p, REJECTED) - seq_logp(ref, REJECTED))
        l = -F.logsigmoid(beta * m)
        o.zero_grad(); l.backward(); o.step()
    with torch.no_grad():
        probs = F.softmax(p, -1)
    print(f"  beta={beta:<4} -> P(✓)={probs[CHOSEN]:.3f}  P(✗)={probs[REJECTED]:.3f}  其他token概率总和={1-probs[CHOSEN]-probs[REJECTED]:.3f}")
print("""
beta 小：模型敢大幅改动（可能顺带把无关知识带偏）
beta 大：步子小、稳，但对齐慢 —— 工程上常取 0.1~0.5""")

print("=" * 60)
print("第 5 步：DPO vs PPO 一句话总结 + 作业")
print("=" * 60)
print("""
PPO : 真强化学习，要奖励模型+采样+优势估计，强但难伺候
DPO : 把偏好直接变成分类式损失，简单稳定，但上限略低、
      容易把概率质量过度集中（配合参考模型+beta缓解）
近年的 GRPO/CISPO 是两者的改进混血（DeepSeek-R1 用 GRPO 训推理）。

作业：
1. 把 CHOSEN/REJECTED 换成别的 token 对，验证学的是'相对偏好'
2. 思考：DPO 训练时 ref 模型为什么要冻结？（提示：否则'差值'失去参照系）""")
