# 周2 Day1：预训练数据管线 —— 1.2GB 文本怎么喂进显卡
# 对照源码: repos/minimind/dataset/lm_dataset.py 第 37-55 行 (PretrainDataset)
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "repos" / "minimind" / "dataset"
CACHE = ROOT / "lessons" / "artifacts"
CACHE.mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ROOT / "repos" / "minimind" / "model")

print("=" * 60)
print("第 1 步：切一小块真实数据出来（完整文件 1.2GB，先取 3000 行学习）")
print("=" * 60)
sample_file = CACHE / "pretrain_sample.jsonl"
with open(DATA / "pretrain_t2t_mini.jsonl", encoding="utf-8") as f, \
     open(sample_file, "w", encoding="utf-8") as out:
    for i, line in enumerate(f):
        if i >= 3000: break
        out.write(line)
print(f"已生成: {sample_file.name}（{sample_file.stat().st_size/1e6:.1f} MB）")
with open(sample_file, encoding="utf-8") as f:
    first = json.loads(f.readline())
print(f"一条原始数据的样子: text = {first['text'][:60]}...")

print()
print("=" * 60)
print("第 2 步：PretrainDataset 对每条数据做了什么（对照源码 47-55 行）")
print("=" * 60)
MAX_LEN = 256
def process(text):
    # 源码逻辑：截断 -> 加 bos/eos -> pad 到定长 -> 生成 labels
    ids = tok(text, add_special_tokens=False, max_length=MAX_LEN - 2, truncation=True).input_ids
    ids = [tok.bos_token_id] + ids + [tok.eos_token_id]
    input_ids = ids + [tok.pad_token_id] * (MAX_LEN - len(ids))
    labels = input_ids.copy()
    for j in range(len(ids), MAX_LEN):
        labels[j] = -100          # padding 位置不参与 loss（Day3 讲过 -100）
    return ids, input_ids, labels

ids, input_ids, labels = process(first["text"])
print(f"原文前 40 字: {first['text'][:40]}")
print(f"真实 token 数: {len(ids)}（含首尾特殊token）")
print(f"补齐到定长:   {MAX_LEN}")
print(f"input_ids 头部: {input_ids[:8]}  (1=bos开头)")
print(f"labels   头部: {labels[:8]}")
print(f"labels   尾部: {labels[-8:]}  (-100=padding,不计损失)")
real = sum(1 for l in labels if l != -100)
print(f"有效 token: {real}/{MAX_LEN}（{real/MAX_LEN:.0%}）")

print()
print("=" * 60)
print("第 3 步：为什么所有序列必须'等长'？")
print("=" * 60)
print("""
GPU 是'打包计算'的：一批矩阵必须形状一致才能并行。
类比：卡车运货，不管每单多大都装进统一规格的箱子(batch内等长)，
箱子没装满的部分填泡沫(padding)，运费只算真货(-100不计损失)。
代价：有效token比例越低，浪费越多 -> 这就是数据预处理要
'相近长度的样本尽量凑一批'的原因。""")

print()
print("=" * 60)
print("第 4 步：算一笔账 —— 数据量与训练的关系")
print("=" * 60)
import itertools
with open(sample_file, encoding="utf-8") as f:
    n_lines = sum(1 for _ in f)
    total_tokens = 0
    for line in itertools.islice(f, 0, 0): pass
with open(sample_file, encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)["text"]
        total_tokens += min(len(tok(t, add_special_tokens=False).input_ids), MAX_LEN - 2)
avg = total_tokens / n_lines
print(f"样本数: {n_lines}, 平均每条 {avg:.0f} tokens")
print(f"这 3000 条样本共约 {3000*avg/1e6:.2f}M tokens（完整文件还要大两个数量级）")
print(f"MiniMind 官方完整预训练约 1B tokens = 这个量的 {1e9/(3000*avg):.0f} 倍")
print("""
公式感：模型见过多少 token，直接决定它多聪明(Chinchilla定律：
最优训练token数 ≈ 模型参数数 x 20)。数据管线看似枯燥，
但'喂什么'比'怎么训'更影响最终质量。""")
print("明天 Day2：正式写训练循环，在你显卡上真跑起来。")
