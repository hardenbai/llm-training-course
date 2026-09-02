# 周 1：Tokenizer 与语言模型基础

## 目标
理解"文本 → token → 概率分布"的完整链路，读懂 MiniMind 的模型代码。

## Day 1-2：整体认知
- [ ] 通读 `repos/minimind/README.md`
- [ ] 理解因果语言模型的目标：P(x_t | x_1..x_{t-1})
- [ ] 运行 README 里的快速开始，先"不明所以地跑通一次"

## Day 3-4：Tokenizer
- [ ] 理解 BPE 算法（子词切分，为什么不用字符/单词）
- [ ] 阅读并运行 `repos/minimind/trainer/train_tokenizer.py`（产出 model/tokenizer.json）
- [ ] 观察：同一句话如何被切成 token

## Day 5-7：模型结构（重点）
逐行读 `repos/minimind/model/model_minimind.py`（LoRA 结构在 `model/model_lora.py`），对照以下概念：
- [ ] Token Embedding：查表，形状 (batch, seq) → (batch, seq, dim)
- [ ] RMSNorm：为什么比 LayerNorm 省算力
- [ ] RoPE 旋转位置编码：相对位置如何编码进 Q/K
- [ ] SwiGLU：前馈层的门控结构
- [ ] KV-Cache：推理时为什么能省重复计算
- [ ] MoE（选读）：专家路由与负载均衡

## 作业
1. 手画一层 Transformer 的计算图，每个张量标注形状
2. 回答：GPT 和 BERT 的注意力 mask 有什么区别？为什么？

## 自测问题
- 为什么推理时 KV-Cache 有效，而训练时不用？
- RoPE 相对可学习位置编码的优势是什么？
