# 进阶资源手册：手搓预训练 · 后训练 · Infra

> 前置：完成本课程四周（MiniMind 全流程）。
> 目标：从"会用 MiniMind 的代码"进阶到"自己从空白文件搓出"。
> 硬件现实：单卡 4060 Ti 8GB —— 能搓的都标了 ✅，只能读代码理解的标了 📖。

## 三条路线总览（可并行，建议顺序：预训练 → 后训练 → infra）

```
手搓预训练                手搓后训练                  手搓 infra
─────────────           ─────────────             ─────────────
CS336 Assignment1       SFT/DPO 自己写            KV-Cache 推理引擎
nanoGPT 精读            TinyZero (GRPO)           nano-vllm 精读
happy-llm (中文)        LLaMA-Factory/Unsloth     llama.cpp 量化部署
从零重写 train_pretrain  给自己的模型写GRPO         Flash Attention 原理
```

---

## 一、Stanford CS336《Language Modeling from Scratch》（理论+作业主线）

- 课程主页（每年更新）：https://cs336.stanford.edu/ ｜ 2024 存档：https://cs336.stanford.edu/spring2024/
- GitHub 组织（全部仓库）：https://github.com/stanford-cs336
- 讲义代码（可执行的讲座）：https://github.com/stanford-cs336/lectures
- **作业（课程的灵魂，"巨大且深入"，Assignment1 的 PDF 就有 50+ 页）**：
  - https://github.com/stanford-cs336/assignment1-basics —— 从 BPE tokenizer、Transformer、AdamW 全部从零实现，带完整测试
  - 后续 assignment 覆盖 scaling laws、数据、系统（见组织仓库列表）

**与本课程的对接表**（学完对应周就去做 CS336 的对应部分）：

| 我们的周 | CS336 对应内容 | 做什么 |
|---|---|---|
| 周1（模型结构） | Assignment1 的 BPE + Transformer 部分 | 用自己的代码（非抄）重实现 tokenizer 和 attention，跑通它的测试 |
| 周2（预训练） | Assignment1 的优化器 + 训练部分 | 手写 AdamW，理解为什么不用 SGD |
| 周3-4（后训练） | 对应 alignment 讲座 | 看讲座即可，动手项目走下面的后训练线 |

社区参考实现（卡住时对照，别照抄）：https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments

---

## 二、手搓预训练线

### 1. karpathy/nanoGPT 📖✅
https://github.com/karpathy/nanoGPT
- ~300 行核心代码训 GPT-2，业界公认的"最佳起点"
- 用法：通读 `model.py` + `train.py`，对照你已学的 MiniMind——你会发现自己全能看懂
- 8GB 可跑：Shakespeare char-level 版本几分钟出结果

### 2. karpathy/llm.c 📖
https://github.com/karpathy/llm.c
- 纯 C/CUDA 写 GPT-2 训练，无 Python 依赖
- 用法：读懂它 = 理解"训练循环在硬件层面到底在干什么"（infra 线也用它）

### 3. datawhalechina/happy-llm ✅（中文首选）
https://github.com/datawhalechina/happy-llm
- Datawhale 出品，从 0 搭 215M mini-Llama2，覆盖预训练+微调，13k+ star
- 用法：作为 MiniMind 之后的第二遍全流程，重点看它和 MiniMind 的不同选择

### 4. datawhalechina/llms-from-scratch-cn ✅（中文）
https://github.com/datawhalechina/llms-from-scratch-cn
- Sebastian Raschka《LLMs from Scratch》的 Datawhale 中文实践版，仅需 Python 基础

### 5. datawhalechina/tiny-universe 📖✅
https://github.com/datawhalechina/tiny-universe
- "白盒"导向的大模型全链路手搓指南，适合查漏补缺单项深入

### 手搓项目（预训练毕业考核）
> 打开空白文件，不参考 MiniMind，写 `my_train.py`：BPE 分词数据加载 → 自实现
> Transformer（RoPE+因果掩码+KV-Cache）→ 训练循环（warmup+cosine+梯度裁剪）
> → 在 pretrain_t2t_mini.jsonl 上训到 loss < 6.0。
> 卡住才能翻 MiniMind，这就是"手搓"的考核标准。

---

## 三、手搓后训练线

### 1. MiniMind 自带脚本 ✅（已在课程里跑过教学版）
`trainer/train_full_sft.py`、`train_dpo.py`、`train_grpo.py`、`train_ppo.py`、`train_agent.py`——
完整版跑一遍，这是最小成本的"全后训练菜单"。

### 2. datawhalechina/self-llm ✅（中文，微调实战大全）
https://github.com/datawhalechina/self-llm
- 开源大模型食用指南：部署+使用+微调（含 LLaMA-Factory 章节）

### 3. hiyouga/LLaMA-Factory ✅
https://github.com/hiyouga/LLaMA-Factory
- 100+ 模型的统一微调框架，无代码 GUI，可挂 Unsloth 后端提速
- 用法：给自己的 Qwen/MiniMind 模型做一次"工业级"微调，对比手搓版的差距

### 4. unslothai/unsloth ✅（单卡之王，正合适你的 8GB）
https://github.com/unslothai/unsloth
- 单卡训练快 ~2 倍、省 ~70% 显存；社区共识：消费级显卡首选
- 用法：用它 LoRA 微调 Qwen2.5-1.5B，体验"8GB 也能玩 1.5B 模型"

### 5. Jiayi-Pan/TinyZero 📖（GRPO 里程碑）
https://github.com/Jiayi-Pan/TinyZero
- Berkeley 用 ~$30 复现 DeepSeek R1-Zero 的"aha moment"，GRPO 训 Qwen 小模型
- 8GB 跑不动原版（需 ~2×80GB 或租卡），但 notebook 可读 + 换 0.5B 模型+LoRA 有机会
- 配套：philschmid 的 mini-deepseek-r1 GRPO notebook（同 repo 系）

### 6. huggingface/trl ✅
https://github.com/huggingface/trl
- SFT/DPO/PPO/GRPO 官方实现，MiniMind 同款依赖；读它的 DPOTrainer 源码对照你的手搓版

### 手搓项目（后训练毕业考核）
> 给自己预训练的模型写简化版 GRPO：倒计时任务（生成数字算 24 点），
> 规则奖励（全对=1, 部分=0.x）+ 组内相对优势，训 500 步看 reward 曲线上涨。

---

## 四、Infra 线

### 1. GPU MODE 讲座 📖（CUDA/GPU 性能第一课）
https://github.com/gpu-mode （100+ 免费讲座）
- 必看：Lecture 12 Flash Attention（笔记：christianjmills.com/posts/cuda-mode-notes/lecture-012/）
- 用法：每周看 1-2 讲，配合作业；理解 Triton 写 kernel

### 2. GeeCGH/nano-vLLM 📖✅
https://github.com/GeeCGH/nano-vLLM
- DeepSeek 工程师写的 1200 行迷你 vLLM：PagedAttention、prefix caching、CUDA Graph
- 用法：先读懂，然后对照自己写的"朴素推理引擎"找差距

### 3. flash-attention 📖
https://github.com/Dao-AILab/flash-attention
- 注意力 kernel 的工业标准；读论文+看 GPU MODE 讲座即可，CUDA 实现选学

### 4. ggml-org/llama.cpp ✅（量化部署，本机可玩）
https://github.com/ggml-org/llama.cpp
- 用法：把 MiniMind 模型转 GGUF → int8/int4 量化 → 本机 CPU 推理
- 量化前后对比生成质量和速度 = 一份真实的"部署工程"作业

### 5. microsoft/DeepSpeed + NVIDIA/Megatron-LM 📖
https://github.com/microsoft/DeepSpeed ｜ https://github.com/NVIDIA/Megatron-LM
- 分布式训练（ZeRO、张量/流水线并行）；单卡玩不了，读概念+看图理解即可
- 了解"8 张 4060 Ti 也训不动的模型是怎么训的"

### 手搓项目（infra 毕业考核）
> 给自己训的模型写 `my_engine.py`：KV-Cache 增量生成 + batch 推理（padding 对齐）
> + top-p/top-k 采样 + 简单 prefix cache，对比 naive 实现的 tokens/s。
> 再用 llama.cpp 转 GGUF 量化部署，写一份"部署报告"。

---

## 五、推荐节奏（课程结束后 8-10 周）

| 周 | 做什么 |
|---|---|
| 1-2 | CS336 Assignment1（BPE+Transformer 部分）+ nanoGPT 精读 |
| 3 | 手搓项目①：从零重写预训练（见上） |
| 4 | 完整跑 MiniMind 官方版 train_pretrain(2h) → SFT → DPO 全链 |
| 5 | happy-llm 第二遍全流程（中文对照）+ LLaMA-Factory/Unsloth 微调 Qwen |
| 6 | 手搓项目②：简化版 GRPO + TinyZero 论文/代码精读 |
| 7-8 | GPU MODE 讲座（Flash Attention 等）+ nano-vLLM 精读 |
| 9 | 手搓项目③：推理引擎 + llama.cpp 量化部署 |
| 10 | 回到 Marin（github.com/marin-community/marin）：读 8B retrospective 的失败实验复盘——现在你读得懂了 |

## 8GB 显存的诚实边界
- ✅ 能搓：7B 以下模型推理、1.5B LoRA 微调（Unsloth）、0.5B 全流程 RL（LoRA）、自训 <1B 模型、量化部署
- 📖 只能读：GRPO 原版规模、DeepSpeed/Megatron 实操、百亿级预训练
- 想实操大项目：租卡（AutoDL 等，4090 约 ¥2/小时），TinyZero 复现约几十元
