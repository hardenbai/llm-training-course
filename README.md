# 大模型训练系统化学习课程

> 目标：从零理解并亲手实践大模型训练全流程（预训练 → SFT → RLHF/RL）
> 硬件：RTX 4060 Ti 8GB —— 足够跑通本课程所有实验
> 核心材料：MiniMind（动手）+ Stanford CS336（理论）+ Marin（真实研究）

## 学习路径总览

```
第 0 阶段：环境准备（已完成 ✅）
第 1 阶段：MiniMind 从零实现（4 周，动手为主）
第 2 阶段：CS336 理论补强（3 周，视频 + 作业）
第 3 阶段：Marin 真实研究阅读（持续，当研究材料读）
```

---

## 第 0 阶段：环境（已就绪）

- [x] Python (Miniconda) + Git
- [x] PyTorch CUDA（4060 Ti 8GB）
- [x] MiniMind 仓库克隆到 `repos/minimind`
- [x] 依赖安装

## 第 1 阶段：MiniMind 从零实现（核心，4 周）

仓库位置：`repos/minimind`，模型只有 26M~104M 参数，4060 Ti 单卡可训。

### 周 1：Tokenizer 与语言模型基础
- 阅读 `repos/minimind/README.md` 全文
- 学习 `model/model.py`：逐行理解 Token Embedding、RoPE、RMSNorm、SwiGLU、KV-Cache
- 运行 `train_minimind.py` 里的 tokenizer 训练，理解 BPE
- **作业**：手画 Transformer 一层的计算图，标注张量形状

### 周 2：预训练 Pretrain
- 阅读 `pretrain_preprocess.py`（数据清洗→二进制分块）与 `train_pretrain.py`
- 下载预训练数据集，跑一次完整 pretrain（~2h）
- 观察loss曲线，理解学习率调度（warmup + cosine）、梯度裁剪
- **作业**：改一个超参（如 lr、batch size），对比 loss 曲线

### 周 3：SFT 与 LoRA
- `train_sft.py`：理解 chat 模板、因果注意力 mask、损失只算 answer 部分
- `train_lora.py`：理解低秩分解为什么省显存
- 用自己写的几条对话数据做微调，和模型聊天验证
- **作业**：让模型学会一个新"人设"

### 周 4：对齐与强化学习
- `train_dpo.py`：偏好对，理解被拒绝/被接受样本的相对损失
- `train_grpo.py`（或 PPO/CISPO 分支）：理解 reward、rollout、advantage
- **作业**：跑通 DPO，用主观打分对比 SFT 前后模型

## 第 2 阶段：CS336 理论补强（3 周）

课程主页：https://stanford-cs336.github.io/spring2024/ （视频在 YouTube 搜 "Stanford CS336"）

按 MiniMind 实践过的内容去听理论，事半功倍：
1. Tokenization & Architecture（对应周 1）
2. Training dynamics & Mixed precision（对应周 2）
3. Scaling laws & Data（理解 Chinchilla 定律）
4. Alignment / RLHF 理论（对应周 4）
5. Inference & Serving（KV-Cache、量化）

## 第 3 阶段：Marin 真实研究阅读（持续）

仓库：https://github.com/marin-community/marin ，文档：https://marin-community.github.io/marin/

推荐阅读顺序：
1. 官方教程：train a tiny model（看真实工程的数据管线）
2. DCLM / 1B 实验的 experiment 脚本（代码即实验记录）
3. 8B retrospective 博客：看失败实验和决策过程
4. Scaling law 相关实验（Delphi suite）

---

## 环境速查

```bat
:: 激活虚拟环境（CMD）
C:\Users\Administrator\.zcode\workspace\default\llm-training-course\venv\Scripts\activate.bat

:: 进入 MiniMind
cd C:\Users\Administrator\.zcode\workspace\default\llm-training-course\repos\minimind

:: 数据集下载走 HF 镜像（建议写入系统环境变量）
set HF_ENDPOINT=https://hf-mirror.com
```

详细环境说明见 [env.md](env.md)，每周任务明细见 [week1.md](week1.md) ~ [week4.md](week4.md)。
