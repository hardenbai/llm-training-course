# 周 2：预训练 Pretrain

## 目标
完整跑通一次预训练，理解训练动态。

## Day 1-2：数据管线
- [ ] 阅读 `repos/minimind/dataset/` 下各 README 与脚本：json → 清洗 → tokenize → 二进制分块存储
- [ ] 下载数据集（见主 README，用 `modelscope download` 或 HF 镜像 `HF_ENDPOINT=https://hf-mirror.com`）
- [ ] 理解为什么存成二进制段而不是 json

## Day 3-4：训练循环
- [ ] 阅读 `trainer/train_pretrain.py` 与 `trainer/trainer_utils.py`：
  - [ ] 混合精度训练（bf16/fp16）与梯度缩放
  - [ ] 学习率调度：warmup + cosine decay
  - [ ] 梯度裁剪 clip_grad_norm_
  - [ ] 损失：交叉熵，只预测下一个 token
- [ ] 跑一次完整预训练（~2h，单卡）

## Day 5-7：观察与分析
- [ ] 看 loss 曲线：warmup 阶段、下降速度、平台期
- [ ] 用训练中的 checkpoint 生成文本，观察从"乱码"到"通顺"的变化
- [ ] 理解 wandb / tensorboard 记录的指标

## 作业（核心实验）
改一个超参重训，对比 loss 曲线：
- 学习率 ×3 和 ÷3（观察发散与收敛缓慢）
- 或 batch size 减半（观察抖动）

## 自测问题
- 为什么学习率需要 warmup？
- bf16 和 fp16 的区别？为什么优先 bf16？
- 预训练损失是"每个 token 的平均交叉熵"，它和困惑度 PPL 什么关系？
