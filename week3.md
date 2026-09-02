# 周 3：SFT 与 LoRA

## 目标
把"会接话"的基座模型变成"会对话"的助手，并理解参数高效微调。

## Day 1-2：SFT 数据与训练
- [ ] 阅读 SFT 数据格式：多轮对话、system/user/assistant 角色
- [ ] 理解 chat 模板（chat template）如何拼接对话
- [ ] 阅读 `trainer/train_full_sft.py`：
  - [ ] 关键：损失只对 assistant 部分计算（loss mask）
  - [ ] 多轮对话中每轮 assistant 都算损失
- [ ] 下载 SFT 数据集，基于周 2 的 pretrain 模型跑 SFT

## Day 3-4：对话测试
- [ ] 用 `chat.py` / WebUI 和模型对话
- [ ] 理解 temperature、top-p 采样对生成的影响

## Day 5-7：LoRA
- [ ] 原理：W + BA，B 初始为 0，只训练低秩矩阵
- [ ] 阅读 `trainer/train_lora.py`，对比全参微调的可训练参数量
- [ ] 跑一次 LoRA 微调

## 作业
1. 手写 20 条带固定"人设"的对话数据（jsonl），SFT 后验证模型是否学会
2. 回答：为什么 B 初始化为 0？r=8 时可训练参数大约减少多少？

## 自测问题
- SFT 和 pretrain 的损失计算有什么本质区别？
- LoRA 为什么对大模型省显存？省的是哪部分？
