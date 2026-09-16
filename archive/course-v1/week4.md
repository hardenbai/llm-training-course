# 周 4：对齐与强化学习（DPO / GRPO）

## 目标
理解并跑通"人类偏好对齐"，这是 ChatGPT 区别于普通基座的关键一环。

## Day 1-3：DPO（直接偏好优化）
- [ ] 前置概念：RLHF 三阶段（SFT → 奖励模型 → PPO），DPO 如何跳过奖励模型
- [ ] 阅读 `trainer/train_dpo.py`：
  - [ ] 偏好数据格式：(prompt, chosen, rejected)
  - [ ] 损失：被接受与被拒绝回答的相对对数概率差 + sigmoid
  - [ ] Beta 参数的作用（越保守越贴近参考模型）
- [ ] 下载偏好数据集，跑 DPO 训练

## Day 4-6：GRPO / RL（进阶，可选）
- [ ] 理解 policy、reward、rollout、advantage（组内相对优势）
- [ ] 阅读 `trainer/train_grpo.py`（PPO 在 `train_ppo.py`，智能体 RL 在 `train_agent.py`）
- [ ] 若跑不动 8GB，读懂代码 + 看作者博客即可

## Day 7：总结
- [ ] 画出完整链路：pretrain → SFT → DPO →（RL），每步的输入/输出/损失
- [ ] 用同一个问题对比三个阶段模型的回答差异
- [ ] 进入 CS336 理论阶段（见 README 第 2 阶段）

## 作业
主观盲测：把 SFT 模型和 DPO 模型的回答打乱，自己打分，验证 DPO 是否真的"更对齐"。

## 自测问题
- DPO 相比 PPO 省掉了什么？代价是什么？
- reward hacking 是什么？为什么 RL 阶段容易发生？
`n## 配套脚本`n- Day1-3: lessons/week4_day1_dpo.py（DPO从零实现）`n- Day7: lessons/week4_day2_capstone.py（毕业项目：三阶段流水线）
