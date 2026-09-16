# 来源、边界与许可

本项目重新组织并原创绘制全部讲解、图形和交互。外部材料用于核对概念、选择案例和定位真实代码，不直接复制其章节或图示。

## 三个主要参考项目

### 《深入理解 AI Infra：量化分析与系统设计》

- 仓库：<https://github.com/bojieli/ai-infra-book>
- 许可：Apache License 2.0
- 使用范围：从硬件约束推导系统设计；计算、存储、通信与依赖关系；训练和推理中的数据搬移。
- 建议延伸阅读：第 1、3、4、6、8、9、10 章及配套计算工具。

### MiniMind

- 仓库：<https://github.com/jingyaogong/minimind>
- 许可：Apache License 2.0
- 使用范围：Tokenizer、Dense/MoE 模型结构、预训练、SFT、DPO/GRPO、推理与部署的端到端代码映射。
- 边界：“2 小时”是特定硬件、数据和阶段下的项目实测，不在本课程中作为完整预训练的普遍承诺。

### nanochat

- 仓库：<https://github.com/karpathy/nanochat>
- 许可：MIT
- 使用范围：计算最优模型系列、训练 speedrun、数据加载、优化器、评测、KV Cache 推理以及从实验到指标的闭环。
- 边界：其 GPT-2 级实验依赖多块数据中心 GPU；本课程只把它作为现代训练系统案例，不声称消费级显卡能复现相同性能。

## 硬件与并行系统的一手资料

- [NVIDIA DGX H100/H200 系统说明](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)
- [Google Cloud TPU 系统架构](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [PyTorch FSDP 文档](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch Tensor Parallel 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [华为昇腾文档中心](https://www.hiascend.com/document)

## 估算约定

- 参数显存使用十进制 GB，`1B 参数 × 1 byte ≈ 1 GB`。
- 混合精度 AdamW 的训练状态按约 `16 bytes/参数` 展示，用于解释数量级；具体实现会因优化器、分片、量化和框架而变化。
- KV Cache 使用 `2 × 层数 × KV heads × head_dim × 每元素字节数 × token 数 × batch`。
- 通信量展示算法级近似，不包含协议、拓扑、拥塞和计算通信重叠带来的修正。
- 峰值算力、峰值带宽和真实有效吞吐严格区分；网页不以营销峰值预测真实训练时间。
