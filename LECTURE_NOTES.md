# 《AI Infra：从模型到机器》详细讲义

> 面向已经了解 AI 产业、希望把模型、系统与硬件连成因果链的听众。核心路线 75 分钟，加入代码实验后约 90 分钟。

## 导航

- [课程目标](#1-课程目标)
- [授课准备与时间安排](#2-授课准备)
- [五个通用公式](#3-全场通用的五把尺子)
- [逐章讲义](#4-逐章讲义)
- [15 分钟代码演示](#5-15-分钟代码演示-runbook)
- [常见追问](#6-常见追问)
- [内容边界与延伸阅读](#7-内容边界与延伸阅读)

## 1. 课程目标

这不是芯片型号介绍，也不是模型结构复习。结束时，听众应能面对一个新的模型或业务，依次回答：

1. 张量是什么形状：batch、sequence、hidden、expert、图像 patch 或 latent 分别多大？
2. 状态是否装得下：权重、梯度、优化器、激活、KV Cache 谁占容量？
3. 每一步搬什么：存储到 Host、DDR 到 HBM、设备到设备分别传什么？
4. 计算单元在等什么：等矩阵、等 HBM、等集合通信，还是等 CPU/存储？
5. 优化改变了哪一项：计算、容量、字节数、通信频率还是尾延迟？

贯穿全场的一句话：**模型架构决定数据流；数据流决定等待；AI Infra 的价值是减少这些等待。**

## 2. 授课准备

启动网页：

```bash
python -m http.server 8000
```

打开 <http://localhost:8000>。使用左右方向键换章，滚轮或上下方向键滚动当前章节，空格暂停动画。

预演六个无需第三方依赖的实验：

```bash
python labs/02_memory_budget.py --preset 70b --context 8192 --batch 8 --tp 8
python labs/04_parallelism_cost.py --mode all --tokens 4096 --gpus 8
python labs/05_roofline_prefill_decode.py --decode-batch 1
python labs/06_moe_routing.py --skew 0.7
python labs/07_continuous_batching.py
python labs/08_workload_shapes.py
```

### 时间安排

| 模块 | 核心路线 | 完整路线 |
| --- | ---: | ---: |
| 开场与全生命周期 | 7 分钟 | 9 分钟 |
| 模型如何变成负载 | 8 分钟 | 10 分钟 |
| 一步训练与容量 | 10 分钟 | 13 分钟 |
| 多卡并行 | 11 分钟 | 14 分钟 |
| Prefill、Decode、KV Cache | 13 分钟 | 16 分钟 |
| 在线服务系统 | 7 分钟 | 10 分钟 |
| 硬件地图 | 8 分钟 | 8 分钟 |
| 决策实验与总结 | 11 分钟 | 10 分钟 |

## 3. 全场通用的五把尺子

### 参数计算

```text
前向线性层 FLOPs ≈ 2 × 活跃参数量 × token 数
训练 FLOPs 数量级 ≈ 6 × 参数量 × token 数
```

真实结果还包含 Attention、归一化、激活函数和框架开销。

### 训练模型状态

| 状态 | 教学估算 |
| --- | ---: |
| BF16 参数 | 2 bytes/param |
| BF16 梯度 | 2 bytes/param |
| FP32 主参数 | 4 bytes/param |
| Adam 一阶矩 | 4 bytes/param |
| Adam 二阶矩 | 4 bytes/param |
| 合计 | 约 16 bytes/param |

所以 70B 仅模型状态约为 1.12TB，还没有算激活、临时 buffer 与碎片。

### KV Cache

```text
KV bytes = 2 × layers × KV heads × head_dim
           × context × batch × bytes/value
```

第一项 2 代表 Key 和 Value。GQA/MQA 减少 KV heads，因此直接压缩 KV Cache。

### Roofline

```text
算术强度 = FLOPs / 搬移字节数
硬件拐点 = 峰值 FLOPS / HBM 带宽
```

算术强度低于拐点时更可能受带宽限制，高于拐点时更可能受计算限制。

### 通信时间

```text
通信时间 ≳ 消息字节数 / 有效带宽 + 通信轮次 × 单轮延迟
```

因此总流量不大仍可能很慢：如果每层都有小消息，就会重复支付延迟。

---

## 4. 逐章讲义

## 00 开场：模型为什么需要这些机器（3 分钟）

先问：**如果 GPU 峰值算力翻倍，一个大模型请求一定快一倍吗？** 不立即回答，把问题留到推理章节。

指向网页中央的数据搬移：计算单元变换数据，HBM/DDR/存储保存不同寿命的状态，PCIe、NVLink/ICI、RDMA 网络让数据跨边界移动。如果下一批数据没有及时到达，再多乘加单元也会空转。

只留下三句话：

1. 峰值算力不等于有效算力；
2. 模型架构会落成具体的访存与通信模式；
3. 训练、Prefill、Decode 是不同工作负载。

转场：**要知道机器为什么不同，先跟着数据走完一生。**

## 01 全生命周期：数据不断改变形态（6 分钟）

在网页依次点击五个阶段。每到一个阶段，都问：输入是什么、要保存什么、最可能等谁？

| 阶段 | 输入/输出 | 常驻状态 | 关键硬件 | 常见等待 |
| --- | --- | --- | --- | --- |
| 数据进入 | 对象/文件/样本 | 原始语料、索引 | 对象存储、NVMe、网络 | 小文件、元数据、吞吐 |
| 清洗分词 | 文本 → token IDs | tokenizer、数据 shard | CPU、DDR、存储 | 解压、解析、随机读取 |
| 训练 | token → 梯度 → 参数 | 参数、激活、梯度、优化器 | 加速器、HBM、互联 | GEMM、HBM、All-Reduce |
| Checkpoint | 训练状态 → 文件 | 参数、优化器、随机状态 | HBM、DDR、并行文件系统 | 同步停顿、写带宽 |
| 在线推理 | 请求 → token 流 | 权重、KV Cache、队列 | 加速器、HBM、网络 | TTFT、TPOT、尾延迟 |

深化两点：

- Checkpoint 不只是权重。无缝恢复还需要优化器、调度器、随机数、数据加载位置和并行元数据。
- CPU 没有消失。数据解析、Tokenizer、请求调度、算子发射与内存管理仍在 Host；GPU 利用率低也可能是 Host 喂不饱。

现场提问：训练和推理都叫“加载模型”，它们要保存的东西有何不同？答案是推理主要保存权重与 KV，训练还要保存激活、梯度和优化器状态。

## 02 模型不是名字，而是工作负载（9 分钟）

网页依次切换 Dense、MoE、VLM、Diffusion，重点看它们如何改变张量形状和重复模式。

### Dense Transformer

每个 token 经过同一组参数，计算规则，容易形成大 GEMM。训练常见计算与激活压力；低 batch Decode 中权重复用不足，会转向 HBM 带宽压力。

### MoE

MoE 有两个参数量：总参数决定权重容量，活跃参数更接近每 token 计算与权重读取。Router 执行 Top-k，专家跨设备时产生 dispatch/combine All-to-All；热点专家会形成长尾。

MiniMind 的实现正好体现这条路径：Router scores → Top-k expert index → 按 expert 聚合 token → 加权合并；训练时增加辅助负载均衡损失。

### VLM

图像先变成 patch/token。若高和宽都翻倍，图像 token 数约变四倍，全注意力 score 元素可能约变十六倍。额外压力不只是视觉编码器，还包括更长 Prefill、跨模态位置编码和中间激活。

### Diffusion

Diffusion 在 latent 网格上重复执行去噪网络：总工作量约等于单步工作量乘去噪步数。它更强调 latent 分辨率、重复迭代、VAE 前后处理，而不是 LLM 式 KV Cache。

演示：

```bash
python labs/08_workload_shapes.py
python labs/08_workload_shapes.py --image-size 896 --denoise-steps 50
```

先让听众预测图像边长翻倍的影响，再看输出。脚本中的 2P×positions 只是线性层数量级，不是完整 FLOPs。

## 03 一步训练：同一批 token 的三次生命（11 分钟）

### 前向

读取参数和输入，执行 Embedding、Attention、FFN、残差与归一化，写出 logits，并保存反向传播所需激活。大矩阵有利于计算单元利用率，但激活随 micro-batch、sequence、hidden、layers 增长。

### 反向

从 loss 按计算图逆序传播，读取保存的激活，生成参数梯度与上游梯度。数据并行时，梯度 bucket 会逐步触发 All-Reduce。反向通常比前向更重，因为既要算输入梯度，也要算权重梯度。

### 优化器

AdamW 的数学不复杂，却要读写参数、梯度、一阶矩、二阶矩和可能存在的 FP32 主参数，所以常表现为大规模 HBM 流式读写。

### 梯度累积与 Activation Checkpointing

MiniMind 与 nanochat 的训练循环都展示了梯度累积：多个 micro-batch 前向/反向后才更新一次参数。它降低单次激活峰值，不减少完成同样 token 所需的总计算，也不会让未分片模型状态消失。

Activation Checkpointing 不保存部分激活，反向时重新计算，是典型的“用算力换容量”。

### 演示 A：真实训练 step

```bash
python labs/01_training_step.py --batch 8 --seq 32 --width 64
python labs/01_training_step.py --batch 16 --seq 64 --width 64
```

观察参数、梯度、Adam 状态与前向张量的增长方式。脚本中的 hook 只统计几个可见张量，不等于 autograd 精确峰值；若用 CUDA，会额外输出 allocator 峰值。

### 演示 B：70B 为什么必须分片

```bash
python labs/02_memory_budget.py --preset 70b --context 8192 --batch 8 --tp 8 --shards 8
```

讲 ZeRO 时不要背名称，只看谁开始分片：

| 阶段 | 分片对象 |
| --- | --- |
| ZeRO-0 | 不分片 |
| ZeRO-1 | 优化器状态 |
| ZeRO-2 | 优化器状态 + 梯度 |
| ZeRO-3 / FSDP FULL_SHARD | 参数 + 梯度 + 优化器状态 |

常见误区：

- “8 张 80GB 卡共有 640GB，所以 70B 一定装得下。”若没有状态分片，每卡仍可能保留完整副本。
- “混合精度就是所有状态都用 BF16。”优化器常保留 FP32 状态。
- “梯度累积等于扩大显存。”它主要降低 micro-batch 激活峰值。

## 04 多卡并行：切开工作，也切出通信（12 分钟）

每种并行只回答两个问题：**切了什么？必须传什么？**

| 策略 | 切分对象 | 主要通信 | 频率 | 关键约束 |
| --- | --- | --- | --- | --- |
| DP | batch | 梯度 All-Reduce | 每个训练 step，常分 bucket | 模型副本、跨节点带宽 |
| TP | 矩阵/hidden/head | 激活 All-Reduce/All-Gather | 每层多次 | 低延迟 scale-up 互联 |
| PP | layers | stage 边界激活/梯度 | 每个 micro-batch | pipeline bubble |
| EP | experts | routed token All-to-All | 每个 MoE 层 | bisection 带宽、负载均衡 |

### DP

每张卡计算不同 micro-batch，反向后同步梯度。Ring All-Reduce 每卡通信量常用近似为：

```text
每卡流量 ≈ 2 × (N-1)/N × 梯度大小
```

实现会把梯度拆成 bucket，并尝试和反向计算重叠。

### TP

矩阵按行/列切分后，局部结果必须 All-Gather 或 All-Reduce。单次消息未必大，但每层都要通信，所以对延迟与节点内互联敏感。

### PP

每张设备保存连续层，stage 之间传边界激活。微批过少会出现 bubble；增加微批可提高填充率，但也增加调度和激活驻留复杂度。

### EP

Router 后把 token 发给对应 expert，再把结果发回。除平均流量外还要看热点 expert、capacity factor、token drop、expert placement 和跨节点比例。

### 演示：通信对象与频率

```bash
python labs/04_parallelism_cost.py --mode all --tokens 4096 --gpus 8 --bandwidth 400
```

让听众先按单次载荷排序，再按每步总流量/频率排序。DP 单次载荷可能大，TP 却在很多层反复支付通信。

### 演示：MoE 热点

```bash
python labs/06_moe_routing.py --tokens 4096 --experts 8 --top-k 2 --skew 0
python labs/06_moe_routing.py --tokens 4096 --experts 8 --top-k 2 --skew 0.7
```

第二次运行中，平均 assignments 不变，但最大 expert 负载、容量溢出与尾部等待显著变化。这就是辅助负载均衡损失存在的系统原因。

组合并行的初始原则：节点内高速互联优先承载高频 TP/EP；节点间网络优先承载相对低频 DP/PP；先解决容量，再在可行候选中比较吞吐、延迟与成本。

## 05 推理：Prefill 与 Decode 是两种世界（13 分钟）

### Prefill

一次处理整个提示词。大量 token 同时使用同一份权重，矩阵 M 维较大，权重被充分复用，算术强度较高。重点指标是 TTFT、长提示吞吐与大矩阵核效率。

### Decode

每步只为每个请求生成一个 token。低 batch 下计算很少，却要访问大量活跃权重，并读取持续增长的 KV Cache。重点指标是 TPOT、HBM 带宽、KV 容量和跨卡延迟。

### Roofline 演示

```bash
python labs/05_roofline_prefill_decode.py --decode-batch 1
python labs/05_roofline_prefill_decode.py --decode-batch 32
```

Prefill 中权重被数千 token 复用；batch=1 Decode 的算术强度接近个位数。增大 Decode batch 可提高权重复用，但会占用更多 KV，并可能提高排队时间。

回收开场问题：GPU 算力翻倍只有在当前瓶颈位于计算侧、软件又能利用新增算力时，才可能接近速度翻倍。

### KV Cache 到底缓存什么

每层 Attention 把历史 token 投影成 K/V。没有缓存时，每步都重新投影历史；有缓存时，只投影新 token，并追加 K/V。

- 避免历史 K/V 重复投影；
- 新 Q 仍要读取并关注全部历史 K/V；
- 计算减少，但容量和带宽压力上升。

```bash
python labs/03_kv_cache_demo.py --prompt 128 --generate 32
```

脚本逐 token 比较完整因果注意力与缓存输出，最大误差应接近浮点误差，同时报告 K/V 投影数和 attention score 元素数。nanochat 的推理引擎进一步展示了 batch=1 Prefill 后复制 Cache，再批量 Decode 多个样本的做法。

生产 KV 管理还包括 block/page 分配、碎片、prefix 共享、eviction、KV 量化、TP 切分，以及 Prefill/Decode 分离时的 KV 传输。

## 06 服务系统：优化对象是请求集合（8 分钟）

单次 model.forward 很快，不代表在线系统好。生产流量有不同到达时间、prompt 长度、输出长度、优先级、SLA 与 Cache 命中。

### Continuous Batching

静态批通常等整批全部完成才换批，短请求结束后留下空槽。Continuous Batching 在 token 边界重组，完成一个请求就补入新请求。

```bash
python labs/07_continuous_batching.py
python labs/07_continuous_batching.py --slots 4 --requests 32 --max-output 96
```

比较 makespan、slot-util 与 p95。真实系统中 batch 增大会改变单步耗时，因此槽位利用率不是唯一指标。

### Paged Attention

它不改变 Attention 数学，而是把 KV 切成固定块按需分配、回收和共享，减少连续大内存预留与外部碎片，提高可接纳请求数。

### Prefill/Decode 分离

Prefill 偏计算、Decode 偏带宽，分离后可分别选择 batch、并行度与资源。但会新增 KV 传输、资源池匹配、排队和故障边界，并非必然更优。

### Speculative Decode

小模型先提出多个候选，大模型并行验证，以减少大模型串行步数。收益取决于接受率、草稿成本与验证效率。

必须同时观察：

| 用户体验 | 系统效率 |
| --- | --- |
| TTFT | tokens/s/GPU |
| TPOT | slot/batch 利用率 |
| p50/p95/p99 | HBM/KV 使用率 |
| SLA 达标率 | 每百万 token 成本 |

只优化平均吞吐可能牺牲尾延迟；只优化单请求延迟可能导致资源闲置。

## 07 硬件地图：每种硬件解决一类等待（9 分钟）

在网页逐个点击部件，始终用“它减少哪种等待”解释。

- **存储/NVMe**：保存数据、Checkpoint、权重和日志；关注持续吞吐、小文件、元数据和多机并发。
- **CPU/DDR**：承担控制面、预处理、Tokenizer、调度和内存管理；关注核心、DDR、NUMA 与设备亲和性。
- **PCIe/DMA**：连接 Host、加速器、NIC 与本地存储；Pinned memory 和异步拷贝用于通信计算重叠。
- **GPU/TPU/NPU**：执行张量算子。重要的是算子能否映射、有效利用率、精度支持和集群扩展，而不是孤立峰值。
- **HBM**：容量决定能否运行，带宽决定状态能否及时送到计算单元，两者不能互相替代。
- **Scale-up 互联**：节点/超级节点内的高频低延迟通信，常承载 TP/EP。
- **Scale-out 网络**：跨节点承载 DP/PP/EP、Checkpoint 与服务流量；关注延迟、拥塞、bisection bandwidth 与故障域。

三个常见误判：只看 FLOPS 不看 HBM/算子覆盖；只看单链路速率不看集合通信全局路径；只看硬件不看编译栈、框架和观测能力。

## 08 从工作负载反推系统（7 分钟）

固定判断顺序：

```text
形状 → 容量 → 瓶颈 → 拓扑/软件 → 具体硬件
```

### 案例 A：70B 低延迟聊天

1. BF16 权重约 140GB，先确定量化或 TP；
2. 低 batch Decode 偏 HBM 带宽；
3. 跨卡后还要支付每层通信；
4. 优先减少跨卡边界、管理 KV、提高有限 batch 下权重复用；
5. 评价 TTFT、TPOT、p99，而不是只看总 tokens/s。

### 案例 B：MoE 多节点预训练

1. 总参数决定权重与训练状态容量；
2. 活跃参数决定主要 token 计算；
3. EP 产生 All-to-All；
4. Router skew 制造热点与 straggler；
5. 监控 expert load、drop rate、All-to-All 时间与 MFU。

### 案例 C：高分辨率多模态在线推理

1. 分辨率决定视觉 token；
2. 视觉编码与长 Prefill 影响 TTFT；
3. 文本 Decode 又可能回到带宽受限；
4. 图像预处理可能让 CPU 成为前端瓶颈；
5. 分别测预处理、视觉编码、Prefill、Decode，而不是只报端到端平均。

### 决策速查

| 问题 | 若为“是” | 首先检查 |
| --- | --- | --- |
| 单卡装不下？ | 必须切分/量化/卸载 | 权重、训练状态、KV、workspace |
| 大 GEMM 主导？ | 可能计算受限 | 有效 FLOPS、MFU、融合 |
| 低 batch Decode？ | 可能带宽受限 | HBM、量化、batch、KV |
| 每层集合通信？ | 对延迟敏感 | 节点内互联、消息频率 |
| MoE Router 不均？ | 会出现长尾 | expert load、capacity、All-to-All |
| 请求长度差异大？ | 静态批浪费槽位 | continuous batching、KV blocks |
| 数据供给不足？ | 加速器空转 | CPU、DDR、存储、prefetch |

## 09 代码证据与收束（10–15 分钟，可选）

不要逐行读完整程序，只寻找“数据在哪里被读写”。推荐顺序：

1. `01_training_step.py`：forward、backward、optimizer 三类状态；
2. `02_memory_budget.py`：先判断装不装得下；
3. `03_kv_cache_demo.py`：以容量换历史 K/V 重投影；
4. `04_parallelism_cost.py`：切分后必然产生通信；
5. `06_moe_routing.py`：稀疏计算把问题转移到路由与网络；
6. `07_continuous_batching.py`：优化对象从单请求变成请求集合。

最后让听众复述：模型架构决定数据流；瓶颈随阶段迁移；硬件价值在减少等待。

---

## 5. 15 分钟代码演示 Runbook

### 容量（3 分钟）

先问 70B、BF16、8 卡时每卡权重和训练状态各是多少。

```bash
python labs/02_memory_budget.py --preset 70b --context 8192 --batch 8 --tp 8 --shards 8
```

注意 TP 每卡权重与 ZeRO-3 数字不同：前者是推理权重切分，后者包含完整训练状态分片。

### 并行（3 分钟）

```bash
python labs/04_parallelism_cost.py --mode all --tokens 4096 --gpus 8
```

先按单次载荷排序，再按每步总流量/频率排序，两个答案通常不同。

### Roofline（3 分钟）

```bash
python labs/05_roofline_prefill_decode.py --decode-batch 1
python labs/05_roofline_prefill_decode.py --decode-batch 32
```

Decode 算术强度会上升。追问为何不能无限加 batch：KV 容量、排队与 SLA 会限制它。

### MoE（3 分钟）

```bash
python labs/06_moe_routing.py --skew 0
python labs/06_moe_routing.py --skew 0.7
```

比较最大/平均负载与容量溢出，把辅助 loss 和网络长尾连接起来。

### 调度（3 分钟）

```bash
python labs/07_continuous_batching.py
```

比较 makespan、slot utilization 与 p95；吞吐改善不自动等于所有请求延迟改善。

## 6. 常见追问

### GPU 利用率越高越好吗？

不一定。无用 padding、重复计算或错误 batch 也会让设备忙。应同时看有效 tokens/s、MFU、延迟与成本。

### HBM 容量够了，为什么仍 OOM？

通常遗漏了激活、workspace、通信 buffer、CUDA graph、量化元数据、碎片和生命周期峰值重叠。

### 量化为什么既影响容量又影响速度？

更少字节降低容量与带宽压力，但速度收益取决于硬件低精度单元、内核成熟度和量化/反量化开销。

### 为什么 TP 不宜随便跨节点？

TP 通信往往每层发生，同步严格，对延迟敏感。是否可行仍要用真实网络与实现测量。

### Paged Attention 会减少 KV 理论字节数吗？

通常不改变单 token KV 公式，主要改善分配粒度、碎片、共享与回收。

### Prefill/Decode 分离一定更好吗？

不一定。它提供独立调优空间，却新增 KV 传输、排队和资源池匹配。

### 如何判断性能数字是否可信？

要求同时给出模型与精度、输入输出长度、batch/并发、硬件数量和拓扑、软件版本、预热方法，以及平均值还是分位数。

## 7. 内容边界与延伸阅读

本讲义中的公式用于建立数量级直觉，不替代 profiler、硬件计数器与端到端基准。模型会因 GQA、稀疏度、并行映射、算子融合、量化和调度产生显著差异。

- [AI Infra Book](https://github.com/bojieli/ai-infra-book)：容量、Roofline、并行选择与系统约束；
- [MiniMind](https://github.com/jingyaogong/minimind)：预训练循环与 MoE Router 的可读实现；
- [nanochat](https://github.com/karpathy/nanochat)：现代训练循环、MFU、KV Cache 与推理引擎。

完整来源、官方硬件/并行文档和估算边界见 [`SOURCES.md`](SOURCES.md)。
