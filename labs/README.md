# 配套代码实验

这些实验用于把网页中的图变成可以验证的数字。建议遵循同一授课节奏：**先让观众预测 → 再运行 → 最后改变一个变量解释结果**。所有输出都是教学估算或微实验，不替代生产框架基准测试。

## 最短演示路线（约 15 分钟）

```bash
# 1. 容量先于性能：70B 为什么必须切分
python labs/02_memory_budget.py --preset 70b --context 8192 --batch 8 --tp 8

# 2. 同样是多卡，通信对象和频率完全不同
python labs/04_parallelism_cost.py --mode all --tokens 4096 --gpus 8

# 3. Prefill 与 Decode 的 Roofline 位置不同
python labs/05_roofline_prefill_decode.py --decode-batch 1
python labs/05_roofline_prefill_decode.py --decode-batch 32

# 4. MoE 热点专家如何制造长尾
python labs/06_moe_routing.py --skew 0
python labs/06_moe_routing.py --skew 0.7

# 5. 调度器如何回收短请求留下的空槽
python labs/07_continuous_batching.py
```

上述脚本只依赖 Python 标准库。

## 完整实验地图

| 文件 | 对应章节 | 核心变量 | 观众应观察的结果 | 依赖 |
| --- | --- | --- | --- | --- |
| `01_training_step.py` | 一步训练 | batch、sequence、hidden | 参数、梯度、Adam 状态与可见激活分别增长 | PyTorch |
| `02_memory_budget.py` | 训练/推理容量 | 精度、TP、ZeRO、上下文、并发 | 容量是否装得下是第一道硬约束 | 标准库 |
| `03_kv_cache_demo.py` | Prefill/Decode | prompt、生成长度 | 缓存结果与全量重算一致，但以容量换重复计算 | PyTorch 2.x |
| `04_parallelism_cost.py` | DP/TP/PP/EP | 通信组、tokens、层数、带宽 | 不只看字节数，还要看每步发生频率 | 标准库 |
| `05_roofline_prefill_decode.py` | 推理瓶颈 | batch、峰值 FLOPS、HBM 带宽 | 低 batch Decode 位于带宽侧 | 标准库 |
| `06_moe_routing.py` | MoE | skew、top-k、capacity factor | 平均负载正常不代表没有热点与丢弃 | 标准库 |
| `07_continuous_batching.py` | 在线服务 | 请求长度、到达时间、槽位 | Continuous Batching 提高槽位利用率 | 标准库 |
| `08_workload_shapes.py` | VLM/Diffusion | 分辨率、patch、去噪步数 | 二维分辨率和重复迭代改变工作量形状 | 标准库 |

## 需要 PyTorch 的两个实验

```bash
python labs/01_training_step.py --batch 8 --seq 32 --width 64
python labs/03_kv_cache_demo.py --prompt 128 --generate 32
```

`01` 展示真实 autograd 和 AdamW 状态；`03` 会逐 token 对照 KV Cache 输出与完整因果注意力输出，并报告最大误差。若现场环境没有 PyTorch，可以只展示代码并运行其余六个标准库实验。

## 建议的现场提问

1. 把 Decode batch 从 1 增加到 32，HBM 读取为什么没有按 32 倍增长？
2. DP 单次流量很大，TP 单次流量很小，为什么 TP 仍更依赖节点内互联？
3. MoE 平均每个专家的 token 数不变，为什么少数热点专家仍会拖慢整层？
4. Continuous Batching 提高吞吐后，为什么还必须同时约束 KV Cache 容量和 SLA？
5. 图像边长翻倍时，VLM 图像 token 数和注意力 score 数分别如何增长？

详细讲述顺序、公式推导和预期输出见根目录的 [`LECTURE_NOTES.md`](../LECTURE_NOTES.md)。
