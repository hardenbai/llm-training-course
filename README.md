# AI Infra：从模型到机器

面向具备 AI 产业背景的研究者，用一套可离线运行的交互式网页解释：模型训练与推理的全生命周期中，数据、计算和通信如何流动，以及 CPU、GPU/NPU、显存、互联、网络和存储为什么会成为不同阶段的关键资源。

## 直接开始

网页没有构建步骤，也不依赖在线 CDN。

```bash
python -m http.server 8000
```

浏览器打开 <http://localhost:8000>。也可以直接双击 `index.html`。

演示支持：

- 鼠标滚轮、触控板或触摸手势上下浏览当前章节
- `↑` / `↓` 或 `PageUp` / `PageDown` 分段滚动，`Home` / `End` 回到章节首尾
- `←` / `→` 切换章节；换章后自动回到顶部
- `Space` 暂停或继续动画
- 顶部计时器用于控制 60–90 分钟汇报节奏
- 所有关键图示均支持现场切换模型、并行策略或推理阶段
- 系统开启“减少动态效果”时自动关闭非必要动画

## 推荐讲解节奏

| 模块 | 建议时间 | 观众最终应理解 |
| --- | ---: | --- |
| 01 全生命周期 | 6 分钟 | AI 系统首先是一条数据流 |
| 02 模型变成工作负载 | 9 分钟 | 架构决定 FLOPs、访存和通信模式 |
| 03 一步训练 | 11 分钟 | 前向、反向和优化器各自占用什么 |
| 04 多卡训练 | 12 分钟 | DP、TP、PP、EP 分别在切什么、传什么 |
| 05 推理的两种世界 | 13 分钟 | Prefill 偏计算，Decode 偏带宽 |
| 06 服务系统 | 8 分钟 | 调度、批处理和 KV 管理如何决定吞吐与延迟 |
| 07 硬件地图 | 9 分钟 | 每一类硬件在数据路径中的职责 |
| 08 工作负载决策 | 7 分钟 | 从模型和业务目标反推瓶颈 |
| 09 代码证据 | 10–15 分钟 | 概念如何落到 PyTorch 与真实项目 |

核心路线约 75 分钟；代码与案例讨论可扩展到 90 分钟。

## 教学代码

`labs/` 提供与网页一一对应的短实验：

- `01_training_step.py`：最小可读的 Transformer 训练步骤
- `02_memory_budget.py`：训练状态与 KV Cache 的显存估算
- `03_kv_cache_demo.py`：对比无缓存与增量解码
- `04_parallelism_cost.py`：估算 DP/TP 的单步通信量

其中 02、04 只依赖 Python 标准库；01、03 需要 PyTorch。

## 内容依据

- [bojieli/ai-infra-book](https://github.com/bojieli/ai-infra-book)：硬件约束、数据搬移与系统设计
- [jingyaogong/minimind](https://github.com/jingyaogong/minimind)：消费级硬件上的完整 LLM 训练链路
- [karpathy/nanochat](https://github.com/karpathy/nanochat)：现代化 tokenizer、预训练、后训练、评测和推理实验框架

完整引用与许可说明见 [SOURCES.md](SOURCES.md)。网页中的性能和容量结果是用于解释数量级的可调教学估算，不是任何具体硬件的基准测试。

## 历史版本

原来的四周 MiniMind 学习课程完整保存在 [`archive/course-v1/`](archive/course-v1/README.md)，没有从 Git 历史中删除。
