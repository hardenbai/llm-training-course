# 配套代码实验

这些实验刻意保持短小，适合在汇报中逐段运行。它们用于验证原理，而不是替代 MiniMind、nanochat 或生产训练框架。

```bash
# 无第三方依赖
python labs/02_memory_budget.py --preset 70b --context 8192 --batch 8
python labs/04_parallelism_cost.py --params 70 --gpus 8 --mode tp

# 需要 PyTorch
python labs/01_training_step.py
python labs/03_kv_cache_demo.py
```

建议讲法：先在网页里预测结果，再运行脚本，用输出检验判断。
