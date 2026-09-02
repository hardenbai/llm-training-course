# GPU 环境自检 + CPU vs GPU 性能对比
# 在 VS Code 里打开本文件，点右上角 ▶ 运行
import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch
import time

print("=" * 56)
print("GPU 自检")
print("=" * 56)
print(f"PyTorch 版本        : {torch.__version__}")
print(f"CUDA 是否可用       : {torch.cuda.is_available()}")
print(f"GPU 型号            : {torch.cuda.get_device_name(0)}")
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"显存总量            : {vram:.1f} GB")
print(f"支持 bf16 混合精度  : {torch.cuda.is_bf16_supported()}  <- 周2预训练会用到")

print()
print("=" * 56)
print("实验：CPU vs GPU 矩阵乘法（深度学习的基本运算）")
print("=" * 56)
# 模拟一层神经网络的计算: 两个 2048x2048 的大矩阵相乘
n = 2048
a_cpu = torch.randn(n, n)
b_cpu = torch.randn(n, n)
a_gpu = a_cpu.cuda()
b_gpu = b_cpu.cuda()

# 预热（首次调用有初始化开销，不计入）
_ = a_gpu @ b_gpu
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    c_cpu = a_cpu @ b_cpu
t_cpu = (time.perf_counter() - t0) / 10

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()
t_gpu = (time.perf_counter() - t0) / 10

print(f"CPU 耗时 : {t_cpu*1000:8.1f} 毫秒/次")
print(f"GPU 耗时 : {t_gpu*1000:8.1f} 毫秒/次")
print(f"加速比   : {t_cpu/t_gpu:.0f} 倍")
print()
print("这就是为什么大模型训练必须用 GPU：")
print("GPU 有几千个核心同时算，CPU 只有几十个。")
print("一个矩阵乘法快几十倍，整个训练就快几十倍。")

print()
print("=" * 56)
print("显存占用观察")
print("=" * 56)
print(f"当前已用显存: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
x = torch.randn(4096, 4096, device="cuda")  # 一个 4096x4096 的 fp32 张量
print(f"分配一个 4096x4096 张量后: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
del x
torch.cuda.empty_cache()
print(f"释放后: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print()
print("你的显卡 8GB 显存 = 能同时装下多少这样的张量，")
print("这就是为什么 batch size 太大会报 CUDA out of memory")

print()
print("✅ 全部通过。周2跑 train_pretrain.py 时加 --device cuda 就会用到这块 GPU")
