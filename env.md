# 环境说明

## 硬件
- GPU: NVIDIA GeForce RTX 4060 Ti (8GB)
- 系统: Windows 10 x64

## 安装位置
- Python 3.11.9: `C:\Users\Administrator\Python311`（已加入 PATH）
- 虚拟环境: `llm-training-course\venv`（PyTorch CUDA + MiniMind 依赖）
- Git (MinGit): `C:\Users\Administrator\mingit`（已加入用户 PATH）
- MiniMind: `llm-training-course\repos\minimind`（经 gh-proxy.com 镜像克隆）

## 常用命令

```bat
:: 激活虚拟环境（CMD）
C:\Users\Administrator\.zcode\workspace\default\llm-training-course\venv\Scripts\activate.bat

:: 验证 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

:: 进入 MiniMind 目录
cd C:\Users\Administrator\.zcode\workspace\default\llm-training-course\repos\minimind
```

## pip 网络
直连 GitHub / PyTorch 官方源较慢或被阻断，已统一使用国内镜像：
- PyPI 清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`
- GitHub 克隆镜像：`https://gh-proxy.com/https://github.com/...`
- HuggingFace 建议设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`（数据集下载用）

## 8GB 显存提示
- MiniMind 小模型（26M~104M）单卡可训，batch size 过大时调小
- 训练脚本均支持 `--device cuda`
