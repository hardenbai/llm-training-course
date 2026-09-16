# 课程实验脚本索引

所有脚本在 VS Code 打开后点 ▶ 运行（或 `venv\Scripts\python.exe lessons\xxx.py`）。
按顺序学习。`artifacts/` 是脚本生成的模型和数据缓存（已 gitignore，可随时重新生成）。

## 周 0 · 补课（原理恐惧症特效药）
| 脚本 | 内容 | 产出 |
|---|---|---|
| `week0_principles.py` | 20行代码看懂训练本质：前向→loss→backward→更新 | 理解 GPT 训练循环=同一套四行 |

## 周 1 · 模型结构（配合精读 model_minimind.py）
| 脚本 | 内容 |
|---|---|
| `week1_day1_demo.py` | tokenizer 切分观察 / 压缩率 / chat 模板 |
| `week1_day2_attention.py` | QKV / 因果掩码 / RoPE / KV-Cache（含5倍加速实测） |
| `week1_day3_loss.py` | 错位对齐 / 交叉熵手算 / -100掩码 / 困惑度 |
| `week1_day4_architecture.py` | RMSNorm / SwiGLU / 残差(2500倍梯度差) / MoE路由 / 采样 |

## 周 2 · 预训练（配合精读 train_pretrain.py）
| 脚本 | 内容 | 产出 |
|---|---|---|
| `week2_day1_pretrain_data.py` | 数据管线：清洗/加特殊token/padding/-100 | artifacts/pretrain_sample.jsonl |
| `week2_day2_pretrain_loop.py` | lr调度/梯度裁剪/**真跑60步GPU预训练** | artifacts/mini_pretrained.pth |
| `week2_day3_observe.py` | 读loss曲线 / 加载checkpoint生成文本 | - |

## 周 3 · SFT 与 LoRA
| 脚本 | 内容 | 产出 |
|---|---|---|
| `week3_day1_sft.py` | chat模板 / **loss掩码逐token可视化** / 真跑SFT | artifacts/mini_sfted.pth |
| `week3_day2_lora.py` | W'=W+BA / B=0初始化 / 训练与合并 / 1.6%参数 | - |

## 周 4 · 对齐与毕业
| 脚本 | 内容 |
|---|---|
| `week4_day1_dpo.py` | DPO损失从零实现 / beta的作用 / vs PPO |
| `week4_day2_capstone.py` | **毕业项目：pretrain→SFT→DPO 全流水线一次跑通** |

## 工具
| 脚本 | 内容 |
|---|---|
| `gpu_check.py` | GPU 自检 + CPU/GPU 12倍加速实测 + 显存观察 |
