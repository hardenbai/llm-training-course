# 周2 Day2：训练循环 —— 在你的 4060 Ti 上真跑一次预训练
# 对照源码: repos/minimind/trainer/train_pretrain.py
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import json
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "repos" / "minimind"))
ART = ROOT / "lessons" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

print("=" * 60)
print("实验 1：学习率调度 —— 为什么要 warmup + cosine")
print("=" * 60)
total_steps, warmup = 100, 10
def lr_at(step):
    if step < warmup: return 1e-3 * step / warmup           # 热身：从小到大
    p = (step - warmup) / (total_steps - warmup)
    return 1e-3 * 0.5 * (1 + math.cos(math.pi * p))          # 余弦：从大到小
bars = ""
for s in range(0, total_steps, 2):
    bars += "█" if s < warmup else "▓"
print(f"lr 曲线（█=warmup ▓=cosine）: {bars}")
print("""
刚初始化的模型参数是随机的，一开始大步走会直接'摔倒'(loss爆炸)，
所以先小步热身(warmup)；快结束时小步微调，避免在最优点附近来回震荡。""")

print()
print("=" * 60)
print("实验 2：梯度裁剪 —— 给梯度装限速器")
print("=" * 60)
g = torch.tensor([0.1, 5.0, -8.0, 0.3])
clipped = g * (1.0 / max(1.0, g.norm() / 1.0))   # 源码: clip_grad_norm_(params, 1.0)
print(f"裁剪前范数: {g.norm():.2f}  (梯度过大 = 训练要失控)")
print(f"裁剪后范数: {clipped.norm():.2f}  (整体按比例缩小到 1，方向不变)")
print("方向不变、幅度受限 —— 既保留'往哪走'的信息，又防'迈太大步'。")

print()
print("=" * 60)
print("实验 3：真跑预训练（小配置，真数据，你的GPU）")
print("=" * 60)
# 完整版用 dim=768/layers=8 训几小时；教学版 dim=256/layers=4 跑几十步看loss下降
print(f"设备: {device}")
cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4, vocab_size=6400,
                     num_attention_heads=8, num_key_value_heads=4)
model = MiniMindForCausalLM(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {n_params/1e6:.1f}M（完整版约 26M）")

tok = AutoTokenizer.from_pretrained(ROOT / "repos" / "minimind" / "model")
MAX_LEN = 256
sample_file = ART / "pretrain_sample.jsonl"
if not sample_file.exists():
    print("请先运行 week2_day1_pretrain_data.py 生成数据样本"); sys.exit(1)

class DS(torch.utils.data.Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                ids = tok(t, add_special_tokens=False, max_length=MAX_LEN-2, truncation=True).input_ids
                ids = [tok.bos_token_id] + ids + [tok.eos_token_id]
                if len(ids) < 32: continue
                ids = ids[:MAX_LEN] + [tok.pad_token_id]*(MAX_LEN-len(ids))
                lab = [i if i != tok.pad_token_id else -100 for i in ids]
                self.rows.append((ids, lab))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        x, y = self.rows[i]
        return torch.tensor(x), torch.tensor(y)

ds = DS(sample_file)
print(f"训练样本: {len(ds)} 条")
dl = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))   # 混合精度的梯度缩放器

STEPS = 60
warmup = 6
loss_log = []
t0 = time.perf_counter()
model.train()
for step in range(STEPS):
    x, y = next(iter(dl))
    x, y = x.to(device), y.to(device)
    lr = 1e-3 * min((step+1)/warmup, 1) * 0.5*(1+math.cos(math.pi*step/STEPS))
    for gparam in opt.param_groups: gparam["lr"] = lr
    with torch.amp.autocast(device_type=device if device=="cuda" else "cpu", enabled=(device=="cuda")):
        out = model(input_ids=x, labels=y)      # 模型内部自动算好 next-token loss (Day3!)
    loss = out.loss
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    loss_log.append(loss.item())
    if step % 10 == 0 or step == STEPS-1:
        tok_per_s = len(dl.dataset.rows[0]) * 0  # placeholder
        print(f"  step {step:3d} | lr {lr:.2e} | loss {loss.item():.4f} | PPL {math.exp(loss.item()):8.0f}")
dt = time.perf_counter() - t0
print(f"\n{STEPS} 步耗时 {dt:.1f} 秒（{dt/STEPS:.2f} 秒/步）")
print(f"loss: {loss_log[0]:.3f} -> {loss_log[-1]:.3f}   <- 你的模型正在学会'接话'！")
print("（ln(6400)=8.76 是随机水平，几十步就明显低于它）")

# 保存 checkpoint 和 loss 记录（Day3 会用）
torch.save(model.state_dict(), ART / "mini_pretrained.pth")
with open(ART / "pretrain_loss.json", "w") as f:
    json.dump(loss_log, f)
print(f"\n已保存: artifacts/mini_pretrained.pth + pretrain_loss.json（明天分析用）")

print()
print("=" * 60)
print("实验 4：混合精度 —— 为什么 bf16 训练快一倍")
print("=" * 60)
print("""
上面代码里的 autocast + GradScaler 就是混合精度：
  前向/反向用 bf16（半内存、双倍速度，精度略低但够用）
  权重更新仍用 fp32（钱要算得分毫不差）
你的 4060 Ti 支持 bf16（gpu_check.py 验证过）。完整预训练不加混合精度，
显存直接翻倍、速度减半 —— 这就是 train_pretrain.py 里都有它的原因。""")
