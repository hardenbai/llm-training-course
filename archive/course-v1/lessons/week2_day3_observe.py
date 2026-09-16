# 周2 Day3：观察训练 —— loss曲线怎么读 + 用checkpoint生成文本
# 前置：先跑 week2_day2_pretrain_loop.py（会生成 artifacts/mini_pretrained.pth）
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import json
import math
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "repos" / "minimind"))
ART = ROOT / "lessons" / "artifacts"
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

ckpt = ART / "mini_pretrained.pth"
loss_file = ART / "pretrain_loss.json"
if not ckpt.exists():
    print("找不到 artifacts/mini_pretrained.pth"); print("请先运行 week2_day2_pretrain_loop.py"); sys.exit(1)

print("=" * 60)
print("第 1 部分：读 loss 曲线")
print("=" * 60)
loss_log = json.loads(loss_file.read_text())
print(f"共 {len(loss_log)} 步")
lo, hi = min(loss_log), max(loss_log)
for i, l in enumerate(loss_log):
    if i % 2 == 0:
        width = int((l - lo) / max(hi - lo, 1e-9) * 44)
        print(f"  step {i:3d} | {'█'*width}{l:8.3f}")
print("""
读图三要素：
1. 前几步快速下降 —— 模型在学'最常见的字词组合'（最容易的规律）
2. 中后期变缓 —— 容易的学完了，开始学语法/语义等难规律
3. 若 loss 突然飙升 —— lr 太大或数据有脏东西，训练崩了要回滚checkpoint
完整训练(几小时)的曲线长这样: 陡降 -> 缓降 -> 长长的平台期""")

print()
print("=" * 60)
print("第 2 部分：加载 checkpoint，亲手生成文本")
print("=" * 60)
tok = AutoTokenizer.from_pretrained(ROOT / "repos" / "minimind" / "model")
cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4, vocab_size=6400,
                     num_attention_heads=8, num_key_value_heads=4)
model = MiniMindForCausalLM(cfg).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print("checkpoint 加载成功 —— 这就是'保存/恢复'模型，训练中断续跑全靠它\n")

prompt_text = "人工智能是"
ids = tok(prompt_text).input_ids
x = torch.tensor([ids]).to(device)
print(f"提示词: {prompt_text!r}  ->  tokens {ids}")
out = model.generate(inputs=x, attention_mask=torch.ones_like(x),
                     max_new_tokens=30, do_sample=True, temperature=0.8, top_p=0.9)
gen = tok.decode(out[0][len(ids):], skip_special_tokens=True)
print(f"模型续写: {gen!r}")
print("""
期望值管理：这个教学模型只训了 60 步(几十万token)，
输出还是'半乱码'是正常的 —— 它只来得及学会最高频的字组合。
官方完整训练(1B tokens)后才能输出通顺中文。
想看'真学会'的样子：跑完整预训练(命令在课程README)后回到本脚本再看。""")

print()
print("=" * 60)
print("第 3 部分：怎么发起完整预训练（选做，2小时+）")
print("=" * 60)
print(r"""
在 CMD 里:
  cd repos\minimind\trainer
  ..\..\..\venv\Scripts\python.exe train_pretrain.py --device cuda --epochs 1 --batch_size 32
训练产物在 repos\minimind\out\ 下。
中断不怕：脚本每过一段自动存 checkpoint，重跑会自动续训。
跑完把 loss 曲线截图存档 —— 这是你的第一个'完整炼丹记录'。""")

print()
print("=" * 60)
print("第 4 部分：作业")
print("=" * 60)
print("1. 把 Day2 脚本里 STEPS 改成 150、lr 改成 5e-3 再跑：观察 loss 是不是先崩后稳（lr过大）")
print("2. 把 temperature 改成 0.1 和 2.0 各生成一次，感受'保守 vs 抽风'")
print("3. 思考：为什么训练用 PPL=exp(loss) 汇报，而不是直接用 loss？")
