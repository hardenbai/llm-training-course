# 周3 Day1：SFT —— 从'会接话'到'会对话'
# 对照源码: dataset/lm_dataset.py 第 58-119 行 (SFTDataset)
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "repos" / "minimind"))
ART = ROOT / "lessons" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer

print("=" * 60)
print("第 1 步：SFT 数据和预训练数据的唯一区别 —— 结构")
print("=" * 60)
tok = AutoTokenizer.from_pretrained(ROOT / "repos" / "minimind" / "model")

# 切一小块 sft 数据
sft_cache = ART / "sft_sample.jsonl"
src = ROOT / "repos" / "minimind" / "dataset" / "sft_t2t_mini.jsonl"
with open(src, encoding="utf-8") as f, open(sft_cache, "w", encoding="utf-8") as out:
    for i, line in enumerate(f):
        if i >= 500: break
        out.write(line)
with open(sft_cache, encoding="utf-8") as f:
    sample = json.loads(f.readline())
print("一条 SFT 数据（对话格式）:")
for turn in sample["conversations"][:4]:
    print(f"  [{turn['role']:9}] {turn['content'][:40]}")
print("""
预训练数据: {"text": "一大段文章..."}          <- 学'语言'
SFT 数据:   {"conversations": [user, assistant]} <- 学'对话'""")

print("=" * 60)
print("第 2 步：拼模板 —— 对话变文本（复刻 lm_dataset.py 第 71-86 行）")
print("=" * 60)
rendered = tok.apply_chat_template(sample["conversations"][:4], tokenize=False)
print(rendered[:400])

print("=" * 60)
print("第 3 步：loss 掩码 —— SFT 的灵魂（对照源码 88-104 行）")
print("=" * 60)
# 源码逻辑：只在 <assistant\n ... eos> 之间的 token 上计算损失
ids = tok(rendered).input_ids
labels = [-100] * len(ids)
bos_seq = tok(f"{tok.bos_token}assistant\n", add_special_tokens=False).input_ids
eos_seq = tok(f"{tok.eos_token}\n", add_special_tokens=False).input_ids
i = 0
while i < len(ids):
    if ids[i:i+len(bos_seq)] == bos_seq:
        start = i + len(bos_seq)
        end = start
        while end < len(ids) and ids[end:end+len(eos_seq)] != eos_seq:
            end += 1
        for j in range(start, min(end + len(eos_seq), len(ids))):
            labels[j] = ids[j]
        i = end + len(eos_seq)
    else:
        i += 1

print("逐 token 看掩码（✓=计算loss ✗=忽略）:")
shown, count = 0, 0
for j in range(len(ids)):
    mark = "✓" if labels[j] != -100 else "✗"
    if labels[j] != -100: count += 1
    if shown < 60 and (labels[j] != -100 or j % 8 == 0):
        t = tok.decode([ids[j]])
        print(f"  {mark} {t!r}", end="")
        shown += 1
        if shown % 6 == 0: print()
print(f"\n\n有效loss token: {count}/{len(ids)}（{count/len(ids):.0%}）")
print("""
user 的问题部分标 -100 —— 模型不需要学'怎么提问'
assistant 的回答部分保留 —— 模型只学'怎么回答'
这就是 SFT 与预训练的全部区别：同一个模型、同一个交叉熵，
只是'哪里算损失'变了。""")

print("=" * 60)
print("第 4 步：真跑 SFT（在 Day2 预训练的模型上继续训）")
print("=" * 60)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
import math, time
from torch.utils.data import DataLoader

cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4, vocab_size=6400,
                     num_attention_heads=8, num_key_value_heads=4)
model = MiniMindForCausalLM(cfg).to(device)
ckpt = ART / "mini_pretrained.pth"
if ckpt.exists():
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print("已加载周2预训练的 checkpoint（SFT 必须'继续'训，不能从零开始）")
else:
    print("警告: 找不到预训练checkpoint，从零SFT（效果差，仅演示）")

MAX_LEN = 256
class SFTDS(torch.utils.data.Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                convs = json.loads(line)["conversations"]
                text = tok.apply_chat_template(convs, tokenize=False)
                ids = tok(text, truncation=True, max_length=MAX_LEN).input_ids
                if len(ids) < 32: continue
                ids = ids + [tok.pad_token_id] * (MAX_LEN - len(ids))
                lab = [-100] * len(ids)
                i = 0
                while i < len(ids):
                    if ids[i:i+len(bos_seq)] == bos_seq:
                        s2 = i + len(bos_seq); e2 = s2
                        while e2 < len(ids) and ids[e2:e2+len(eos_seq)] != eos_seq: e2 += 1
                        for j in range(s2, min(e2 + len(eos_seq), MAX_LEN)): lab[j] = ids[j]
                        i = e2 + len(eos_seq)
                    else: i += 1
                self.rows.append((ids, lab))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        x, y = self.rows[i]
        return torch.tensor(x), torch.tensor(y)

ds = SFTDS(sft_cache)
print(f"SFT 样本: {len(ds)} 条")
dl = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

model.train()
t0 = time.perf_counter()
for step in range(30):
    x, y = next(iter(dl))
    x, y = x.to(device), y.to(device)
    out = model(input_ids=x, labels=y)
    opt.zero_grad(); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 10 == 0 or step == 29:
        print(f"  step {step:3d} | sft loss {out.loss.item():.4f}")
print(f"耗时 {time.perf_counter()-t0:.1f}s")
torch.save(model.state_dict(), ART / "mini_sfted.pth")
print("已保存: artifacts/mini_sfted.pth（周4 对齐实验的基础）")

print(r"""
课后阅读（真正的训练脚本）:
  repos/minimind/trainer/train_full_sft.py  -- 和本课逻辑一致，多了日志/保存
完整 SFT 命令:
  cd repos\minimind\trainer
  ..\..\..\venv\Scripts\python.exe train_full_sft.py --device cuda --out_dir ../out_sft""")
