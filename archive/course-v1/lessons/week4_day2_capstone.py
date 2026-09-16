# 周4 Day2：毕业项目 —— 三阶段流水线一次跑通
# pretrain -> SFT -> DPO，在你的 GPU 上用同一个玩具模型走完全程
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "repos" / "minimind"))
ART = ROOT / "lessons" / "artifacts"
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

print("""
╔══════════════════════════════════════════════════════╗
║  毕业项目：一条命令走完大模型训练的完整生命周期        ║
║                                                      ║
║  [阶段1] 预训练  : 大量文本,  学'语言'               ║
║  [阶段2] SFT    : 对话数据,  学'回答', 只算assistant ║
║  [阶段3] DPO    : 偏好数据,  学'回答得更好'          ║
║                                                      ║
║  三个阶段 = 同一个模型 + 同一套反向传播               ║
║  区别只有: 数据长什么样 + 损失怎么算                 ║
╚══════════════════════════════════════════════════════╝""")

tok = AutoTokenizer.from_pretrained(ROOT / "repos" / "minimind" / "model")
cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4, vocab_size=6400,
                     num_attention_heads=8, num_key_value_heads=4)
model = MiniMindForCausalLM(cfg).to(device)
print(f"模型: {sum(p.numel() for p in model.parameters())/1e6:.1f}M 参数 | 设备: {device}")

bos_seq = tok(f"{tok.bos_token}assistant\n", add_special_tokens=False).input_ids
eos_seq = tok(f"{tok.eos_token}\n", add_special_tokens=False).input_ids
MAX_LEN = 192

def masked_labels(ids):
    lab = [-100] * len(ids); i = 0
    while i < len(ids):
        if ids[i:i+len(bos_seq)] == bos_seq:
            s2 = i + len(bos_seq); e2 = s2
            while e2 < len(ids) and ids[e2:e2+len(eos_seq)] != eos_seq: e2 += 1
            for j in range(s2, min(e2 + len(eos_seq), len(ids))): lab[j] = ids[j]
            i = e2 + len(eos_seq)
        else: i += 1
    return lab

def seq_logprob(model, ids):
    """整段序列的逐token log概率和（DPO 用）"""
    x = torch.tensor([ids[:-1]]).to(device)
    y = torch.tensor(ids[1:]).to(device)
    logits = model(input_ids=x).logits[0]
    return F.cross_entropy(logits, y, reduction="sum")

# ========== 阶段 1：预训练 ==========
print("\n" + "=" * 56)
print("[阶段1] 预训练 —— 学语言（数据: 通用文本）")
print("=" * 56)
pt_file = ART / "pretrain_sample.jsonl"
if not pt_file.exists():
    print("缺 artifacts/pretrain_sample.jsonl，请先跑 week2_day1"); sys.exit(1)
rows = []
with open(pt_file, encoding="utf-8") as f:
    for line in f:
        ids = tok(json.loads(line)["text"], add_special_tokens=False,
                  max_length=MAX_LEN-2, truncation=True).input_ids
        ids = [tok.bos_token_id] + ids + [tok.eos_token_id]
        if len(ids) < 32: continue
        ids = ids[:MAX_LEN] + [tok.pad_token_id]*(MAX_LEN-len(ids))
        rows.append((ids, [i if i != tok.pad_token_id else -100 for i in ids]))

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
for step in range(40):
    idx = torch.randint(0, len(rows), (8,))
    x = torch.tensor([rows[i][0] for i in idx]).to(device)
    y = torch.tensor([rows[i][1] for i in idx]).to(device)
    loss = model(input_ids=x, labels=y).loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 10 == 0 or step == 39:
        print(f"  step {step:3d} | pretrain loss {loss.item():.3f} (PPL {math.exp(loss.item()):.0f})")

# ========== 阶段 2：SFT ==========
print("\n" + "=" * 56)
print("[阶段2] SFT —— 学对话（数据: 问答对, 损失只算 assistant）")
print("=" * 56)
sft_file = ART / "sft_sample.jsonl"
sft_rows = []
with open(sft_file, encoding="utf-8") as f:
    for line in f:
        convs = json.loads(line)["conversations"]
        text = tok.apply_chat_template(convs, tokenize=False)
        ids = tok(text, truncation=True, max_length=MAX_LEN).input_ids
        if len(ids) < 32: continue
        ids = ids + [tok.pad_token_id]*(MAX_LEN-len(ids))
        sft_rows.append((ids, masked_labels(ids)))

opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
for step in range(25):
    idx = torch.randint(0, len(sft_rows), (8,))
    x = torch.tensor([sft_rows[i][0] for i in idx]).to(device)
    y = torch.tensor([sft_rows[i][1] for i in idx]).to(device)
    loss = model(input_ids=x, labels=y).loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 8 == 0 or step == 24:
        print(f"  step {step:3d} | sft loss {loss.item():.3f}")

# ========== 阶段 3：DPO ==========
print("\n" + "=" * 56)
print("[阶段3] DPO —— 学偏好（数据: 同一问题的好/坏回答对）")
print("=" * 56)
# 用同一批SFT数据构造玩具偏好对：原始回答=chosen，
# 把回答里的token随机打乱=rejected（乱序回答显然更差）
import random
random.seed(0)
pref = []
with open(sft_file, encoding="utf-8") as f:
    for line in f:
        convs = json.loads(line)["conversations"]
        good = tok.apply_chat_template(convs, tokenize=False)
        bad_conv = [dict(c) for c in convs]
        if bad_conv and bad_conv[-1]["role"] == "assistant" and len(bad_conv[-1]["content"]) > 20:
            words = list(bad_conv[-1]["content"]); random.shuffle(words)
            bad_conv[-1]["content"] = "".join(words)
            bad = tok.apply_chat_template(bad_conv, tokenize=False)
            gi = tok(good, truncation=True, max_length=MAX_LEN).input_ids
            bi = tok(bad, truncation=True, max_length=MAX_LEN).input_ids
            if gi != bi and len(gi) >= 32 and len(bi) >= 32:
                pref.append((gi, bi))
        if len(pref) >= 200: break

ref = MiniMindForCausalLM(cfg).to(device)
ref.load_state_dict(model.state_dict())     # 参考模型 = SFT 刚结束的自己
for p in ref.parameters(): p.requires_grad_(False)
ref.eval()

BETA = 0.1
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
for step in range(15):
    gi, bi = pref[step % len(pref)]
    with torch.no_grad():
        ref_c = seq_logprob(ref, gi); ref_r = seq_logprob(ref, bi)
    pol_c = seq_logprob(model, gi); pol_r = seq_logprob(model, bi)
    margin = (pol_c - ref_c) - (pol_r - ref_r)
    loss = -F.logsigmoid(BETA * margin / 32)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 3 == 0 or step == 14:
        print(f"  step {step:2d} | margin {margin.item()/len(gi):+.5f} | dpo loss {loss.item():.4f}")

print("  margin > 0 且增大 = 模型在把概率质量从'乱序回答'搬向'正常回答'")

# ========== 对比三个阶段 ==========
print("\n" + "=" * 56)
print("[对比] 同一个提示，感受三个阶段")
print("=" * 56)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "你好"}], tokenize=False, add_generation_prompt=True)
pids = tok(prompt).input_ids
x = torch.tensor([pids]).to(device)
model.eval()
out = model.generate(inputs=x, attention_mask=torch.ones_like(x),
                     max_new_tokens=20, do_sample=True, temperature=0.8, top_p=0.9)
print(f"最终模型回答: {tok.decode(out[0][len(pids):], skip_special_tokens=True)!r}")
print("(玩具规模+极短训练，输出质量不重要，流程跑通最重要)")

torch.save(model.state_dict(), ART / "mini_final.pth")
print("\n已保存: artifacts/mini_final.pth —— 完整流水线的最终模型")

print("""
╔══════════════════════════════════════════════════════╗
║  毕业了。你已经亲手完成:                            ║
║  ✓ 训练tokenizer   ✓ 理解注意力/RoPE/KV-Cache      ║
║  ✓ 预训练(真实数据) ✓ SFT(带loss掩码)  ✓ LoRA      ║
║  ✓ DPO偏好对齐     ✓ 完整三阶段流水线              ║
║                                                      ║
║  下一步(按顺序):                                    ║
║  1. 跑完整版: train_pretrain.py(2h) -> train_full_sft.py -> train_dpo.py ║
║  2. 理论补强: Stanford CS336 (youtube/b站搜)         ║
║  3. 看真实实验室: github.com/marin-community/marin   ║
║     从 docs/ 的 tiny model 教程开始，重点读失败实验复盘 ║
╚══════════════════════════════════════════════════════╝""")
