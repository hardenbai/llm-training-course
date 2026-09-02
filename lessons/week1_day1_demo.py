# 周1 Day1 实验：观察 tokenizer —— 文本如何变成数字
# 运行方式（在 llm-training-course 目录下）:
#   venv\Scripts\python.exe lessons\week1_day1_demo.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

REPO = "repos/minimind/model"

tok = AutoTokenizer.from_pretrained(REPO)

print("=" * 60)
print("实验 1：一句话被切成了什么？")
print("=" * 60)
text = "大模型训练其实很简单。"
ids = tok.encode(text)
tokens = tok.convert_ids_to_tokens(ids)
print(f"原句     : {text}")
print(f"token 数 : {len(ids)}")
for i, (t, idx) in enumerate(zip(tokens, ids)):
    print(f"  [{i}] id={idx:5d}  token={t!r}")
print(f"解码还原 : {tok.decode(ids)}  <- 和原句一致，说明编码无损")

print()
print("=" * 60)
print("实验 2：不同类型文本的切分效率（压缩率）")
print("=" * 60)
# 压缩率 = 字符数 / token数，越高说明一个 token 承载的信息越多
samples = {
    "常见中文": "我想学习大模型",
    "生僻内容": "饕餮耄耋觊觎龃龉囹圄魑魅魍魉耄耋之年，犹自孜孜矻矻。",
    "英文": "Large language models predict the next token in a sequence.",
    "代码": "for i in range(10): print(i ** 2)",
    "数字": "圆周率是 3.14159265358979，手机号 13812345678。",
}
for name, t in samples.items():
    ids_ = tok.encode(t)
    ratio = len(t) / len(ids_)
    print(f"{name:6} | {len(t):3} 字符 -> {len(ids_):3} tokens | 压缩率 {ratio:.2f}")

print()
print("思考：为什么生僻字的压缩率低？（提示：BPE 是按'出现频率'合并字节的）")

print()
print("=" * 60)
print("实验 3：语言模型的训练目标 —— 一句话的多种接法")
print("=" * 60)
# 下一个 token 预测：给定前缀，模型要对'下一个 token'输出一个概率分布
prefix = "今天天气真"
candidates = ["好", "差", "冷", "热", "不错"]
print(f"前缀: {prefix!r}")
print("语言模型要做的事：为下一个位置输出概率分布，例如")
for c, p in zip(candidates, [0.62, 0.05, 0.11, 0.08, 0.14]):
    print(f"  P({c!r} | 前缀) = {p}")
print()
print("预训练 = 在海量文本上，让'真实出现的下一个token'概率尽可能高")
print("所有阶段(pretrain/SFT/DPO)训练的都是同一个模型，只是数据格式不同")

print()
print("=" * 60)
print("实验 4：chat 模板 —— 对话如何被拼成训练文本")
print("=" * 60)
messages = [
    {"role": "system", "content": "你是迷你模型"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你？"},
]
rendered = tok.apply_chat_template(messages, tokenize=False)
print(rendered)
print("（这些 <|im_start|> <|im_end|> 就是 tokenizer 里的特殊 token，")
print("  周3学SFT时会看到：损失只对 assistant 部分计算）")
