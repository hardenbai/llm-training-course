# 周0 补课：20 行代码看懂"训练"的本质
# 在 VS Code 里打开，点 ▶ 运行。不需要任何基础。
import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
torch.manual_seed(42)

print("=" * 56)
print("第 1 步：数据 —— 一堆 (x, y) 数字对")
print("=" * 56)
# 假设世界上有个规律: y = 3x + 8，但我们假装不知道
# 只能观察到 100 个样本点
xs = torch.rand(100, 1) * 10
ys = 3 * xs + 8
print("前 5 个样本:")
for i in range(5):
    print(f"  输入 x={xs[i].item():.2f}  ->  观察到 y={ys[i].item():.2f}")

print()
print("=" * 56)
print("第 2 步：模型 —— 两个可调的数字 w 和 b")
print("=" * 56)
# 我们的"模型"就是 y_pred = w*x + b，w、b 随机初始化
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(f"初始 w = {w.item():.3f}（真值是 3）")
print(f"初始 b = {b.item():.3f}（真值是 8）")
print("「模型」= 一个带可调数字的公式。就这么朴素。")

print()
print("=" * 56)
print("第 3 步：训练 —— 微调 w、b，让误差越来越小")
print("=" * 56)
optimizer = torch.optim.SGD([w, b], lr=0.01)  # 负责"微调"的工具
for step in range(500):
    y_pred = w * xs + b            # 前向：用当前 w、b 算预测
    loss = ((y_pred - ys) ** 2).mean()  # 误差：预测和真实差多远
    optimizer.zero_grad()          # 清空上一轮的梯度
    loss.backward()                # 反向：算出"w、b 往哪调能减小误差"
    optimizer.step()               # 更新：真的去调 w、b
    if step % 100 == 0 or step == 499:
        print(f"  第 {step:3d} 轮 | 误差 loss = {loss.item():.4f} | w = {w.item():.3f}, b = {b.item():.3f}")

print()
print("看：loss 一路下降，w 逼近 3，b 逼近 8 —— 模型自己「学」出了规律！")
print("没有任何人告诉它 3 和 8，它只靠「调小误差」就找到了。")

print()
print("=" * 56)
print("第 4 步：GPT 和这个的区别在哪？")
print("=" * 56)
print("""
  这个模型          GPT
  ---------------  -------------------------------
  公式 y = w*x+b   Transformer 网络（几十亿个 w、b）
  输入 x           前文 token 序列
  预测 y           下一个 token 的概率分布
  训练循环          完全相同的四行：
                   前向 -> 算loss -> backward -> step

  你刚才看懂的「前向、loss、backward、更新」，
  就是 train_pretrain.py 训练 26M 参数模型时循环里一模一样的四步。
  区别只是参数多了几百万倍、公式换成了 Transformer。
""")
print("结论：你已经理解了「训练」的原理。剩下的四周只是在换数据和换公式。")
