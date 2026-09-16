"""把 LLM、VLM 与 Diffusion 还原成 token/latent 形状与重复执行次数。"""

from __future__ import annotations

import argparse


def tflops(params_b: float, positions: int, passes: int = 1) -> float:
    return 2 * params_b * 1e9 * positions * passes / 1e12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=float, default=7, help="活跃主干参数，单位 B")
    parser.add_argument("--text-tokens", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--vision-patch", type=int, default=14)
    parser.add_argument("--diffusion-size", type=int, default=1024)
    parser.add_argument("--vae-scale", type=int, default=8)
    parser.add_argument("--latent-patch", type=int, default=2)
    parser.add_argument("--denoise-steps", type=int, default=30)
    args = parser.parse_args()

    values = vars(args).values()
    if any(value <= 0 for value in values):
        parser.error("所有参数必须大于 0")

    vision_tokens = (args.image_size // args.vision_patch) ** 2
    vlm_positions = args.text_tokens + vision_tokens
    latent_side = args.diffusion_size // args.vae_scale // args.latent_patch
    latent_tokens = latent_side ** 2
    if vision_tokens < 1 or latent_tokens < 1:
        parser.error("patch/scale 不能大于对应图像或 latent 尺寸")

    print("工作负载        位置数/步   主干执行次数   线性层 FLOPs 量级   主要新增压力")
    print(f"Text LLM       {args.text_tokens:9,d} {1:12d} {tflops(args.params, args.text_tokens):14.1f} T   KV Cache / 自回归")
    print(f"VLM Prefill    {vlm_positions:9,d} {1:12d} {tflops(args.params, vlm_positions):14.1f} T   图像 token 拉长序列")
    print(f"Diffusion      {latent_tokens:9,d} {args.denoise_steps:12d} {tflops(args.params, latent_tokens, args.denoise_steps):14.1f} T   latent 状态 × 去噪迭代")
    print(f"\nVLM 图像 token：({args.image_size}/{args.vision_patch})² = {vision_tokens:,}")
    print(f"Diffusion latent token：({args.diffusion_size}/{args.vae_scale}/{args.latent_patch})² = {latent_tokens:,}")
    print("\n现场实验：把图像边长翻倍；二维 token 数约变 4 倍，而全注意力 score 数可能约变 16 倍。")
    print("边界：FLOPs 只用 2P×positions 展示主干线性层数量级，未计视觉编码器、Attention 二次项、VAE 和 CFG。")


if __name__ == "__main__":
    main()
