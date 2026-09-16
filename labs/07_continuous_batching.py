"""离散模拟静态批处理与 Continuous Batching 的 Decode 槽位利用率。"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival: int
    output_tokens: int


def percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * ratio))]


def static_batches(requests: list[Request], slots: int) -> tuple[int, int, list[int]]:
    now = 0
    useful = 0
    completions = []
    for start in range(0, len(requests), slots):
        batch = requests[start:start + slots]
        now = max(now, max(item.arrival for item in batch))
        duration = max(item.output_tokens for item in batch)
        now += duration
        useful += sum(item.output_tokens for item in batch)
        completions.extend([now - item.arrival for item in batch])
    return now, useful, completions


def continuous_batches(requests: list[Request], slots: int) -> tuple[int, int, list[int]]:
    now = 0
    waiting = list(requests)
    active: list[list[int]] = []
    completion_by_id = {}
    useful = 0
    while waiting or active:
        while waiting and waiting[0].arrival <= now and len(active) < slots:
            item = waiting.pop(0)
            active.append([item.request_id, item.arrival, item.output_tokens])
        if not active:
            now = waiting[0].arrival
            continue
        now += 1
        useful += len(active)
        next_active = []
        for request_id, arrival, remaining in active:
            remaining -= 1
            if remaining == 0:
                completion_by_id[request_id] = now - arrival
            else:
                next_active.append([request_id, arrival, remaining])
        active = next_active
    completions = [completion_by_id[item.request_id] for item in requests]
    return now, useful, completions


def report(name: str, makespan: int, useful: int, completions: list[int], slots: int) -> None:
    utilization = useful / (makespan * slots)
    print(
        f"{name:12s} makespan={makespan:4d} steps  slot-util={utilization:6.1%}  "
        f"p50={percentile(completions, .5):3d}  p95={percentile(completions, .95):3d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--max-arrival", type=int, default=12)
    parser.add_argument("--min-output", type=int, default=4)
    parser.add_argument("--max-output", type=int, default=48)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if args.max_arrival < 0 or min(args.requests, args.slots, args.min_output, args.max_output) < 1 or args.min_output > args.max_output:
        parser.error("请求、槽位和输出长度参数无效")

    rng = random.Random(args.seed)
    requests = [
        Request(index, rng.randint(0, args.max_arrival), rng.randint(args.min_output, args.max_output))
        for index in range(args.requests)
    ]
    requests.sort(key=lambda item: (item.arrival, item.request_id))
    static = static_batches(requests, args.slots)
    continuous = continuous_batches(requests, args.slots)

    print("请求（arrival → output tokens）：")
    print("  " + "  ".join(f"R{r.request_id}:{r.arrival}→{r.output_tokens}" for r in requests))
    print("\n策略对比（每个 step 每个活跃槽位生成 1 token）：")
    report("Static", *static, args.slots)
    report("Continuous", *continuous, args.slots)
    print(f"\n完成时间缩短：{static[0] / continuous[0]:.2f}x")
    print("解释：静态批要等最长请求，短请求完成后的槽位闲置；连续批在每个 token 边界补入新请求。")
    print("边界：模拟器不含 Prefill、KV 容量、调度开销、抢占和每步 batch 大小对时延的影响。")


if __name__ == "__main__":
    main()
