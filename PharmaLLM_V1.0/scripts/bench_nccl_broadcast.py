from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NCCL broadcast latency/bandwidth.")
    parser.add_argument("--numel", type=int, default=622_329_856)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--src", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    dtype = DTYPES[args.dtype]
    element_size = torch.empty((), dtype=dtype).element_size()
    size_gib = args.numel * element_size / 1024**3

    tensor = torch.empty(args.numel, dtype=dtype, device=device)
    if rank == args.src:
        tensor.fill_(1)
    else:
        tensor.zero_()

    dist.barrier()
    if rank == 0:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            f"world_size={world_size} visible={visible} "
            f"numel={args.numel} dtype={args.dtype} size={size_gib:.2f} GiB src={args.src}",
            flush=True,
        )

    for _ in range(args.warmup):
        dist.broadcast(tensor, src=args.src)
    torch.cuda.synchronize()
    dist.barrier()

    elapsed: list[float] = []
    for _ in range(args.iters):
        if rank == args.src:
            tensor.add_(1)
        dist.barrier()
        start = time.perf_counter()
        dist.broadcast(tensor, src=args.src)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed.append(time.perf_counter() - start)

    local_times = torch.tensor(elapsed, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local_times) for _ in range(world_size)]
    dist.all_gather(gathered, local_times)

    if rank == 0:
        all_times = torch.stack(gathered).cpu()
        per_iter_max = all_times.max(dim=0).values
        for index, seconds in enumerate(per_iter_max.tolist(), start=1):
            gib_per_second = size_gib / seconds
            print(f"iter={index} max_time={seconds:.4f}s bandwidth={gib_per_second:.2f} GiB/s", flush=True)
        avg = per_iter_max.mean().item()
        print(f"avg_max_time={avg:.4f}s avg_bandwidth={size_gib / avg:.2f} GiB/s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
