from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist


def log(message: str) -> None:
    local_rank = os.environ.get("LOCAL_RANK", "?")
    rank = dist.get_rank() if dist.is_initialized() else "?"
    print(f"[rank={rank} local_rank={local_rank}] {message}", flush=True)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    log("start")
    torch.cuda.set_device(local_rank)
    log("set_device done")
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    log("init_process_group done")

    tensor = torch.empty(1, dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    log("empty tensor done")
    tensor.fill_(1 if dist.get_rank() == 0 else 0)
    log("fill launched")
    torch.cuda.synchronize()
    log("fill synchronize done")

    dist.barrier()
    log("barrier done")
    dist.broadcast(tensor, src=0)
    log("broadcast launched")
    torch.cuda.synchronize()
    log(f"broadcast synchronize done value={tensor.item()}")

    time.sleep(0.2)
    dist.destroy_process_group()
    log("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"probe failed: {exc}", file=sys.stderr, flush=True)
        raise
