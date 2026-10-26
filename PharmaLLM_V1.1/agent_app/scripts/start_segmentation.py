#!/usr/bin/env python3
"""Start private segmentation HTTP, keeping the selected model on its GPU by default."""

import argparse
import csv
import io
import os
import subprocess
import sys
from env_utils import load


def select_gpu(index, min_free_mib=0):
    """Resolve the nvidia-smi physical index to a stable CUDA UUID without importing torch."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=index,uuid,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        rows = list(csv.reader(io.StringIO(result.stdout)))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise ValueError
        selected, uuid, name, free, total = [value.strip() for value in rows[0]]
        if int(selected) != index or not uuid.startswith("GPU-"):
            raise ValueError
        free, total = int(free), int(total)
    except (OSError, subprocess.SubprocessError, ValueError):
        raise SystemExit(f"无法查询物理 GPU {index}，请先用 nvidia-smi 确认编号与驱动状态") from None
    print(f"分割服务使用物理 GPU {index}: {name}, {uuid}；当前空闲 {free}/{total} MiB", flush=True)
    if free < min_free_mib:
        raise SystemExit(f"GPU {index} 空闲显存 {free} MiB，低于启动门槛 {min_free_mib} MiB；未结束任何进程")
    return uuid


def main():
    parser = argparse.ArgumentParser()
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument(
        "--gpu", type=int, help="Explicit physical GPU index; choose a GPU with enough free memory"
    )
    device.add_argument("--cpu", action="store_true", help="Use CPU inference (slow)")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--load-mode", choices=["resident", "per-task"], default="resident")
    parser.add_argument("--min-free-mib", type=int, default=0)
    args = parser.parse_args()
    if args.gpu is not None and args.gpu < 0:
        parser.error("GPU 编号不能为负数")
    if not 1 <= args.port <= 65535 or args.min_free_mib < 0:
        parser.error("端口或显存门槛无效")
    load()
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        select_gpu(args.gpu, args.min_free_mib) if args.gpu is not None else ""
    )
    os.environ["SEGMENTATION_LOAD_MODE"] = args.load_mode.replace("-", "_")
    os.environ["NNUNET_DEVICE"] = "cuda" if args.gpu is not None else "cpu"
    os.environ["NNUNET_GPU_INDEX"] = str(args.gpu) if args.gpu is not None else "cpu"
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "pharma_agent.nnunet_service:create_segmentation_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--no-access-log",
            "--workers",
            "1",
        ],
    )


if __name__ == "__main__":
    main()
