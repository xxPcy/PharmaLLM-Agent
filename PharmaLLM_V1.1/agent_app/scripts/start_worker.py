#!/usr/bin/env python3
"""CPU Celery worker; optionally own a resident segmentation service in this terminal."""

import argparse
import asyncio
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

from env_utils import load
from process_control import stop_process_group


def worker_command():
    return [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "pharma_agent.worker:app",
        "worker",
        "--loglevel=WARNING",
        "--concurrency=2",
        "--without-gossip",
        "--without-mingle",
    ]


def local_endpoint(url, requested_port=None):
    try:
        parsed = urlsplit(url)
        port = parsed.port or 80
    except ValueError:
        raise ValueError("SEGMENTATION_BASE_URL 的地址或端口格式无效") from None
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("合并启动需要 SEGMENTATION_BASE_URL 为 http://127.0.0.1:端口，且不带 /v1")
    if requested_port is not None and requested_port != port:
        raise ValueError("--seg-port 与 SEGMENTATION_BASE_URL 的端口不一致，请先统一配置")
    return "127.0.0.1", port


def ensure_port_free(host, port):
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex((host, port)) == 0:
            raise ValueError(
                f"分割端口 {port} 已有服务，未结束任何进程；请先停止原分割服务，或不带 --seg-gpu 仅启动 Worker"
            )


def wait_ready(process, gateway, timeout):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("分割服务启动进程已退出，请检查本终端的启动信息和私有 resident.log")
        detail = asyncio.run(gateway.probe_detail())
        if detail["status"] == "ready":
            if process.poll() is not None:
                raise RuntimeError("本次分割服务已退出，拒绝连接可能被替换的端口")
            print("分割服务已就绪，正在启动 CPU 队列 Worker。", flush=True)
            return
        if detail["message"] != previous:
            print(detail["message"], flush=True)
            previous = detail["message"]
        if detail["status"] not in {"loading", "offline"}:
            raise RuntimeError(detail["message"])
        time.sleep(0.5)
    raise RuntimeError("等待分割模型加载超时，请检查私有 resident.log；可调整 --startup-timeout")


def supervise(seg_command, seg_env, gateway, timeout):
    children, handlers = [], {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, interrupted)
        segmentation = subprocess.Popen(seg_command, env=seg_env, start_new_session=True)
        children.append(segmentation)
        print(f"本终端管理分割服务 PID={segmentation.pid}；正在加载模型。", flush=True)
        wait_ready(segmentation, gateway, timeout)
        worker_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        worker = subprocess.Popen(worker_command(), env=worker_env, start_new_session=True)
        children.append(worker)
        print(f"CPU Worker PID={worker.pid}；Ctrl+C 会关闭本次启动的 Worker 和分割服务。", flush=True)
        while True:
            for child in children:
                code = child.poll()
                if code is not None:
                    return code if code >= 0 else 128 - code
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 130
    finally:
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for child in reversed(children):
                if child.poll() is None:
                    stop_process_group(child.pid, grace_seconds=10)
                child.wait(timeout=2)
        finally:
            signal.signal(signal.SIGINT, previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--seg-gpu", type=int, help="分割使用的物理 GPU 编号；同时启动独立分割服务")
    device.add_argument("--seg-cpu", action="store_true", help="合并启动 CPU 分割服务（慢，仅排障）")
    parser.add_argument("--seg-port", type=int)
    parser.add_argument("--seg-load-mode", choices=["resident", "per-task"], default="resident")
    parser.add_argument("--seg-min-free-mib", type=int, default=0, help="启动时空闲显存门槛，不是显存预留")
    parser.add_argument("--startup-timeout", type=int)
    args = parser.parse_args()
    if args.seg_gpu is not None and args.seg_gpu < 0 or args.seg_min_free_mib < 0:
        parser.error("GPU 编号与显存门槛不能为负数")
    load()
    if args.seg_gpu is None and not args.seg_cpu:
        print("仅启动 CPU Worker；分割服务需另行启动，或使用 --seg-gpu 指定卡并合并启动。", flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.execv(sys.executable, worker_command())
        return
    from pharma_agent.segmentation import SegmentationGateway
    from pharma_agent.settings import Settings

    settings = Settings()
    try:
        host, port = local_endpoint(settings.segmentation_base_url, args.seg_port)
        if len(settings.segmentation_api_key.get_secret_value()) < 24:
            raise ValueError("分割服务密钥未配置，请先配置 SEGMENTATION_API_KEY")
        timeout = args.startup_timeout or settings.segmentation_startup_timeout_seconds
        if not 10 <= timeout <= 1800:
            raise ValueError("启动超时需为 10–1800 秒")
        ensure_port_free(host, port)
        command = [
            sys.executable,
            str(Path(__file__).with_name("start_segmentation.py")),
            "--port",
            str(port),
            "--load-mode",
            args.seg_load_mode,
            "--min-free-mib",
            str(args.seg_min_free_mib),
        ]
        command += ["--gpu", str(args.seg_gpu)] if args.seg_gpu is not None else ["--cpu"]
        env = {**os.environ, "SEGMENTATION_STARTUP_TIMEOUT_SECONDS": str(timeout)}
        return supervise(command, env, SegmentationGateway(settings), timeout)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
