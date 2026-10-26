"""Supervise only a newly launched process group; no global process-name cleanup."""

import os
import signal
import subprocess
import time


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_process_group(pgid, grace_seconds=5):
    if pgid <= 1 or pgid == os.getpgrp():
        raise ValueError("Refusing to terminate the caller's process group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not group_exists(pgid):
            return False
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False


def run_supervised(command):
    """Ctrl+C/TERM/HUP shuts down this launch's workers, including orphaned ones."""
    process = None
    previous_handlers = {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        # A dedicated group prevents cleanup from touching the shell or another vLLM service.
        process = subprocess.Popen(command, start_new_session=True)
        print(f"vLLM PID={process.pid}；Ctrl+C 会清理本次启动的进程组。", flush=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            code = 130
    finally:
        if process is not None:
            old_interrupt = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                if group_exists(process.pid):
                    print("正在关闭本次 vLLM 子进程；5 秒后强制清理未响应的 Worker。", flush=True)
                    stop_process_group(process.pid)
                process.wait(timeout=2)
            finally:
                signal.signal(signal.SIGINT, old_interrupt)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return code if code >= 0 else 128 - code
