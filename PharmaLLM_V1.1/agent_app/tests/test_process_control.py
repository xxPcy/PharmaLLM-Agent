"""CPU-only checks using isolated temporary processes, never vLLM or CUDA."""

import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys

import pytest

control = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/process_control.py"))


def test_cleanup_refuses_own_process_group():
    with pytest.raises(ValueError):
        control["stop_process_group"](os.getpgrp(), grace_seconds=0)


def test_cleanup_escalates_only_the_launched_group():
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready'); time.sleep(60)",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert os.getpgid(child.pid) == child.pid
        assert control["stop_process_group"](child.pid, grace_seconds=0.1)
        assert child.wait(timeout=3) == -signal.SIGKILL
        assert control["group_exists"](os.getpgrp())  # The test runner's group was untouched.
        assert not control["group_exists"](child.pid)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
        child.stdout.close()


def test_supervisor_preserves_normal_exit_status():
    assert control["run_supervised"]([sys.executable, "-c", "raise SystemExit(7)"]) == 7
