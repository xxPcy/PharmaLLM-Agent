"""Isolated launcher and error-reporting regressions; no CUDA or model imports."""

import builtins
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_launcher_pins_requested_physical_gpu_by_uuid(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    launcher = runpy.run_path(str(SCRIPTS / "start_segmentation.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    monkeypatch.setenv("NNUNET_GPU_INDEX", "old-value")
    monkeypatch.setenv("NNUNET_DEVICE", "cpu")
    monkeypatch.setattr(sys, "argv", ["start_segmentation.py", "--gpu", "4", "--port", "8092"])
    launcher["main"].__globals__["load"] = lambda: None
    gpu_uuid = "GPU-00000000-0000-0000-0000-000000000004"

    def query(command, **kwargs):
        assert command[command.index("-i") + 1] == "4"
        assert kwargs["timeout"] == 10
        return SimpleNamespace(stdout=f"4, {gpu_uuid}, Synthetic GPU, 45000, 81920\n")

    class InterceptExec(Exception):
        pass

    def intercept(binary, args):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu_uuid
        assert os.environ["NNUNET_GPU_INDEX"] == "4"
        assert os.environ["NNUNET_DEVICE"] == "cuda"
        assert args[args.index("--port") + 1] == "8092"
        raise InterceptExec

    monkeypatch.setattr(subprocess, "run", query)
    monkeypatch.setattr(os, "execv", intercept)
    with pytest.raises(InterceptExec):
        launcher["main"]()


@pytest.mark.parametrize(
    "message,expected_marker",
    [
        ("CUDA error: out of memory", True),
        ("CUDA out of memory. Tried to allocate a block", True),
        ("CPU allocator: out of memory", False),
    ],
)
def test_runner_reports_cuda_oom_without_reclassifying_cpu_errors(
    monkeypatch, capsys, message, expected_marker
):
    # Interrupt the first heavy import; the actual nnU-Net/torch packages never execute.
    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError(message)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nnunet_case_runner.py",
            "--model-folder",
            "unused",
            "--input",
            "unused",
            "--output",
            "unused",
            "--device",
            "cuda",
        ],
    )
    with pytest.raises(RuntimeError, match=message):
        runpy.run_path(str(SCRIPTS / "nnunet_case_runner.py"), run_name="__main__")
    assert ("PHARMA_ERROR:CUDA_OUT_OF_MEMORY" in capsys.readouterr().out) == expected_marker
