"""Launcher composition and scoped cleanup with fake children; never starts Celery or a GPU."""

import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return runpy.run_path(str(SCRIPTS / "start_worker.py"))


def test_legacy_cpu_worker_does_not_start_segmentation_or_keep_cuda_mask(monkeypatch):
    script = launcher(monkeypatch)
    script["main"].__globals__["load"] = lambda: None
    monkeypatch.setattr(sys, "argv", ["start_worker.py"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    calls = []
    monkeypatch.setattr(os, "execv", lambda python, args: calls.append(args))
    assert script["main"]() is None
    assert calls[0][1:3] == ["-m", "celery"] and os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_combined_launcher_requires_matching_private_root_endpoint(monkeypatch):
    script = launcher(monkeypatch)
    assert script["local_endpoint"]("http://127.0.0.1:8092", 8092) == ("127.0.0.1", 8092)
    for url, port in [
        ("http://127.0.0.1:8092/v1", 8092),
        ("http://remote:8092", 8092),
        ("http://127.0.0.1:8092", 8093),
    ]:
        with pytest.raises(ValueError):
            script["local_endpoint"](url, port)


def test_seg_gpu_option_builds_resident_service_for_the_requested_card(monkeypatch):
    from pharma_agent import settings as settings_module, segmentation as gateway_module

    script = launcher(monkeypatch)
    scope = script["main"].__globals__
    scope["load"] = lambda: None
    fake_settings = SimpleNamespace(
        segmentation_base_url="http://127.0.0.1:8092",
        segmentation_api_key=SimpleNamespace(get_secret_value=lambda: "synthetic-key-" * 3),
        segmentation_startup_timeout_seconds=300,
    )
    monkeypatch.setattr(settings_module, "Settings", lambda: fake_settings)
    monkeypatch.setattr(gateway_module, "SegmentationGateway", lambda settings: "fixture-gateway")
    scope["ensure_port_free"] = lambda *args: None
    invoke = Mock(return_value=0)
    scope["supervise"] = invoke
    monkeypatch.setattr(sys, "argv", ["start_worker.py", "--seg-gpu", "4"])
    assert script["main"]() == 0
    command, env, gateway, timeout = invoke.call_args.args
    assert command[command.index("--gpu") + 1] == "4"
    assert command[command.index("--load-mode") + 1] == "resident"
    assert command[command.index("--port") + 1] == "8092"
    assert gateway == "fixture-gateway" and timeout == 300


def test_busy_port_is_reported_before_any_subprocess_is_started(monkeypatch):
    script = launcher(monkeypatch)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def connect_ex(self, address):
            return 0

    monkeypatch.setattr(script["ensure_port_free"].__globals__["socket"], "socket", Connection)
    with pytest.raises(ValueError, match="未结束任何进程"):
        script["ensure_port_free"]("127.0.0.1", 8092)


def test_combined_launcher_waits_for_model_readiness(monkeypatch):
    script = launcher(monkeypatch)
    gateway = SimpleNamespace(
        probe_detail=AsyncMock(
            side_effect=[{"status": "loading", "message": "loading"}, {"status": "ready", "message": "ready"}]
        )
    )
    monkeypatch.setattr(script["wait_ready"].__globals__["time"], "sleep", lambda value: None)
    script["wait_ready"](SimpleNamespace(poll=lambda: None), gateway, 10)
    assert gateway.probe_detail.await_count == 2


@pytest.mark.parametrize("failed", [False, True])
def test_combined_supervisor_starts_cpu_worker_only_after_ready_and_cleans_owned_children(
    monkeypatch, failed
):
    script = launcher(monkeypatch)
    scope = script["supervise"].__globals__
    processes, commands, cleaned = [], [], []

    class Process:
        def __init__(self, number):
            self.pid = 900001 + number
            self.code = 0 if number == 1 else None

        def poll(self):
            return self.code

        def wait(self, timeout):
            return self.code

    def popen(command, **kwargs):
        assert kwargs["start_new_session"]
        if processes:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        commands.append(command)
        process = Process(len(processes))
        processes.append(process)
        return process

    def stop(pid, grace_seconds):
        cleaned.append(pid)
        next(process for process in processes if process.pid == pid).code = -15

    def ready(*args):
        assert len(commands) == 1
        if failed:
            raise RuntimeError("fixture model failed")

    monkeypatch.setattr(scope["subprocess"], "Popen", popen)
    scope["wait_ready"], scope["stop_process_group"] = ready, stop
    if failed:
        with pytest.raises(RuntimeError):
            script["supervise"](["fake-seg"], {}, None, 10)
    else:
        assert script["supervise"](["fake-seg"], {}, None, 10) == 0
        assert "celery" in commands[1]
    assert len(commands) == (1 if failed else 2)
    assert cleaned == [900001]
