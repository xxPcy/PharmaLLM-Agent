"""Private IPC supervisor for one model in the independent nnU-Net Python.

This module never imports torch/nnU-Net, and never accepts paths from an HTTP client.
"""

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path

PREFIX = b"PHARMA_RESIDENT:"
ERRORS = {
    "CUDA_OUT_OF_MEMORY",
    "MODEL_LOAD_FAILED",
    "MODEL_STARTUP_TIMEOUT",
    "RESIDENT_UNAVAILABLE",
    "RESIDENT_PROTOCOL_ERROR",
    "NNUNET_INFERENCE_FAILED",
}


class ResidentFailure(RuntimeError):
    def __init__(self, code):
        self.code = code if code in ERRORS else "NNUNET_INFERENCE_FAILED"
        super().__init__(self.code)


class ResidentModel:
    def __init__(self, settings, root, manifest, device, stop_child):
        self.settings, self.root, self.manifest, self.device = settings, root, manifest, device
        self.stop_child = stop_child
        self.status, self.error_code = "loading", None
        self.process = self.reader = None
        self.events, self.started = asyncio.Queue(maxsize=64), asyncio.Event()
        self.lock = asyncio.Lock()
        self.closing = self.busy = False

    def command(self):
        runner = Path(__file__).resolve().parents[2] / "scripts/nnunet_case_runner.py"
        return [
            str(self.settings.nnunet_python),
            "-u",
            str(runner),
            "--model-folder",
            str(self.settings.nnunet_model_folder.resolve()),
            "--device",
            self.device,
            "--resident",
            "--task-root",
            str(self.root),
        ]

    def fail(self, code):
        self.status = "failed"
        self.error_code = self.error_code or code
        self.started.set()

    async def read_events(self):
        path = self.root / "resident.log"
        try:
            with path.open("ab") as handle:
                os.chmod(path, 0o600)
                while line := await self.process.stdout.readline():
                    handle.write(line)
                    handle.flush()
                    if line.startswith(PREFIX):
                        value = json.loads(line[len(PREFIX) :])
                        if not isinstance(value, dict):
                            raise ValueError("Invalid resident frame")
                        if value.get("event") == "startup_failed":
                            code = value.get("error_code")
                            self.error_code = code if code in ERRORS else "MODEL_LOAD_FAILED"
                        self.events.put_nowait(value)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.fail("RESIDENT_PROTOCOL_ERROR")
        finally:
            if not self.closing:
                self.fail("RESIDENT_UNAVAILABLE")
                if self.events.full():
                    while not self.events.empty():
                        self.events.get_nowait()
                self.events.put_nowait({"event": "unavailable"})

    async def start(self):
        try:
            env = {
                **os.environ,
                "nnUNet_compile": "false",
                "PYTHONUNBUFFERED": "1",
                "PHARMA_PARENT_PID": str(os.getpid()),
            }
            env.pop("PYTHONPATH", None)
            self.process = await asyncio.create_subprocess_exec(
                *self.command(),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                limit=1024 * 1024,
            )
            self.reader = asyncio.create_task(self.read_events())
            async with asyncio.timeout(self.settings.segmentation_startup_timeout_seconds):
                value = await self.events.get()
            if value.get("event") == "startup_failed":
                raise ResidentFailure(value.get("error_code"))
            if (
                value.get("event") != "ready"
                or value.get("model_id") != self.manifest["id"]
                or value.get("model_version") != self.manifest["version"]
            ):
                raise ResidentFailure("RESIDENT_PROTOCOL_ERROR")
            if self.status == "failed" or self.process.returncode is not None:
                raise ResidentFailure(self.error_code or "RESIDENT_UNAVAILABLE")
            self.status = "ready"
            print("分割常驻模型已加载；等待任务。", flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.fail(
                "MODEL_STARTUP_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else exc.code
                if isinstance(exc, ResidentFailure)
                else "MODEL_LOAD_FAILED"
            )
            print(f"分割常驻模型未就绪：{self.error_code}；详情见私有 resident.log。", flush=True)
            await self.stop_child(self.process)
        finally:
            self.started.set()

    def health(self):
        if self.status == "ready" and (not self.process or self.process.returncode is not None):
            self.fail("RESIDENT_UNAVAILABLE")
        return {"status": self.status, "error_code": self.error_code, "busy": self.busy}

    async def run(self, task_id, input_path, output_prefix, phase):
        await self.started.wait()
        async with self.lock:
            if self.health()["status"] != "ready":
                raise ResidentFailure(self.error_code or "RESIDENT_UNAVAILABLE")
            self.busy = True
            try:
                message = {"request_id": task_id, "input": str(input_path), "output": str(output_prefix)}
                self.process.stdin.write((json.dumps(message) + "\n").encode())
                await self.process.stdin.drain()
                while True:
                    value = await self.events.get()
                    if value.get("event") == "unavailable":
                        raise ResidentFailure(self.error_code or "RESIDENT_UNAVAILABLE")
                    if value.get("request_id") != task_id:
                        self.fail("RESIDENT_PROTOCOL_ERROR")
                        raise ResidentFailure("RESIDENT_PROTOCOL_ERROR")
                    kind = value.get("event")
                    if kind == "phase" and value.get("phase") in {
                        "preprocessing",
                        "inference",
                        "postprocessing",
                    }:
                        await phase(value["phase"])
                    elif kind == "completed":
                        return
                    elif kind == "failed":
                        raise ResidentFailure(value.get("error_code"))
                    else:
                        self.fail("RESIDENT_PROTOCOL_ERROR")
                        raise ResidentFailure("RESIDENT_PROTOCOL_ERROR")
            except (BrokenPipeError, ConnectionError):
                self.fail("RESIDENT_UNAVAILABLE")
                raise ResidentFailure("RESIDENT_UNAVAILABLE") from None
            finally:
                self.busy = False

    async def close(self):
        self.closing = True
        if self.process and self.process.stdin:
            with suppress(Exception):
                self.process.stdin.close()
        await self.stop_child(self.process)
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.status != "failed":
            self.status = "stopped"
