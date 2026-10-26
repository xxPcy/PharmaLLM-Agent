"""Private standalone nnU-Net HTTP adapter. One GPU subprocess at a time, no DB needed."""

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import timedelta, datetime
import hashlib
import gzip
import shutil
import hmac
import json
import os
from pathlib import Path
import re
import numpy as np
import signal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse, JSONResponse

from pharma_agent.db import now
from pharma_agent.imaging import ImagingStore
from pharma_agent.main import BodyLimitMiddleware
from pharma_agent.settings import Settings
from pharma_agent.storage import LocalStorage
from pharma_agent.resident_model import ResidentModel, ResidentFailure

STAGES = {"preprocessing", "inference", "postprocessing"}


async def stop_child(process):
    if process is None or getattr(process, "returncode", None) is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), 5)
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


def build_command(settings, input_path, output_prefix, device):
    runner = Path(__file__).resolve().parents[2] / "scripts" / "nnunet_case_runner.py"
    return [
        str(settings.nnunet_python),
        "-u",
        str(runner),
        "--model-folder",
        str(settings.nnunet_model_folder.resolve()),
        "--input",
        str(input_path),
        "--output",
        str(output_prefix),
        "--device",
        device,
    ]


def create_segmentation_app(settings=None):
    settings = settings or Settings()
    secret = settings.segmentation_api_key.get_secret_value()
    if len(secret) < 24:
        raise ValueError("SEGMENTATION_API_KEY 需至少 24 字符，禁止无鉴权启动分割服务")
    root = settings.segmentation_service_root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = json.loads((settings.nnunet_model_folder / "model.json").read_text())
    checkpoint = settings.nnunet_model_folder / ("fold_" + str(manifest["fold"])) / manifest["checkpoint"]
    if not checkpoint.is_file() or not settings.nnunet_python.is_file():
        raise ValueError("未找到已准备的权重或独立 nnU-Net Python，请先运行模型准备脚本并核对 NNUNET_PYTHON")
    storage = LocalStorage(root / "validation")
    imaging = ImagingStore(storage, settings)
    mutex, slot, tasks, processes = asyncio.Lock(), asyncio.Semaphore(1), set(), {}
    device = os.environ.get("NNUNET_DEVICE", "cpu")
    resident = (
        ResidentModel(settings, root, manifest, device, stop_child)
        if settings.segmentation_load_mode == "resident"
        else None
    )

    def task_dir(task_id):
        if not re.fullmatch(r"seg_[a-f0-9]{32}", task_id):
            raise HTTPException(404, "任务不存在")
        return root / task_id

    def cleanup():
        for folder in root.glob("seg_*"):
            if folder.is_symlink() or not re.fullmatch(r"seg_[a-f0-9]{32}", folder.name):
                continue
            path = folder / "task.json"
            if path.is_file():
                state = json.loads(path.read_text())
                if folder.name not in processes and datetime.fromisoformat(state["expires_at"]) <= now():
                    shutil.rmtree(folder)

    def read(task_id):
        path = task_dir(task_id) / "task.json"
        if not path.exists():
            raise HTTPException(404, "任务不存在")
        result = json.loads(path.read_text())
        if datetime.fromisoformat(result["expires_at"]) <= now():
            raise HTTPException(410, "任务已过保留期限")
        return result

    def write(folder, state):
        temp = folder / "task.tmp"
        with temp.open("w") as handle:
            os.chmod(temp, 0o600)
            json.dump(state, handle)
        os.replace(temp, folder / "task.json")

    async def execute(task_id):
        process = None
        failure_code = None
        folder = task_dir(task_id)
        state = read(task_id)
        try:
            async with slot:
                state["status"] = "preprocessing"
                write(folder, state)
                env = {**os.environ, "nnUNet_compile": "false", "PYTHONUNBUFFERED": "1"}
                # Remove app-only import paths from the independent inference environment.
                env.pop("PYTHONPATH", None)
                command = build_command(settings, folder / "image_0000.nii.gz", folder / "mask", device)
                async with asyncio.timeout(settings.segmentation_timeout_seconds):
                    if resident:

                        async def phase(stage):
                            state["status"] = stage
                            write(folder, state)

                        await resident.run(task_id, folder / "image_0000.nii.gz", folder / "mask", phase)
                        code = 0
                    else:
                        process = await asyncio.create_subprocess_exec(
                            *command,
                            env=env,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                            start_new_session=True,
                            limit=1024 * 1024,
                        )
                        processes[task_id] = process
                        with (folder / "runner.log").open("wb") as log_file:
                            os.chmod(folder / "runner.log", 0o600)
                            while line := await process.stdout.readline():
                                log_file.write(line)
                                if line.strip() == b"PHARMA_ERROR:CUDA_OUT_OF_MEMORY":
                                    failure_code = "CUDA_OUT_OF_MEMORY"
                                if line.startswith(b"PHARMA_STAGE:"):
                                    stage = line.decode().strip().split(":", 1)[1]
                                    if stage in STAGES:
                                        state["status"] = stage
                                        write(folder, state)
                        code = await process.wait()
                    if code != 0 or not (folder / "mask.nii.gz").is_file():
                        raise RuntimeError("NNUNET_INFERENCE_FAILED")
                    # Preserve header units from the source. nnU-Net/ITK exporter may omit NIfTI unit codes;
                    # voxel transform is checked by the main worker before publishing the mask.
                    import nibabel as nib

                    source = nib.load(folder / "image_0000.nii.gz")
                    result = nib.load(folder / "mask.nii.gz")
                    if result.shape != source.shape or not np.allclose(
                        result.affine, source.affine, atol=1e-4
                    ):
                        raise RuntimeError("MASK_SPACE_MISMATCH")
                    result.header.set_xyzt_units(source.header.get_xyzt_units()[0])
                    nib.save(result, folder / "aligned.nii.gz")
                    os.replace(folder / "aligned.nii.gz", folder / "mask.nii.gz")
                    os.chmod(folder / "mask.nii.gz", 0o600)
                state["status"] = "completed"
                write(folder, state)
        except asyncio.CancelledError:
            state.update(status="failed", error_code="SERVICE_INTERRUPTED")
            write(folder, state)
            raise
        except Exception as exc:
            if resident and isinstance(exc, TimeoutError):
                resident.fail("RESIDENT_UNAVAILABLE")
                await resident.close()
            state.update(
                status="failed",
                error_code="NNUNET_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else exc.code
                if isinstance(exc, ResidentFailure)
                else failure_code or "NNUNET_INFERENCE_FAILED",
            )
            write(folder, state)
        finally:
            await stop_child(process)
            processes.pop(task_id, None)

    def enqueue(task_id):
        job = asyncio.create_task(execute(task_id))
        tasks.add(job)
        job.add_done_callback(tasks.discard)

    @asynccontextmanager
    async def lifespan(app):
        loading = asyncio.create_task(resident.start()) if resident else None
        cleanup()
        for path in root.glob("seg_*/task.json"):
            state = json.loads(path.read_text())
            if state["status"] == "queued" and datetime.fromisoformat(state["expires_at"]) > now():
                enqueue(path.parent.name)
            elif state["status"] in STAGES:
                state.update(status="failed", error_code="SERVICE_INTERRUPTED")
                write(path.parent, state)
        try:
            yield
        finally:
            for job in list(tasks):
                job.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if loading:
                loading.cancel()
                await asyncio.gather(loading, return_exceptions=True)
            if resident:
                await resident.close()

    app = FastAPI(title="Private PharmaLLM nnU-Net", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(
        BodyLimitMiddleware, upload_limit=settings.max_upload_bytes, upload_paths={"/v1/segmentations"}
    )

    async def authenticate(authorization: str = Header("")):
        if not hmac.compare_digest(authorization, "Bearer " + secret):
            raise HTTPException(401, "需要分割服务鉴权")

    @app.get("/health", dependencies=[Depends(authenticate)])
    async def health():
        state = resident.health() if resident else {"status": "ready"}
        body = {
            **state,
            "model_id": manifest["id"],
            "model_version": manifest["version"],
            "engine": "nnUNet",
            "api_version": 1,
            "load_mode": settings.segmentation_load_mode,
        }
        return JSONResponse(body, status_code=200 if state["status"] == "ready" else 503)

    @app.post("/v1/segmentations", status_code=202, dependencies=[Depends(authenticate)])
    async def submit(
        image: UploadFile = File(...),
        model_id: str = Form(...),
        task_type: str = Form(...),
        request_id: str = Form(...),
        idempotency_key: str = Header(...),
    ):
        if resident and resident.health()["status"] != "ready":
            await image.close()
            state = resident.health()
            return JSONResponse(
                {**state, "model_id": manifest["id"], "model_version": manifest["version"]}, status_code=503
            )
        if model_id != manifest["id"] or task_type not in manifest["task_types"]:
            raise HTTPException(422, "模型或任务不允许")
        if request_id != idempotency_key or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", request_id):
            raise HTTPException(422, "请求标识无效")
        try:
            body = await image.read(settings.max_upload_bytes + 1)
        finally:
            await image.close()
        if not body or len(body) > settings.max_upload_bytes:
            raise HTTPException(413, "影像超过大小上限")
        try:
            await asyncio.to_thread(imaging.nifti, body)
        except Exception:
            raise HTTPException(422, "需要有效单通道三维 NIfTI") from None
        task_id = "seg_" + hashlib.sha256(request_id.encode()).hexdigest()[:32]
        fingerprint = hashlib.sha256(body + manifest["version"].encode()).hexdigest()
        async with mutex:
            cleanup()
            folder = task_dir(task_id)
            if folder.exists():
                state = read(task_id)
                if state["fingerprint"] != fingerprint:
                    raise HTTPException(409, "同一标识不能用于不同输入")
                return {"task_id": task_id}
            if len(tasks) >= 8:
                raise HTTPException(429, "分割队列已满")
            folder.mkdir(mode=0o700)
            (folder / "image_0000.nii.gz").write_bytes(
                body if body.startswith(b"\x1f\x8b") else gzip.compress(body)
            )
            os.chmod(folder / "image_0000.nii.gz", 0o600)
            state = {
                "task_id": task_id,
                "status": "queued",
                "fingerprint": fingerprint,
                "model_id": manifest["id"],
                "model_version": manifest["version"],
                "expires_at": (now() + timedelta(days=settings.asset_retention_days)).isoformat(),
            }
            write(folder, state)
            enqueue(task_id)
        return {"task_id": task_id}

    @app.get("/v1/segmentations/{task_id}", dependencies=[Depends(authenticate)])
    async def status(task_id: str):
        state = read(task_id)
        return {
            k: state[k]
            for k in ("task_id", "status", "model_id", "model_version", "error_code")
            if k in state
        }

    @app.get("/v1/segmentations/{task_id}/mask", dependencies=[Depends(authenticate)])
    async def mask(task_id: str):
        state = read(task_id)
        if state["status"] != "completed":
            raise HTTPException(409, "分割尚未完成")
        return FileResponse(
            task_dir(task_id) / "mask.nii.gz",
            media_type="application/gzip",
            headers={"Cache-Control": "no-store"},
        )

    return app
