"""Allowlisted, independent HTTP segmentation adapter and durable worker."""

import asyncio
import gzip
import re
from datetime import timedelta

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from pharma_agent.db import Asset, ToolJob, User, create_database, new_id, now, audit
from pharma_agent.errors import AppError
from pharma_agent.imaging import ImagingStore
from pharma_agent.models import config_load
from pharma_agent.observability import digest, log, trace_id
from pharma_agent.storage import LocalStorage


class SegmentationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=120)
    task_types: list[str] = Field(min_length=1)
    modalities: list[str] = Field(min_length=1)
    channels: int = Field(1, ge=1, le=8)
    labels: dict[int, str] = Field(min_length=2, max_length=256)


def service_http_error(response):
    """Map only known statuses/codes; never expose upstream bodies, URLs or credentials."""
    if response.status_code in {401, 403}:
        return AppError(
            "SEGMENTATION_AUTH_FAILED", "分割服务鉴权失败，请核对 API、Worker 与分割服务的密钥并重启", 503
        )
    if response.status_code == 404:
        return AppError(
            "SEGMENTATION_ENDPOINT_NOT_FOUND",
            "分割服务接口或任务不存在，请核对服务根地址（不带 /v1）并重试",
            503,
        )
    if response.status_code == 429:
        return AppError("SEGMENTATION_BUSY", "分割队列已满，请等待已有任务完成后重试", 503, True)
    try:
        body = response.json()
    except ValueError:
        body = {}
    if isinstance(body, dict):
        code = body.get("error_code")
        if body.get("status") == "loading":
            return AppError("SEGMENTATION_LOADING", "分割模型正在加载，等待终端 C 显示就绪后重试", 503, True)
        if code == "CUDA_OUT_OF_MEMORY":
            return AppError(
                "SEGMENTATION_CUDA_OUT_OF_MEMORY",
                "分割 GPU 显存不足，请换有足够空闲显存的卡并重启分割服务",
                503,
            )
        if code in {
            "MODEL_LOAD_FAILED",
            "MODEL_STARTUP_TIMEOUT",
            "RESIDENT_UNAVAILABLE",
            "RESIDENT_PROTOCOL_ERROR",
        }:
            return AppError(
                "SEGMENTATION_MODEL_NOT_READY",
                "分割常驻模型未就绪，请查看终端 C 和私有 resident.log 后重启",
                503,
            )
    return AppError("SEGMENTATION_HTTP_ERROR", "分割服务返回错误，请检查终端 C 的分割服务状态", 503, True)


class SegmentationGateway:
    def __init__(self, settings):
        self.settings = settings
        config = config_load(settings.segmentation_models_config)
        rows = [SegmentationModel.model_validate(row) for row in config.get("models", [])]
        self.models = {row.id: row for row in rows}
        if len(self.models) != len(rows) or any(
            0 not in row.labels or any(v < 0 for v in row.labels) for row in rows
        ):
            raise ValueError("分割模型 ID 需唯一，标签需包含 0 背景且编号非负")

    def public_config(self):
        return {
            "status": "configured" if self.settings.segmentation_base_url and self.models else "unconfigured",
            "models": [row.model_dump(mode="json") for row in self.models.values()],
        }

    async def probe(self, transport=None):
        return (await self.probe_detail(transport))["status"]

    async def probe_detail(self, transport=None):
        if self.public_config()["status"] == "unconfigured":
            return {
                "status": "unconfigured",
                "code": "SEGMENTATION_UNCONFIGURED",
                "message": "分割服务未配置，仍可查看影像",
            }
        headers = {"Authorization": "Bearer " + self.settings.segmentation_api_key.get_secret_value()}
        try:
            async with httpx.AsyncClient(
                trust_env=False, transport=transport, timeout=2, follow_redirects=False
            ) as client:
                response = await client.get(
                    self.settings.segmentation_base_url.rstrip("/") + "/health", headers=headers
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError
                if result.get("status") == "loading":
                    return {
                        "status": "loading",
                        "code": "SEGMENTATION_LOADING",
                        "message": "分割模型正在加载，请等待终端 C 显示就绪",
                    }
                if not isinstance(result.get("model_id"), str) or not isinstance(
                    result.get("model_version"), str
                ):
                    raise ValueError
                model = self.models.get(result.get("model_id"))
                if result.get("api_version", 1) != 1:
                    return {
                        "status": "incompatible",
                        "code": "SEGMENTATION_PROTOCOL_ERROR",
                        "message": "分割服务协议版本不匹配，请更新并重启分割服务",
                    }
                if not model or result.get("model_version") != model.version:
                    return {
                        "status": "incompatible",
                        "code": "MODEL_VERSION_MISMATCH",
                        "message": "分割服务模型版本与配置不一致，请核对权重配置并重启",
                    }
                if result.get("status") != "ready":
                    return {
                        "status": "unavailable",
                        "code": "SEGMENTATION_MODEL_NOT_READY",
                        "message": "分割模型未就绪，请查看终端 C 的启动信息",
                    }
                return {"status": "ready", "code": None, "message": "分割服务已就绪"}
        except httpx.HTTPStatusError as exc:
            error = service_http_error(exc.response)
            return {
                "status": "loading" if error.code == "SEGMENTATION_LOADING" else "unavailable",
                "code": error.code,
                "message": error.message,
            }
        except httpx.ConnectError:
            return {
                "status": "offline",
                "code": "SEGMENTATION_CONNECTION_FAILED",
                "message": "无法连接分割服务，请在终端 C 使用 --seg-gpu 指定显卡启动",
            }
        except httpx.TimeoutException:
            return {
                "status": "offline",
                "code": "SEGMENTATION_CONNECTION_TIMEOUT",
                "message": "分割服务响应超时，请检查终端 C",
            }
        except (ValueError, TypeError):
            return {
                "status": "incompatible",
                "code": "SEGMENTATION_PROTOCOL_ERROR",
                "message": "分割服务返回格式不符合接口，请检查是否连接到了正确服务",
            }
        except httpx.HTTPError:
            return {
                "status": "offline",
                "code": "SEGMENTATION_CONNECTION_FAILED",
                "message": "分割服务连接异常，请检查终端 C",
            }

    def validate(self, meta, model_id, task_type):
        if not self.settings.segmentation_base_url or not self.models:
            raise AppError("SEGMENTATION_UNCONFIGURED", "分割服务未配置；可先上传并查看影像", 503)
        spec = self.models.get(model_id)
        if not spec or task_type not in spec.task_types:
            raise AppError("SEGMENTATION_MODEL_NOT_ALLOWED", "分割模型或任务不在服务端白名单中", 422)
        if meta.get("kind") != "volume" or len(meta.get("dimensions", [])) != 3:
            raise AppError("NOT_3D_VOLUME", "该模型需要三维体数据，不能使用二维图片或单层 DICOM", 422)
        if meta.get("channels") != spec.channels:
            raise AppError("CHANNEL_MISMATCH", "文件通道数不符合模型要求；当前入口每个文件为单通道", 422)
        if meta.get("modality") not in spec.modalities:
            raise AppError("MODALITY_MISMATCH", "影像模态未确认或与模型要求不符；请上传时准确声明模态", 422)
        return spec

    async def infer(self, path, spec, task_type, key, phase, transport=None):
        headers = (
            {"Authorization": "Bearer " + self.settings.segmentation_api_key.get_secret_value()}
            if self.settings.segmentation_api_key.get_secret_value()
            else {}
        )
        async with httpx.AsyncClient(
            base_url=self.settings.segmentation_base_url.rstrip("/") + "/",
            headers=headers,
            trust_env=False,
            follow_redirects=False,
            timeout=30,
            transport=transport,
        ) as client:
            try:
                with path.open("rb") as handle:
                    response = await client.post(
                        "v1/segmentations",
                        files={"image": ("image.nii.gz", handle, "application/gzip")},
                        data={"model_id": spec.id, "task_type": task_type, "request_id": key},
                        headers={"Idempotency-Key": key},
                    )
                response.raise_for_status()
                remote_id = response.json()["task_id"]
                if not isinstance(remote_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", remote_id):
                    raise ValueError
                await phase("queued", remote_id)
                while True:
                    response = await client.get(f"v1/segmentations/{remote_id}")
                    response.raise_for_status()
                    result = response.json()
                    status = result["status"]
                    if status == "failed":
                        if result.get("error_code") == "CUDA_OUT_OF_MEMORY":
                            raise AppError(
                                "SEGMENTATION_CUDA_OUT_OF_MEMORY",
                                "分割 GPU 显存不足；请选择空闲显存更多的 GPU，重启分割服务后重试",
                                503,
                            )
                        if result.get("error_code") in {
                            "MODEL_LOAD_FAILED",
                            "MODEL_STARTUP_TIMEOUT",
                            "RESIDENT_UNAVAILABLE",
                            "RESIDENT_PROTOCOL_ERROR",
                        }:
                            raise AppError(
                                "SEGMENTATION_MODEL_NOT_READY",
                                "分割常驻模型异常，请查看终端 C 和私有 resident.log 后重启",
                                503,
                            )
                        raise AppError(
                            "SEGMENTATION_FAILED", "分割服务执行失败，请检查服务日志或输入要求", 502
                        )
                    if status == "completed":
                        if result.get("model_id") != spec.id or result.get("model_version") != spec.version:
                            raise AppError("MODEL_VERSION_MISMATCH", "分割服务模型版本与配置不一致", 502)
                        break
                    if status not in {"queued", "preprocessing", "inference", "postprocessing"}:
                        raise ValueError
                    await phase(status, remote_id)
                    await asyncio.sleep(2)
                await phase("postprocessing", remote_id)
                async with client.stream("GET", f"v1/segmentations/{remote_id}/mask") as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.settings.max_upload_bytes:
                            raise AppError("MASK_TOO_LARGE", "分割结果超过配置大小上限", 502)
                        chunks.append(chunk)
                return b"".join(chunks)
            except AppError:
                raise
            except httpx.HTTPStatusError as exc:
                raise service_http_error(exc.response) from None
            except httpx.ConnectError:
                raise AppError(
                    "SEGMENTATION_CONNECTION_FAILED",
                    "无法连接分割服务，请在终端 C 使用 --seg-gpu 指定显卡启动",
                    503,
                    True,
                ) from None
            except httpx.TimeoutException:
                raise AppError(
                    "SEGMENTATION_CONNECTION_TIMEOUT", "分割服务响应超时，请检查终端 C 后重试", 503, True
                ) from None
            except (ValueError, KeyError, TypeError):
                raise AppError(
                    "SEGMENTATION_PROTOCOL_ERROR",
                    "分割服务返回格式不符合接口，请检查是否连接到了正确服务",
                    503,
                ) from None
            except httpx.HTTPError:
                raise AppError(
                    "SEGMENTATION_CONNECTION_FAILED", "分割服务连接中断，请检查终端 C 后重试", 503, True
                ) from None


async def create_segmentation_draft(
    svc, actor, args, conversation_id, key, allowed_ids, execution_mode="ask_for_approval"
):
    from pharma_agent.imaging_api import ready_asset
    from pharma_agent.security import Actor, require_scope

    if execution_mode not in {"ask_for_approval", "full_access"}:
        raise AppError("INVALID_REQUEST", "执行模式无效", 422)
    if args.file_id not in allowed_ids:
        raise AppError("ATTACHMENT_REQUIRED", "请先在当前会话上传影像，或明确指出已上传的文件", 422)
    require_scope(actor, "tasks:draft")
    _, meta = await ready_asset(svc, actor, args.file_id)
    spec = svc.segmentation.validate(meta, args.model_id, args.task_type)
    inputs = {
        **args.model_dump(),
        "conversation_id": conversation_id,
        "file_name": meta["name"],
        "model": spec.model_dump(mode="json"),
        "execution_mode": execution_mode,
        "department_id": actor.department_id,
    }
    async with svc.sessions() as session, session.begin():
        current_user = await session.scalar(
            select(User).where(User.id == actor.id, User.tenant_id == actor.tenant_id).with_for_update()
        )
        if not current_user or current_user.status != "active":
            raise AppError("PERMISSION_DENIED", "账户已停用", 403)
        require_scope(
            Actor(current_user.id, current_user.tenant_id, current_user.role, current_user.department_id),
            "tasks:draft",
        )
        previous = (
            await session.scalars(
                select(ToolJob).where(
                    ToolJob.tenant_id == actor.tenant_id,
                    ToolJob.user_id == actor.id,
                    ToolJob.tool_name == "segment_medical_volume",
                    ToolJob.idempotency_key == key,
                )
            )
        ).first()
        if previous:
            if digest({k: v for k, v in previous.input_json.items() if k != "attempt"}) != digest(inputs):
                raise AppError("IDEMPOTENCY_CONFLICT", "同一幂等键不能用于不同分割请求", 409)
            return previous
        job = ToolJob(
            id=new_id("job"),
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            tool_name="segment_medical_volume",
            tool_version="1.0",
            status="draft",
            idempotency_key=key,
            input_json=inputs,
            result_json={"phase": "draft", "labels": spec.model_dump(mode="json")["labels"]},
        )
        session.add(job)
        audit(session, actor, "job.segmentation_draft", "job", job.id, "created", trace_id())
        await session.flush()
        return job


async def execute_segmentation_job(settings, job_id):
    engine, sessions = create_database(settings.database_url)
    storage = LocalStorage(settings.storage_root)
    imaging, gateway = ImagingStore(storage, settings), SegmentationGateway(settings)
    claim, result_id, completed = new_id("claim"), None, False
    try:
        async with sessions() as session, session.begin():
            job = await session.get(ToolJob, job_id, with_for_update=True)
            if (
                not job
                or job.tool_name != "segment_medical_volume"
                or not job.confirmed_at
                or job.status not in {"queued", "running"}
            ):
                return
            if job.status == "running" and job.lease_until and job.lease_until > now():
                return
            job.status, job.claim_token = "running", claim
            job.lease_until = now() + timedelta(seconds=120)
            inputs, tenant_id, user_id = dict(job.input_json), job.tenant_id, job.user_id

        async def phase(stage, remote_id=None):
            async with sessions() as session, session.begin():
                item = await session.get(ToolJob, job_id, with_for_update=True)
                if item.claim_token != claim or item.status != "running":
                    raise AppError("JOB_LEASE_LOST", "任务执行权已失效", 409)
                item.result_json = {**item.result_json, "phase": stage, "remote_task_id": remote_id}
                item.lease_until = now() + timedelta(seconds=120)

        async with asyncio.timeout(settings.segmentation_timeout_seconds):
            await phase("preprocessing")
            # Recheck ownership/status at execution; never follow file paths supplied by the LLM.
            async with sessions() as session:
                asset = await session.get(Asset, inputs["file_id"])
                if (
                    not asset
                    or asset.tenant_id != tenant_id
                    or asset.owner_user_id != user_id
                    or asset.status != "ready"
                ):
                    raise AppError("ASSET_UNAVAILABLE", "源影像已删除或不可使用", 422)
            spec = gateway.validate(imaging.metadata(asset), inputs["model_id"], inputs["task_type"])
            if spec.model_dump(mode="json") != inputs["model"]:
                raise AppError("MODEL_VERSION_MISMATCH", "模型配置已更改，请重新创建任务", 409)
            body = await gateway.infer(
                storage.path(asset.id + ".volume.nii.gz"),
                spec,
                inputs["task_type"],
                job_id + "-" + str(inputs.get("attempt", 1)),
                phase,
            )
            result_id = new_id("asset")
            meta = await asyncio.to_thread(
                imaging.accept_mask, result_id, body, asset.id, spec.model_dump(mode="json")["labels"]
            )
            key = result_id + ".bin"
            if not body.startswith(b"\x1f\x8b"):
                body = gzip.compress(body)
            file_hash = storage.write(key, body)
            meta.update(
                conversation_id=inputs["conversation_id"],
                retention_days=settings.asset_retention_days,
                expires_at=(now() + timedelta(days=settings.asset_retention_days)).isoformat(),
            )
            imaging.save_meta(result_id, meta)
            async with sessions() as session, session.begin():
                item = await session.get(ToolJob, job_id, with_for_update=True)
                source = await session.get(Asset, asset.id, with_for_update=True)
                if item.claim_token != claim or item.status != "running" or source.status != "ready":
                    raise AppError("ASSET_UNAVAILABLE", "任务失效或源影像已删除", 409)
                session.add(
                    Asset(
                        id=result_id,
                        tenant_id=tenant_id,
                        owner_user_id=user_id,
                        object_key=key,
                        file_hash=file_hash,
                        mime="application/gzip",
                        size_bytes=len(body),
                    )
                )
                item.status, item.finished_at, item.lease_until = "succeeded", now(), None
                item.result_json = {
                    "phase": "completed",
                    "asset_id": result_id,
                    "source_asset_id": asset.id,
                    "model_id": spec.id,
                    "model_version": spec.version,
                    "labels": meta["labels"],
                    "measurements": meta["measurements"],
                    "interpretation_limit": "仅分割标签与体积，不等同于疾病诊断",
                }
            completed = True
    except Exception as exc:
        code = (
            exc.code
            if isinstance(exc, AppError)
            else "SEGMENTATION_TIMEOUT"
            if isinstance(exc, TimeoutError)
            else "SEGMENTATION_FAILED"
        )
        log("segmentation.failed", job_id=job_id, error_code=code)
        async with sessions() as session, session.begin():
            job = await session.get(ToolJob, job_id, with_for_update=True)
            if job and job.claim_token == claim and job.status == "running":
                job.status, job.error_code, job.finished_at, job.lease_until = "failed", code, now(), None
                job.result_json = {
                    **job.result_json,
                    "phase": "failed",
                    "error_message": exc.message
                    if isinstance(exc, AppError)
                    else "分割执行未完成，请检查服务后重试",
                }
    finally:
        if result_id and not completed:
            storage.remove(result_id + ".bin")
            imaging.purge(result_id)
        await engine.dispose()
