"""Authenticated imaging resources; no public filesystem paths or upstream secrets."""

import asyncio
from datetime import datetime

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select

from pharma_agent.api import CurrentActor, services
from pharma_agent.db import Asset, Conversation, ToolJob, audit, new_id, now
from pharma_agent.errors import AppError, not_found
from pharma_agent.observability import trace_id
from pharma_agent.security import rate_limit, require_owned

router = APIRouter(prefix="/api/v1")


async def ready_asset(svc, actor, asset_id):
    asset = await require_owned(svc, Asset, actor, asset_id)
    if asset.status != "ready":
        raise not_found("文件")
    meta = svc.imaging.metadata(asset)
    if meta.get("expires_at") and datetime.fromisoformat(meta["expires_at"]) <= now():
        raise AppError("ASSET_EXPIRED", "文件已到保留期限，请重新上传", 410)
    return asset, meta


@router.get("/imaging/capabilities")
async def capabilities(request: Request, actor: CurrentActor):
    svc = services(request)
    segmentation = svc.segmentation.public_config()
    detail = await svc.segmentation.probe_detail()
    segmentation.update(service_status=detail["status"], service_message=detail["message"], service_error_code=detail["code"])
    return {
        "max_upload_bytes": svc.settings.max_upload_bytes,
        "max_series_files": svc.settings.imaging_max_series_files,
        "retention_days": svc.settings.asset_retention_days,
        "segmentation": segmentation,
        "viewer": {"mpr": True, "surface_3d": True, "surface_method": "voxel_boundary"},
        "formats": ["png", "jpeg", "nii", "nii.gz", "dicom", "dicom_zip", "txt", "json"],
    }


@router.get("/model-info")
async def model_info(request: Request, actor: CurrentActor):
    svc = services(request)
    configured = svc.models.snapshot()
    bound = configured["models"][configured["roles"]["medical_answer"]]
    mode = "fake" if bound["provider"] == "fake" else "openai"
    try:
        await asyncio.wait_for(svc.models.health(), 6)
        status = "demo" if mode == "fake" else "ready"
    except (AppError, TimeoutError):
        status = "unavailable"
    return {
        "model_id": configured["models"][configured["roles"]["medical_answer"]].get(
            "model_name", "FakeModel"
        ),
        "revision": bound.get("revision", "unknown"),
        "mode": mode,
        "status": status,
        "graph_version": svc.graph_policy["version"],
        "configuration_version": configured.get("config_version"),
        "chat_models": svc.models.chat_choices() if hasattr(svc.models, "chat_choices") else [],
        "default_alias": configured["roles"]["medical_answer"],
    }


@router.get("/assets")
async def list_assets(
    request: Request,
    actor: CurrentActor,
    conversation_id: str | None = None,
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    svc = services(request)
    if conversation_id:
        await require_owned(svc, Conversation, actor, conversation_id)
    async with svc.sessions() as session:
        rows = (
            await session.scalars(
                select(Asset)
                .where(
                    Asset.tenant_id == actor.tenant_id,
                    Asset.owner_user_id == actor.id,
                    Asset.status == "ready",
                )
                .order_by(Asset.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        ).all()
    result = [svc.imaging.metadata(row) for row in rows]
    return [row for row in result if not conversation_id or row.get("conversation_id") == conversation_id]


@router.post("/imaging/uploads", status_code=201)
async def upload(
    request: Request,
    actor: CurrentActor,
    file: list[UploadFile] = File(...),
    modality: str = Form("unknown"),
    conversation_id: str | None = Form(None),
):
    svc = services(request)
    await rate_limit(svc, actor)
    if modality not in {"unknown", "CT", "MR", "T1", "T2", "FLAIR", "ADC", "DWI"}:
        raise AppError("INVALID_MODALITY", "模态不在允许范围内", 422)
    if conversation_id:
        await require_owned(svc, Conversation, actor, conversation_id)
    parts, total = [], 0
    try:
        if len(file) > svc.settings.imaging_max_series_files:
            raise AppError("FILE_TOO_LARGE", "序列文件数量超过上限", 413)
        for part in file:
            body = await part.read(svc.settings.max_upload_bytes - total + 1)
            total += len(body)
            if total > svc.settings.max_upload_bytes:
                raise AppError("FILE_TOO_LARGE", "上传总大小超过配置上限", 413)
            if not body:
                raise AppError("INVALID_ASSET", "文件不能为空", 422)
            parts.append((part.filename or "unnamed", body))
    finally:
        for part in file:
            await part.close()
    asset_id = new_id("asset")
    key = asset_id + ".bin"
    try:
        key, file_hash, mime, size, meta = await asyncio.to_thread(
            svc.imaging.prepare, asset_id, parts, modality, conversation_id
        )
        async with svc.sessions() as session, session.begin():
            session.add(
                Asset(
                    id=asset_id,
                    tenant_id=actor.tenant_id,
                    owner_user_id=actor.id,
                    object_key=key,
                    file_hash=file_hash,
                    mime=mime,
                    size_bytes=size,
                )
            )
            audit(session, actor, "asset.upload", "asset", asset_id, "succeeded", trace_id())
    except BaseException:
        svc.storage.remove(key)
        svc.imaging.purge(asset_id)
        raise
    return {**meta, "asset_id": asset_id, "size_bytes": size, "mime": mime}


@router.get("/assets/{asset_id}/metadata")
async def metadata(asset_id: str, request: Request, actor: CurrentActor):
    _, meta = await ready_asset(services(request), actor, asset_id)
    return meta


@router.get("/assets/{asset_id}/preview")
async def preview(asset_id: str, request: Request, actor: CurrentActor):
    svc = services(request)
    _, meta = await ready_asset(svc, actor, asset_id)
    if meta["kind"] not in {"image", "dicom_slice"}:
        raise AppError("UNSUPPORTED_PREVIEW", "此文件需要体数据查看器", 422)
    content = await asyncio.to_thread(svc.storage.path(asset_id + ".preview.png").read_bytes)
    return Response(content, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/assets/{asset_id}/slice")
async def slice_image(
    asset_id: str,
    request: Request,
    actor: CurrentActor,
    plane: str = Query("axial", pattern="^(axial|coronal|sagittal)$"),
    index: int = Query(0, ge=0),
    window: float | None = Query(None, gt=0, le=1e12),
    level: float | None = Query(None, ge=-1e12, le=1e12),
    mask_id: str | None = None,
    opacity: float = Query(0.45, ge=0, le=1),
    labels: str | None = None,
):
    svc = services(request)
    _, meta = await ready_asset(svc, actor, asset_id)
    if meta["kind"] != "volume":
        raise AppError("NOT_3D_VOLUME", "此文件不是三维医学体数据", 422)
    if mask_id:
        _, mask = await ready_asset(svc, actor, mask_id)
        if mask.get("source_asset_id") != asset_id:
            raise AppError("MASK_SPACE_MISMATCH", "此结果不属于当前影像", 422)
    try:
        label_ids = [int(v) for v in labels.split(",") if v] if labels is not None else None
        if label_ids is not None and len(label_ids) > 256:
            raise ValueError
    except ValueError:
        raise AppError("INVALID_REQUEST", "标签编号无效", 422) from None
    content = await asyncio.to_thread(
        svc.imaging.slice, asset_id, plane, index, window, level, mask_id, opacity, label_ids
    )
    return Response(content, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/conversations/{conversation_id}/imaging-jobs")
async def conversation_jobs(conversation_id: str, request: Request, actor: CurrentActor):
    from pharma_agent.tools import job_dict

    svc = services(request)
    await require_owned(svc, Conversation, actor, conversation_id)
    async with svc.sessions() as session:
        rows = (
            await session.scalars(
                select(ToolJob)
                .where(
                    ToolJob.tenant_id == actor.tenant_id,
                    ToolJob.user_id == actor.id,
                    ToolJob.tool_name == "segment_medical_volume",
                    ToolJob.input_json["conversation_id"].as_string() == conversation_id,
                )
                .order_by(ToolJob.created_at.desc())
                .limit(50)
            )
        ).all()
    return [job_dict(row) for row in reversed(rows)]


@router.post("/assets/{asset_id}/modality")
async def declare_modality(asset_id: str, request: Request, actor: CurrentActor, modality: str = Form(...)):
    svc = services(request)
    if modality not in {"unknown", "CT", "MR", "T1", "T2", "FLAIR", "ADC", "DWI"}:
        raise AppError("INVALID_MODALITY", "不支持的模态", 422)
    async with svc.sessions() as session, session.begin():
        asset = await require_owned(svc, Asset, actor, asset_id, session, lock=True)
        if asset.status != "ready":
            raise not_found("文件")
        meta = svc.imaging.metadata(asset)
        if meta.get("kind") != "volume":
            raise AppError("NOT_3D_VOLUME", "仅三维影像可声明分割输入模态", 422)
        actual = meta.get("acquisition_modality", "unknown")
        declared = "MR" if modality in {"T1", "T2", "FLAIR", "ADC", "DWI"} else modality
        if actual != "unknown" and declared not in {actual, "unknown"}:
            raise AppError("MODALITY_MISMATCH", "与 DICOM 采集模态不一致", 422)
        meta.update(modality=modality, modality_source="user")
        svc.imaging.save_meta(asset_id, meta)
        audit(session, actor, "asset.declare_modality", "asset", asset_id, "succeeded", trace_id())
    return meta


@router.get("/assets/{asset_id}/surface")
async def surface(asset_id: str, request: Request, actor: CurrentActor):
    from pharma_agent.surfaces import mask_surface

    svc = services(request)
    _, meta = await ready_asset(svc, actor, asset_id)
    if meta.get("kind") != "mask":
        raise AppError("NOT_SEGMENTATION", "三维表面需要真实的分割结果", 422)
    await ready_asset(svc, actor, meta["source_asset_id"])
    return await asyncio.to_thread(mask_surface, svc.imaging.load(asset_id, display=False), meta["labels"])
