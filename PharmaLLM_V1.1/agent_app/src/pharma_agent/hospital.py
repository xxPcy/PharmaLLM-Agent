"""Small organization/RBAC and explicit department-scoped result review workflow.

Review access never changes ownership or grants access to the doctor's conversations.
"""

import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from pharma_agent.db import (
    Asset,
    Department,
    SegmentationReview,
    Tenant,
    ToolJob,
    User,
    audit,
    new_id,
    now,
)
from pharma_agent.errors import AppError, not_found
from pharma_agent.observability import trace_id
from pharma_agent.schemas import (
    DepartmentInput,
    ExecutionModeInput,
    MemberInput,
    MemberUpdate,
    ReviewDecision,
    TeamInput,
)
from pharma_agent.security import Actor, authenticate, hash_password, require_owned, require_scope

router = APIRouter(prefix="/api/v1")
CurrentActor = Annotated[Actor, Depends(authenticate)]


async def profile(svc, actor):
    async with svc.sessions() as session:
        user = await session.get(User, actor.id)
        tenant = await session.get(Tenant, actor.tenant_id)
        department = await session.get(Department, user.department_id) if user.department_id else None
        return {
            "user_id": user.id,
            "tenant_id": tenant.id,
            "tenant_code": tenant.login_code,
            "tenant_name": tenant.name,
            "email": user.email,
            "display_name": user.display_name,
            "role": user.role,
            "department_id": user.department_id,
            "department_name": department.name if department else None,
            "execution_mode": user.execution_mode,
            "scopes": sorted(actor.scopes),
        }


async def department_in_tenant(session, tenant_id, department_id):
    department = await session.get(Department, department_id) if department_id else None
    if not department or department.tenant_id != tenant_id:
        raise AppError("INVALID_DEPARTMENT", "请选择本团队中的科室", 422)
    return department


async def manager_lock(session, actor):
    # All membership writes take the tenant lock first, including last-admin checks.
    await session.scalar(select(Tenant.id).where(Tenant.id == actor.tenant_id).with_for_update())
    user = await session.get(User, actor.id)
    if not user or user.status != "active" or user.tenant_id != actor.tenant_id or user.role != "owner":
        raise AppError("PERMISSION_DENIED", "仅团队管理员可管理科室和成员", 403)


@router.patch("/users/me/execution-mode")
async def execution_mode(data: ExecutionModeInput, request: Request, actor: CurrentActor):
    require_scope(actor, "tasks:draft")
    svc = request.app.state.services
    async with svc.sessions() as session, session.begin():
        user = await session.scalar(select(User).where(User.id == actor.id).with_for_update())
        if not user or user.status != "active":
            raise AppError("PERMISSION_DENIED", "账户已停用", 403)
        require_scope(Actor(user.id, user.tenant_id, user.role, user.department_id), "tasks:draft")
        user.execution_mode = data.execution_mode
        audit(session, actor, "user.execution_mode", "user", actor.id, data.execution_mode, trace_id())
    return {"execution_mode": data.execution_mode}


@router.get("/organization")
async def organization(request: Request, actor: CurrentActor):
    require_scope(actor, "organization:manage")
    svc = request.app.state.services
    async with svc.sessions() as session:
        departments = (
            await session.scalars(
                select(Department)
                .where(Department.tenant_id == actor.tenant_id)
                .order_by(Department.name)
                .limit(200)
            )
        ).all()
        members = (
            await session.scalars(
                select(User).where(User.tenant_id == actor.tenant_id).order_by(User.created_at).limit(500)
            )
        ).all()
    return {
        "account": await profile(svc, actor),
        "departments": [{"id": d.id, "name": d.name} for d in departments],
        "members": [
            {
                "id": u.id,
                "email": u.email,
                "display_name": u.display_name,
                "role": u.role,
                "department_id": u.department_id,
                "status": u.status,
            }
            for u in members
        ],
    }


@router.patch("/organization")
async def update_organization(data: TeamInput, request: Request, actor: CurrentActor):
    require_scope(actor, "organization:manage")
    svc = request.app.state.services
    try:
        async with svc.sessions() as session, session.begin():
            await manager_lock(session, actor)
            tenant = await session.get(Tenant, actor.tenant_id)
            tenant.name, tenant.login_code = data.name, data.login_code
            await session.flush()
            audit(session, actor, "organization.update", "tenant", tenant.id, "updated", trace_id())
    except IntegrityError:
        raise AppError("TEAM_CODE_TAKEN", "团队登录代号已被使用，请换一个", 409) from None
    return await profile(svc, actor)


@router.post("/organization/departments", status_code=201)
async def create_department(data: DepartmentInput, request: Request, actor: CurrentActor):
    require_scope(actor, "organization:manage")
    try:
        async with request.app.state.services.sessions() as session, session.begin():
            await manager_lock(session, actor)
            department = Department(id=new_id("dept"), tenant_id=actor.tenant_id, name=data.name)
            session.add(department)
            await session.flush()
            audit(session, actor, "department.create", "department", department.id, "created", trace_id())
            result = {"id": department.id, "name": department.name}
    except IntegrityError:
        raise AppError("DEPARTMENT_EXISTS", "该科室名称已存在", 409) from None
    return result


@router.post("/organization/members", status_code=201)
async def create_member(data: MemberInput, request: Request, actor: CurrentActor):
    require_scope(actor, "organization:manage")
    hashed = await hash_password(data.password)
    try:
        async with request.app.state.services.sessions() as session, session.begin():
            await manager_lock(session, actor)
            await department_in_tenant(session, actor.tenant_id, data.department_id)
            user = User(
                id=new_id("user"),
                tenant_id=actor.tenant_id,
                email=data.email.lower(),
                password_hash=hashed,
                display_name=data.display_name,
                department_id=data.department_id,
                role=data.role,
            )
            session.add(user)
            await session.flush()
            audit(session, actor, "member.create", "user", user.id, data.role, trace_id())
            result = {"id": user.id, "role": user.role, "department_id": user.department_id}
    except IntegrityError:
        raise AppError("MEMBER_EXISTS", "该邮箱已在本团队注册", 409) from None
    return result


@router.patch("/organization/members/{user_id}")
async def update_member(user_id: str, data: MemberUpdate, request: Request, actor: CurrentActor):
    require_scope(actor, "organization:manage")
    async with request.app.state.services.sessions() as session, session.begin():
        await manager_lock(session, actor)
        user = await session.scalar(
            select(User).where(User.id == user_id, User.tenant_id == actor.tenant_id).with_for_update()
        )
        if not user:
            raise not_found("成员")
        if data.department_id:
            await department_in_tenant(session, actor.tenant_id, data.department_id)
        elif data.role != "owner":
            raise AppError("DEPARTMENT_REQUIRED", "医生与审核员必须分配科室", 422)
        if user.id == actor.id and data.status != "active":
            raise AppError("SELF_DISABLE", "不能停用当前登录账户", 409)
        if (
            user.role == "owner"
            and user.status == "active"
            and (data.role != "owner" or data.status != "active")
        ):
            count = await session.scalar(
                select(func.count())
                .select_from(User)
                .where(User.tenant_id == actor.tenant_id, User.role == "owner", User.status == "active")
            )
            if count <= 1:
                raise AppError("LAST_ADMIN", "必须至少保留一名有效管理员", 409)
        user.display_name, user.role = data.display_name, data.role
        user.department_id, user.status = data.department_id, data.status
        if user.role == "reviewer":
            user.execution_mode = "ask_for_approval"
        audit(session, actor, "member.update", "user", user.id, f"{data.role}:{data.status}", trace_id())
    return {"updated": True}


def review_query(actor):
    query = select(SegmentationReview).where(SegmentationReview.tenant_id == actor.tenant_id)
    if actor.role == "owner":
        return query
    if actor.role == "reviewer" and actor.department_id:
        return query.where(SegmentationReview.department_id == actor.department_id)
    return query.where(SegmentationReview.submitter_id == actor.id)


async def require_review(session, actor, review_id, lock=False):
    query = review_query(actor).where(SegmentationReview.id == review_id)
    if lock:
        query = query.with_for_update()
    row = (await session.scalars(query)).first()
    if not row:
        raise not_found("复核记录")
    return row


def review_dict(row):
    return {
        key: getattr(row, key)
        for key in (
            "id",
            "job_id",
            "department_id",
            "submitter_id",
            "source_asset_id",
            "result_asset_id",
            "status",
            "reviewer_id",
            "comment",
            "created_at",
            "decided_at",
        )
    }


async def review_assets(svc, actor, review_id, session=None, lock=False):
    if session is None:
        async with svc.sessions() as connection:
            return await review_assets(svc, actor, review_id, session=connection, lock=lock)
    review = await require_review(session, actor, review_id, lock=lock)
    query = select(ToolJob).where(ToolJob.id == review.job_id)
    job = await session.scalar(query.with_for_update() if lock else query)
    if (
        not job
        or job.tenant_id != actor.tenant_id
        or job.user_id != review.submitter_id
        or job.status != "succeeded"
        or job.result_json.get("asset_id") != review.result_asset_id
        or job.result_json.get("source_asset_id") != review.source_asset_id
    ):
        raise not_found("有效分割结果")
    values = {}
    for asset_id in sorted((review.source_asset_id, review.result_asset_id)):
        query = select(Asset).where(Asset.id == asset_id)
        asset = await session.scalar(query.with_for_update() if lock else query)
        if (
            not asset
            or asset.tenant_id != actor.tenant_id
            or asset.owner_user_id != review.submitter_id
            or asset.status != "ready"
        ):
            raise not_found("文件")
        meta = svc.imaging.metadata(asset)
        if meta.get("expires_at") and datetime.fromisoformat(meta["expires_at"]) <= now():
            raise AppError("ASSET_EXPIRED", "复核影像已超过保留期限", 410)
        values[asset_id] = (asset, meta)
    source = values[review.source_asset_id][1]
    mask = values[review.result_asset_id][1]
    if (
        source.get("kind") != "volume"
        or mask.get("kind") != "mask"
        or mask.get("source_asset_id") != review.source_asset_id
    ):
        raise not_found("分割结果")
    return review, values


@router.post("/jobs/{job_id}/review", status_code=201)
async def submit_review(job_id: str, request: Request, actor: CurrentActor):
    require_scope(actor, "reviews:submit")
    svc = request.app.state.services
    from pharma_agent.imaging_api import ready_asset

    async with svc.sessions() as session, session.begin():
        user = await session.scalar(select(User).where(User.id == actor.id).with_for_update())
        if not user or user.status != "active":
            raise AppError("PERMISSION_DENIED", "账户已停用", 403)
        require_scope(Actor(user.id, user.tenant_id, user.role, user.department_id), "reviews:submit")
        await department_in_tenant(session, actor.tenant_id, user.department_id)
        job = await require_owned(svc, ToolJob, actor, job_id, session, lock=True)
        existing = (
            await session.scalars(select(SegmentationReview).where(SegmentationReview.job_id == job_id))
        ).first()
        if existing:
            return review_dict(existing)
        if (
            job.tool_name != "segment_medical_volume"
            or job.status != "succeeded"
            or not job.result_json.get("asset_id")
            or not job.result_json.get("source_asset_id")
        ):
            raise AppError("RESULT_NOT_READY", "只能提交已完成的真实分割结果", 409)
        source_id, mask_id = job.result_json["source_asset_id"], job.result_json["asset_id"]
        for asset_id in sorted((source_id, mask_id)):
            await require_owned(svc, Asset, actor, asset_id, session, lock=True)
        await ready_asset(svc, actor, source_id)
        _, mask = await ready_asset(svc, actor, mask_id)
        if mask.get("kind") != "mask" or mask.get("source_asset_id") != source_id:
            raise AppError("MASK_SPACE_MISMATCH", "结果与源影像不对应", 409)
        row = SegmentationReview(
            id=new_id("review"),
            tenant_id=actor.tenant_id,
            job_id=job.id,
            department_id=user.department_id,
            submitter_id=actor.id,
            source_asset_id=source_id,
            result_asset_id=mask_id,
        )
        session.add(row)
        await session.flush()
        audit(session, actor, "review.submit", "review", row.id, "pending", trace_id())
        return review_dict(row)


@router.get("/reviews")
async def list_reviews(request: Request, actor: CurrentActor, limit: int = Query(50, ge=1, le=100)):
    async with request.app.state.services.sessions() as session:
        rows = (
            await session.scalars(
                review_query(actor).order_by(SegmentationReview.created_at.desc()).limit(limit)
            )
        ).all()
        return [review_dict(row) for row in rows]


@router.post("/reviews/{review_id}/decision")
async def decide_review(review_id: str, data: ReviewDecision, request: Request, actor: CurrentActor):
    require_scope(actor, "reviews:decide")
    svc = request.app.state.services
    async with svc.sessions() as session, session.begin():
        user = await session.scalar(select(User).where(User.id == actor.id).with_for_update())
        if not user or user.status != "active":
            raise AppError("PERMISSION_DENIED", "账户已停用", 403)
        current = Actor(user.id, user.tenant_id, user.role, user.department_id)
        require_scope(current, "reviews:decide")
        row, _ = await review_assets(svc, current, review_id, session=session, lock=True)
        if row.submitter_id == actor.id:
            raise AppError("SELF_REVIEW", "不能复核自己提交的结果", 403)
        if row.status != "pending":
            if row.reviewer_id == actor.id and row.status == data.decision and row.comment == data.comment:
                return review_dict(row)
            raise AppError("REVIEW_DECIDED", "此结果已有复核结论，请刷新查看", 409)
        row.status, row.comment = data.decision, data.comment
        row.reviewer_id, row.decided_at = actor.id, now()
        audit(session, actor, "review.decision", "review", row.id, data.decision, trace_id())
        return review_dict(row)


async def accessible_review_asset(request, actor, review_id, asset_id):
    svc = request.app.state.services
    row, values = await review_assets(svc, actor, review_id)
    if asset_id not in values:
        raise not_found("文件")
    return svc, row, values, values[asset_id]


@router.get("/reviews/{review_id}/assets/{asset_id}/metadata")
async def review_metadata(review_id: str, asset_id: str, request: Request, actor: CurrentActor):
    _, _, _, (_, meta) = await accessible_review_asset(request, actor, review_id, asset_id)
    return meta


@router.get("/reviews/{review_id}/assets/{asset_id}/slice")
async def review_slice(
    review_id: str,
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
    svc, row, values, (_, meta) = await accessible_review_asset(request, actor, review_id, asset_id)
    if asset_id != row.source_asset_id or meta.get("kind") != "volume":
        raise AppError("INVALID_REQUEST", "请查看复核记录的源影像", 422)
    if mask_id and mask_id != row.result_asset_id:
        raise not_found("分割结果")
    try:
        ids = [int(value) for value in labels.split(",")] if labels is not None else None
        if ids is not None and len(ids) > 256:
            raise ValueError
    except ValueError:
        raise AppError("INVALID_REQUEST", "标签编号无效", 422) from None
    content = await asyncio.to_thread(
        svc.imaging.slice, asset_id, plane, index, window, level, mask_id, opacity, ids
    )
    return Response(content, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/reviews/{review_id}/assets/{asset_id}/surface")
async def review_surface(review_id: str, asset_id: str, request: Request, actor: CurrentActor):
    from pharma_agent.surfaces import mask_surface

    svc, row, _, (_, meta) = await accessible_review_asset(request, actor, review_id, asset_id)
    if asset_id != row.result_asset_id:
        raise AppError("NOT_SEGMENTATION", "三维表面需要分割结果", 422)
    return await asyncio.to_thread(mask_surface, svc.imaging.load(asset_id, display=False), meta["labels"])


@router.get("/reviews/{review_id}/assets/{asset_id}/download")
async def review_download(review_id: str, asset_id: str, request: Request, actor: CurrentActor):
    svc, _, _, (asset, meta) = await accessible_review_asset(request, actor, review_id, asset_id)
    path = svc.storage.path(asset.object_key)
    if not path.is_file():
        raise not_found("文件")
    async with svc.sessions() as session, session.begin():
        audit(session, actor, "review.asset.download", "asset", asset.id, "succeeded", trace_id())
    return FileResponse(
        path,
        filename=meta["name"],
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )
