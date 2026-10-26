import asyncio
import json
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from pharma_agent.db import (
    AgentRun,
    Asset,
    Conversation,
    Document,
    DocumentChunk,
    Feedback,
    Message,
    RunEvent,
    Tenant,
    ToolJob,
    User,
    UserMemory,
    audit,
    new_id,
    now,
)
from pharma_agent.errors import AppError, not_found
from pharma_agent.observability import log, trace_id
from pharma_agent.runs import TERMINAL
from pharma_agent.schemas import (
    ConfirmInput,
    ConversationInput,
    DocumentInput,
    DraftInput,
    FeedbackInput,
    LoginInput,
    MemoryInput,
    RegisterInput,
    RunInput,
)
from pharma_agent.security import (
    Actor,
    auth_rate_limit,
    authenticate,
    hash_password,
    issue_token,
    owned_query,
    rate_limit,
    require_owned,
    verify_password,
)
from pharma_agent.tools import create_draft, job_dict
from pharma_agent.hospital import profile

router = APIRouter(prefix="/api/v1")
CurrentActor = Annotated[Actor, Depends(authenticate)]


def services(request):
    return request.app.state.services


def idem_key(value):
    if value is None:
        return new_id("client")
    if not 1 <= len(value) <= 128 or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise AppError("INVALID_REQUEST", "Idempotency-Key 需为 1–128 个可见 ASCII 字符", 422)
    return value


def run_dict(run):
    return {
        "run_id": run.id,
        "conversation_id": run.conversation_id,
        "status": run.status,
        "trace_id": run.trace_id,
        "error_code": run.error_code,
        "answer": run.final_answer,
        "citations": run.citations,
        "usage": run.usage,
        "graph_version": run.graph_version,
        "created_at": run.created_at,
        "events_url": f"/api/v1/runs/{run.id}/events",
        "asset_ids": run.model_snapshot.get("asset_ids", []),
        "model_alias": run.model_snapshot["roles"]["medical_answer"],
    }


@router.post("/auth/register", status_code=201)
async def register(data: RegisterInput, request: Request):
    svc = services(request)
    if not svc.settings.allow_registration:
        raise AppError("PERMISSION_DENIED", "注册已关闭", 403)
    await auth_rate_limit(svc, request.client.host if request.client else "local")
    hashed = await hash_password(data.password)
    # Registration always creates a fresh tenant; knowing a tenant ID never grants membership.
    try:
        async with svc.sessions() as session, session.begin():
            tenant = Tenant(id=new_id("tenant"), name=data.tenant_name,
                            login_code=data.tenant_code or "team-" + secrets.token_hex(4))
            session.add(tenant)
            await session.flush()
            user = User(
                id=new_id("user"), tenant_id=tenant.id, email=data.email.lower(),
                password_hash=hashed, role="owner",
            )
            session.add(user)
            await session.flush()
            actor = Actor(user.id, tenant.id, user.role)
            audit(session, actor, "auth.register", "user", user.id, "succeeded", trace_id())
    except IntegrityError:
        raise AppError("TEAM_CODE_TAKEN", "团队登录代号已被使用，请换一个", 409) from None
    return {
        **await profile(svc, actor),
        "token_type": "bearer",
        "access_token": issue_token(svc.settings, actor),
    }


@router.post("/auth/login")
async def login(data: LoginInput, request: Request):
    svc = services(request)
    await auth_rate_limit(svc, request.client.host if request.client else "local")
    async with svc.sessions() as session:
        user = (
            await session.scalars(
                select(User)
                .join(Tenant, Tenant.id == User.tenant_id)
                .where(
                    or_(Tenant.id == data.tenant_id, Tenant.login_code == data.tenant_id.lower()),
                    User.email == data.email.lower(),
                    Tenant.status == "active",
                )
            )
        ).first()
    valid = await verify_password(data.password, user.password_hash if user else None)
    if not valid or user.status != "active":
        async with svc.sessions() as session, session.begin():
            audit(session, None, "auth.login", "auth", "login", "denied", trace_id())
        raise AppError("AUTH_REQUIRED", "团队、邮箱或密码错误", 401)
    actor = Actor(user.id, user.tenant_id, user.role, user.department_id)
    async with svc.sessions() as session, session.begin():
        audit(session, actor, "auth.login", "user", user.id, "succeeded", trace_id())
    return {
        "access_token": issue_token(svc.settings, actor),
        "token_type": "bearer",
        **await profile(svc, actor),
    }


@router.get("/users/me")
async def me(request: Request, actor: CurrentActor):
    return await profile(services(request), actor)


@router.post("/conversations", status_code=201)
async def create_conversation(data: ConversationInput, request: Request, actor: CurrentActor):
    async with services(request).sessions() as session, session.begin():
        item = Conversation(
            id=new_id("conv"),
            tenant_id=actor.tenant_id,
            owner_user_id=actor.id,
            title=data.title,
            thread_id=new_id("thread"),
        )
        session.add(item)
        return {"id": item.id, "title": item.title}


@router.get("/conversations")
async def list_conversations(
    request: Request, actor: CurrentActor, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)
):
    async with services(request).sessions() as session:
        rows = (
            await session.scalars(
                select(Conversation)
                .where(
                    Conversation.tenant_id == actor.tenant_id,
                    Conversation.owner_user_id == actor.id,
                    Conversation.deleted_at.is_(None),
                )
                .order_by(Conversation.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
        ).all()
        return [
            {
                "id": row.id,
                "title": row.title,
                "active_run_id": row.active_run_id,
                "created_at": row.created_at,
            }
            for row in rows
        ]


@router.get("/conversations/{conversation_id}/messages")
async def list_messages(
    conversation_id: str,
    request: Request,
    actor: CurrentActor,
    limit: int = Query(100, ge=1, le=200),
    before: str | None = None,
):
    svc = services(request)
    await require_owned(svc, Conversation, actor, conversation_id)
    async with svc.sessions() as session:
        query = select(Message).where(Message.conversation_id == conversation_id)
        if before:
            boundary = (
                await session.scalars(
                    select(Message).where(Message.id == before, Message.conversation_id == conversation_id)
                )
            ).first()
            if boundary is None:
                raise not_found()
            query = query.where(Message.created_at < boundary.created_at)
        rows = (
            await session.scalars(query.order_by(Message.created_at.desc(), Message.id.desc()).limit(limit))
        ).all()
        run_ids = {row.run_id for row in rows}
        runs = (
            (await session.scalars(select(AgentRun).where(AgentRun.id.in_(run_ids)))).all() if run_ids else []
        )
        attachments = {r.id: r.model_snapshot.get("asset_ids", []) for r in runs}
        return [
            {
                "asset_ids": attachments.get(row.run_id, []),
                "id": row.id,
                "run_id": row.run_id,
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at,
            }
            for row in reversed(rows)
        ]


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request, actor: CurrentActor):
    svc = services(request)
    async with svc.sessions() as session, session.begin():
        row = await require_owned(svc, Conversation, actor, conversation_id, session, lock=True)
        if row.active_run_id:
            raise AppError("CONVERSATION_BUSY", "请先停止运行，再删除会话", 409)
        row.deleted_at = now()
        audit(session, actor, "conversation.soft_delete", "conversation", row.id, "succeeded", trace_id())
    return {"deleted": True, "retention": "operator_purge_required"}


@router.post("/conversations/{conversation_id}/runs", status_code=202)
async def create_run(
    conversation_id: str,
    data: RunInput,
    request: Request,
    actor: CurrentActor,
    idempotency_key: str | None = Header(None),
):
    svc = services(request)
    await rate_limit(svc, actor)
    run = await svc.runs.create(
        actor,
        conversation_id,
        data.message,
        idem_key(idempotency_key),
        request.state.request_id,
        asset_ids=data.asset_ids,
        model_alias=data.model_alias,
        segmentation_model_id=data.segmentation_model_id,
        execution_mode=data.execution_mode,
    )
    return run_dict(run)


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request, actor: CurrentActor):
    svc = services(request)
    run = await require_owned(svc, AgentRun, actor, run_id)
    await require_owned(svc, Conversation, actor, run.conversation_id)
    return run_dict(run)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request, actor: CurrentActor):
    svc = services(request)
    async with svc.sessions() as session, session.begin():
        run = await require_owned(svc, AgentRun, actor, run_id, session, lock=True)
        await require_owned(svc, Conversation, actor, run.conversation_id, session)
        if run.status not in TERMINAL:
            if run.status == "queued":
                run.status, run.error_code, run.finished_at = "cancelled", "RUN_CANCELLED", now()
                conv = await session.get(Conversation, run.conversation_id, with_for_update=True)
                if conv.active_run_id == run.id:
                    conv.active_run_id = None
                session.add(
                    RunEvent(
                        run_id=run_id,
                        kind="error",
                        payload={
                            "run_id": run_id,
                            "code": "RUN_CANCELLED",
                            "message": "运行已取消",
                            "retryable": False,
                            "trace_id": run.trace_id,
                        },
                    )
                )
            else:
                # Keep conversation occupied until the executing graph has stopped writing checkpoints.
                run.status = "cancelling"
                session.add(
                    RunEvent(run_id=run_id, kind="status", payload={"run_id": run_id, "phase": "cancelling"})
                )
            audit(session, actor, "run.cancel", "run", run_id, "requested", trace_id())
        return run_dict(run)


@router.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    actor: CurrentActor,
    last_event_id: str | None = Header(None),
    after: int = Query(0, ge=0),
):
    svc = services(request)
    run = await require_owned(svc, AgentRun, actor, run_id)
    await require_owned(svc, Conversation, actor, run.conversation_id)
    try:
        cursor = max(after, int(last_event_id or 0))
        if cursor < 0:
            raise ValueError()
    except ValueError:
        raise AppError("INVALID_REQUEST", "Last-Event-ID 必须为非负整数", 422) from None

    async def poll_events():
        nonlocal cursor
        ticks = 0
        while not await request.is_disconnected():
            async with svc.sessions() as session:
                # Scope every reconnect/poll as well as the initial subscription.
                run = (
                    await session.scalars(
                        owned_query(AgentRun, actor, run_id)
                        .join(Conversation, Conversation.id == AgentRun.conversation_id)
                        .where(Conversation.deleted_at.is_(None))
                    )
                ).first()
                if run is None:
                    return
                events = (
                    await session.scalars(
                        select(RunEvent)
                        .where(RunEvent.run_id == run_id, RunEvent.id > cursor)
                        .order_by(RunEvent.id)
                        .limit(200)
                    )
                ).all()
            for event in events:
                cursor = event.id
                yield f"id: {event.id}\nevent: {event.kind}\ndata: {json.dumps(event.payload, ensure_ascii=False)}\n\n"
            if run.status in TERMINAL and len(events) < 200:
                return
            ticks += 1
            if ticks % 40 == 0:
                yield ": heartbeat\n\n"
            await asyncio.sleep(0.25)

    async def stream():
        try:
            async for event in poll_events():
                yield event
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log(
                "sse.failed",
                run_id=run_id,
                error_code="EVENT_STREAM_UNAVAILABLE",
                exception_type=type(exc).__name__,
            )
            payload = {
                "run_id": run_id,
                "code": "EVENT_STREAM_UNAVAILABLE",
                "message": "事件连接中断，请使用 Last-Event-ID 重连",
                "retryable": True,
                "trace_id": run.trace_id,
            }
            yield "event: error\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/memories")
async def get_memories(request: Request, actor: CurrentActor):
    async with services(request).sessions() as session:
        rows = (
            await session.scalars(
                select(UserMemory).where(
                    UserMemory.tenant_id == actor.tenant_id,
                    UserMemory.user_id == actor.id,
                    (UserMemory.expires_at.is_(None)) | (UserMemory.expires_at > now()),
                )
            )
        ).all()
        return [
            {
                "id": row.id,
                "kind": row.kind,
                "value": row.value,
                "version": row.version,
                "consent_at": row.consent_at,
                "expires_at": row.expires_at,
            }
            for row in rows
        ]


@router.post("/memories")
async def set_memory(data: MemoryInput, request: Request, actor: CurrentActor):
    if data.expires_at and data.expires_at <= now():
        raise AppError("INVALID_REQUEST", "偏好过期时间必须晚于当前时间", 422)
    async with services(request).sessions() as session, session.begin():
        await session.scalar(select(User.id).where(User.id == actor.id).with_for_update())
        row = (
            await session.scalars(
                select(UserMemory).where(
                    UserMemory.tenant_id == actor.tenant_id,
                    UserMemory.user_id == actor.id,
                    UserMemory.kind == data.kind,
                )
            )
        ).first()
        if row is None:
            row = UserMemory(
                id=new_id("mem"),
                tenant_id=actor.tenant_id,
                user_id=actor.id,
                kind=data.kind,
                value=data.value,
                expires_at=data.expires_at,
            )
            session.add(row)
        else:
            row.value, row.expires_at, row.consent_at, row.version = (
                data.value,
                data.expires_at,
                now(),
                row.version + 1,
            )
        audit(session, actor, "memory.upsert", "memory", row.id, "succeeded", trace_id())
        return {"id": row.id, "kind": row.kind, "value": row.value}


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request, actor: CurrentActor):
    svc = services(request)
    async with svc.sessions() as session, session.begin():
        row = await require_owned(svc, UserMemory, actor, memory_id, session)
        await session.delete(row)
        audit(session, actor, "memory.delete", "memory", memory_id, "succeeded", trace_id())
    return {"deleted": True}


@router.post("/jobs/drafts", status_code=201)
async def draft_job(
    data: DraftInput, request: Request, actor: CurrentActor, idempotency_key: str | None = Header(None)
):
    await rate_limit(services(request), actor)
    job = await create_draft(services(request), actor, data.title, idem_key(idempotency_key))
    return job_dict(job)


@router.post("/jobs/{job_id}/confirm", status_code=202)
async def confirm_job(job_id: str, data: ConfirmInput, request: Request, actor: CurrentActor):
    job = await services(request).queue.confirm(actor, job_id)
    return {**job_dict(job), "dispatch_state": "pending_or_published", "requires_worker": True}


@router.get("/jobs")
async def list_jobs(request: Request, actor: CurrentActor, limit: int = Query(20, ge=1, le=100)):
    async with services(request).sessions() as session:
        rows = (
            await session.scalars(
                select(ToolJob)
                .where(ToolJob.tenant_id == actor.tenant_id, ToolJob.user_id == actor.id)
                .order_by(ToolJob.created_at.desc())
                .limit(limit)
            )
        ).all()
        return [job_dict(row) for row in rows]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request, actor: CurrentActor):
    return job_dict(await require_owned(services(request), ToolJob, actor, job_id))


@router.post("/assets", status_code=201)
async def upload_asset(request: Request, actor: CurrentActor, file: UploadFile = File(...)):
    # Preserve the original single-file route, using the same bounded real parser.
    from pharma_agent.imaging_api import upload

    return await upload(request, actor, [file], "unknown", None)


@router.get("/assets/{asset_id}/download")
async def download_asset(asset_id: str, request: Request, actor: CurrentActor):
    svc = services(request)
    from pharma_agent.imaging_api import ready_asset

    asset, _ = await ready_asset(svc, actor, asset_id)
    if asset.status != "ready":
        raise not_found("文件")
    path = svc.storage.path(asset.object_key)
    if not path.is_file():
        raise not_found("文件")
    async with svc.sessions() as session, session.begin():
        audit(session, actor, "asset.download", "asset", asset_id, "succeeded", trace_id())
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=svc.imaging.metadata(asset).get("name", asset.object_key),
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"},
    )


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: str, request: Request, actor: CurrentActor):
    svc = services(request)
    async with svc.sessions() as session, session.begin():
        asset = await require_owned(svc, Asset, actor, asset_id, session, lock=True)
        await asyncio.to_thread(svc.storage.remove, asset.object_key)
        await asyncio.to_thread(svc.imaging.purge, asset.id)
        asset.status = "deleted"
        audit(session, actor, "asset.delete", "asset", asset.id, "content_deleted", trace_id())
    return {"deleted": True}


@router.post("/documents", status_code=201)
async def import_document(data: DocumentInput, request: Request, actor: CurrentActor):
    if "documents:write" not in actor.scopes:
        raise AppError("PERMISSION_DENIED", "仅团队管理员可导入资料", 403)
    svc = services(request)
    await rate_limit(svc, actor)
    result = await svc.retrieval.ingest(data, actor.tenant_id)
    async with svc.sessions() as session, session.begin():
        audit(session, actor, "document.ingest", "document", result["document_id"], "succeeded", trace_id())
    return result


@router.get("/references/{chunk_id}")
async def get_reference(chunk_id: str, request: Request, actor: CurrentActor):
    async with services(request).sessions() as session:
        row = (
            await session.execute(
                select(Document, DocumentChunk)
                .join(DocumentChunk, DocumentChunk.document_id == Document.id)
                .where(
                    DocumentChunk.id == chunk_id,
                    ((Document.visibility == "public") & Document.tenant_id.is_(None))
                    | ((Document.visibility == "tenant") & (Document.tenant_id == actor.tenant_id)),
                )
            )
        ).first()
        if row is None:
            raise not_found("资料")
        document, chunk = row
        return {
            "document_id": document.id,
            "chunk_id": chunk.id,
            "title": document.title,
            "source": document.source,
            "source_version": document.source_version,
            "license": document.license,
            "published_at": document.published_at,
            "index_version": chunk.index_version,
            "content": chunk.content,
        }


@router.post("/feedback")
async def feedback(data: FeedbackInput, request: Request, actor: CurrentActor):
    svc = services(request)
    await require_owned(svc, AgentRun, actor, data.run_id)
    async with svc.sessions() as session, session.begin():
        await session.scalar(select(User.id).where(User.id == actor.id).with_for_update())
        row = (
            await session.scalars(
                select(Feedback).where(Feedback.user_id == actor.id, Feedback.run_id == data.run_id)
            )
        ).first()
        if row:
            row.rating, row.category = data.rating, data.category
        else:
            session.add(Feedback(id=new_id("feedback"), user_id=actor.id, **data.model_dump()))
    return {"saved": True}
