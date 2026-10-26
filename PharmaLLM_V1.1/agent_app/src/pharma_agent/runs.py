import asyncio
import contextlib
from datetime import datetime, timedelta
import hashlib

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opentelemetry.context import attach, detach
from opentelemetry.propagate import extract, inject
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from sqlalchemy import func, select, update

from pharma_agent.db import (
    AgentRun,
    Asset,
    Conversation,
    Message,
    RunEvent,
    ToolJob,
    User,
    audit,
    new_id,
    now,
)
from pharma_agent.errors import AppError
from pharma_agent.graph import RunContext, build_graph
from pharma_agent.observability import digest, log, request_context, span, trace_id
from pharma_agent.security import Actor, require_owned

TERMINAL = {"completed", "failed", "cancelled"}


class RunService:
    def __init__(self, services):
        self.services = services
        self.tasks = set()
        self.worker_id = new_id("api")
        self.stopping = False

    def task_done(self, task):
        self.tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error:
                # Consume task exceptions so asyncio cannot print raw database errors/parameters.
                log(
                    "agent.task_failed",
                    error_code="RUN_PERSISTENCE_UNAVAILABLE",
                    exception_type=type(error).__name__,
                )

    async def create(
        self,
        actor,
        conversation_id,
        message,
        key,
        request_id,
        asset_ids=None,
        model_alias=None,
        segmentation_model_id=None,
        execution_mode=None,
    ):
        services = self.services
        asset_ids = list(dict.fromkeys(asset_ids or []))
        snapshot = services.models.snapshot(model_alias)
        if segmentation_model_id and segmentation_model_id not in services.segmentation.models:
            raise AppError("SEGMENTATION_MODEL_NOT_ALLOWED", "分割模型不在服务端允许列表中", 422)
        fingerprint = digest(
            {
                "message": message,
                **({"asset_ids": asset_ids} if asset_ids else {}),
                **({"model_alias": model_alias} if model_alias else {}),
                **({"segmentation_model_id": segmentation_model_id} if segmentation_model_id else {}),
                **({"execution_mode": execution_mode} if execution_mode else {}),
            }
        )
        async with services.sessions() as session, session.begin():
            # Consistent user->conversation lock order serializes quotas and idempotency checks.
            current_user = await session.scalar(
                select(User)
                .where(User.id == actor.id, User.tenant_id == actor.tenant_id)
                .with_for_update()
            )
            if not current_user or current_user.status != "active":
                raise AppError("PERMISSION_DENIED", "账户已停用", 403)
            mode = execution_mode or current_user.execution_mode or "ask_for_approval"
            if mode not in {"ask_for_approval", "full_access"}:
                raise AppError("INVALID_REQUEST", "执行模式无效", 422)
            if mode == "full_access" and current_user.role not in {"owner", "doctor", "member"}:
                raise AppError("PERMISSION_DENIED", "当前角色不能自动执行分割", 403)
            conversation = await require_owned(
                services, Conversation, actor, conversation_id, session, lock=True
            )
            previous = (
                await session.scalars(
                    select(AgentRun).where(
                        AgentRun.tenant_id == actor.tenant_id,
                        AgentRun.user_id == actor.id,
                        AgentRun.conversation_id == conversation_id,
                        AgentRun.idempotency_key == key,
                    )
                )
            ).first()
            if previous:
                if previous.request_hash != fingerprint:
                    raise AppError("IDEMPOTENCY_CONFLICT", "同一幂等键不能用于不同内容", 409)
                return previous
            asset_metadata = []
            for asset_id in sorted(asset_ids):
                asset = await require_owned(services, Asset, actor, asset_id, session, lock=True)
                meta = services.imaging.metadata(asset)
                if asset.status != "ready":
                    raise AppError("ASSET_UNAVAILABLE", "附件已删除或不可使用", 422)
                if meta.get("expires_at") and datetime.fromisoformat(meta["expires_at"]) <= now():
                    raise AppError("ASSET_EXPIRED", "附件已过保留期限，请重新上传", 410)
                if meta.get("conversation_id") not in {None, conversation_id}:
                    raise AppError("ASSET_CONVERSATION_MISMATCH", "附件属于另一会话，请重新上传", 422)
                asset_metadata.append((asset_id, meta))
            if conversation.active_run_id:
                raise AppError("CONVERSATION_BUSY", "当前会话正在回答，请等待完成", 409, True)
            count = await session.scalar(
                select(func.count())
                .select_from(AgentRun)
                .where(
                    AgentRun.tenant_id == actor.tenant_id,
                    AgentRun.user_id == actor.id,
                    AgentRun.status.in_(["queued", "running", "cancelling"]),
                )
            )
            if count >= services.settings.max_pending_runs_per_user:
                raise AppError("RATE_LIMITED", "当前用户待处理请求已达上限", 429, True)
            for asset_id, meta in asset_metadata:
                meta["conversation_id"] = conversation_id
                services.imaging.save_meta(asset_id, meta)
            carrier = {}
            inject(carrier)
            run = AgentRun(
                id=new_id("run"),
                tenant_id=actor.tenant_id,
                user_id=actor.id,
                conversation_id=conversation_id,
                idempotency_key=key,
                request_hash=fingerprint,
                request_id=request_id,
                trace_id=trace_id(),
                trace_carrier=carrier,
                status="queued",
                model_snapshot=snapshot,
                graph_version=services.graph_policy["version"],
            )
            run.model_snapshot["asset_ids"] = asset_ids
            run.model_snapshot["segmentation_model_id"] = segmentation_model_id
            run.model_snapshot["execution_mode"] = mode
            run.model_snapshot["graph_policy"] = dict(services.graph_policy)
            run.model_snapshot["tool_config"] = dict(services.tools.registry.config)
            session.add(run)
            await session.flush()
            session.add(
                Message(
                    id=new_id("msg"),
                    conversation_id=conversation_id,
                    run_id=run.id,
                    role="user",
                    content=message,
                )
            )
            conversation.active_run_id = run.id
            conversation.version += 1
            session.add(RunEvent(run_id=run.id, kind="status", payload={"run_id": run.id, "phase": "queued"}))
            audit(session, actor, "run.create", "run", run.id, "queued", run.trace_id)
            return run

    async def emit(self, run_id, kind, payload):
        async with self.services.sessions() as session, session.begin():
            # The lease owner alone may append live events.
            run = await session.get(AgentRun, run_id, with_for_update=True)
            if run.status != "running" or run.worker_id != self.worker_id:
                raise AppError("RUN_CANCELLED", "本次运行已停止", 409)
            session.add(RunEvent(run_id=run_id, kind=kind, payload={"run_id": run_id, **payload}))

    async def heartbeat(self, run_id, owner_task):
        while True:
            await asyncio.sleep(1)
            async with self.services.sessions() as session, session.begin():
                run = await session.get(AgentRun, run_id)
                if run is None or run.worker_id != self.worker_id or run.status in TERMINAL:
                    owner_task.cancel()
                    return
                if run.status == "cancelling":
                    owner_task.cancel()
                    return
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.worker_id == self.worker_id,
                        AgentRun.status == "running",
                    )
                    .values(lease_until=now() + timedelta(seconds=self.services.settings.run_lease_seconds))
                )
                if not result.rowcount:
                    owner_task.cancel()
                    return

    async def complete(self, run_id, state=None, error=None, usage=None):
        async with self.services.sessions() as session, session.begin():
            run = await session.get(AgentRun, run_id, with_for_update=True)
            if run.status not in {"running", "cancelling"} or run.worker_id != self.worker_id:
                return
            run.finished_at, run.lease_until = now(), None
            run.usage = {"calls": usage or []}
            if run.status == "cancelling":
                error = AppError("RUN_CANCELLED", "本次运行已取消", 409)
            if error:
                run.status, run.error_code = (
                    ("cancelled" if error.code == "RUN_CANCELLED" else "failed"),
                    error.code,
                )
                session.add(
                    RunEvent(
                        run_id=run_id,
                        kind="error",
                        payload={
                            "run_id": run_id,
                            "code": error.code,
                            "message": error.message,
                            "retryable": error.retryable,
                            "trace_id": run.trace_id,
                        },
                    )
                )
            else:
                run.status, run.final_answer = "completed", state["final_answer"]
                run.citations = state.get("retrieved_docs", [])
                run.error_code = state.get("error_code")
                message_id = new_id("msg")
                session.add(
                    Message(
                        id=message_id,
                        conversation_id=run.conversation_id,
                        run_id=run.id,
                        role="assistant",
                        content=run.final_answer,
                    )
                )
                session.add(
                    RunEvent(
                        run_id=run_id,
                        kind="completed",
                        payload={
                            "run_id": run_id,
                            "message_id": message_id,
                            "citations": run.citations,
                            "trace_id": run.trace_id,
                            "warning_code": run.error_code,
                        },
                    )
                )
            await session.execute(
                update(Conversation)
                .where(Conversation.id == run.conversation_id, Conversation.active_run_id == run_id)
                .values(active_run_id=None)
            )

    async def execute(self, run):
        services = self.services
        parent_token = attach(extract(run.trace_carrier))
        context_token = request_context.set(
            {
                "request_id": run.request_id,
                "run_id": run.id,
                "tenant_id": run.tenant_id,
                "user_id_hash": digest(run.user_id)[7:31],
            }
        )
        heartbeat = asyncio.create_task(self.heartbeat(run.id, asyncio.current_task()))
        ctx = None
        try:
            with span("agent.run", run_id=run.id, graph_version=run.graph_version):
                if run.graph_version != services.graph_policy["version"]:
                    raise AppError("GRAPH_VERSION_CHANGED", "图版本已变更，请重新提交请求", 409, True)
                if run.model_snapshot.get("graph_policy") != services.graph_policy:
                    raise AppError("GRAPH_VERSION_CHANGED", "图配置已变更，请重新提交请求", 409, True)
                if run.model_snapshot.get("tool_config") != services.tools.registry.config:
                    raise AppError("TOOL_CONFIG_CHANGED", "工具配置已变更，请重新提交请求", 409, True)
                async with services.sessions() as session:
                    user = await session.get(User, run.user_id)
                    conversation = await session.get(Conversation, run.conversation_id)
                if not user or user.status != "active" or conversation.deleted_at:
                    raise AppError("PERMISSION_DENIED", "用户或会话已停用", 403)
                ctx = RunContext(
                    Actor(user.id, user.tenant_id, user.role, user.department_id),
                    run.request_id,
                    run.id,
                    run.conversation_id,
                    run.model_snapshot,
                    services,
                    lambda kind, payload: self.emit(run.id, kind, payload),
                )
                async with asyncio.timeout(services.settings.run_timeout_seconds):
                    # Fence checkpoint writes using the SAME connection holding a session advisory lock.
                    # A stale lease cannot let two processes write the same thread concurrently.
                    key = int.from_bytes(
                        hashlib.sha256(conversation.thread_id.encode()).digest()[:8], "big", signed=True
                    )
                    async with await AsyncConnection.connect(
                        services.settings.checkpoint_url,
                        autocommit=True,
                        row_factory=dict_row,
                        prepare_threshold=0,
                    ) as cp_connection:
                        await cp_connection.execute("SELECT pg_advisory_lock(%s)", (key,))
                        graph = build_graph(services, AsyncPostgresSaver(cp_connection))
                        state = await graph.ainvoke(
                            {"status": "running"},
                            {
                                "configurable": {"thread_id": conversation.thread_id},
                                "recursion_limit": services.graph_policy["recursion_limit"],
                            },
                            context=ctx,
                            durability="sync",
                        )
                        # Closing this dedicated connection releases the advisory lock even on cancellation.
                await self.complete(run.id, state=state, usage=ctx.model_usage)
        except asyncio.CancelledError:
            await self.complete(
                run.id, error=AppError("RUN_INTERRUPTED", "服务停止，本轮已中断，请重新提交", 503, True)
            )
            raise
        except TimeoutError:
            await self.complete(run.id, error=AppError("RUN_TIMEOUT", "本轮执行超过时间预算", 504, True))
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, AppError)
                else AppError("INTERNAL_ERROR", "执行失败，请提供 trace_id 排查", 500)
            )
            log("agent.failed", error_code=error.code, exception_type=type(exc).__name__)
            await self.complete(run.id, error=error, usage=ctx.model_usage if ctx else [])
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await heartbeat
            request_context.reset(context_token)
            detach(parent_token)

    async def recover_expired(self):
        async with self.services.sessions() as session, session.begin():
            rows = (
                await session.scalars(
                    select(AgentRun)
                    .where(AgentRun.status.in_(["running", "cancelling"]), AgentRun.lease_until < now())
                    .with_for_update(skip_locked=True)
                    .limit(100)
                )
            ).all()
            for run in rows:
                run.status, run.error_code, run.finished_at = "failed", "RUN_INTERRUPTED", now()
                session.add(
                    RunEvent(
                        run_id=run.id,
                        kind="error",
                        payload={
                            "run_id": run.id,
                            "code": "RUN_INTERRUPTED",
                            "message": "运行进程中断，请重新提交；不会自动重放写操作",
                            "retryable": True,
                            "trace_id": run.trace_id,
                        },
                    )
                )
                await session.execute(
                    update(Conversation)
                    .where(Conversation.id == run.conversation_id, Conversation.active_run_id == run.id)
                    .values(active_run_id=None)
                )
            jobs = (
                await session.scalars(
                    select(ToolJob)
                    .where(ToolJob.status == "running", ToolJob.lease_until < now())
                    .with_for_update(skip_locked=True)
                    .limit(100)
                )
            ).all()
            for job in jobs:
                job.status, job.error_code, job.finished_at = "failed", "WORKER_INTERRUPTED", now()

    async def dispatch(self):
        while not self.stopping:
            try:
                await self.recover_expired()
                while len(self.tasks) < self.services.settings.max_active_runs:
                    async with self.services.sessions() as session, session.begin():
                        run = (
                            await session.scalars(
                                select(AgentRun)
                                .where(AgentRun.status == "queued")
                                .order_by(AgentRun.created_at)
                                .with_for_update(skip_locked=True)
                                .limit(1)
                            )
                        ).first()
                        if run is None:
                            break
                        run.status, run.worker_id, run.started_at = "running", self.worker_id, now()
                        run.lease_until = now() + timedelta(seconds=self.services.settings.run_lease_seconds)
                    task = asyncio.create_task(self.execute(run))
                    self.tasks.add(task)
                    task.add_done_callback(self.task_done)
                await self.services.queue.dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log("dispatcher.failed", error_code="DISPATCH_UNAVAILABLE", exception_type=type(exc).__name__)
            await asyncio.sleep(0.5)

    async def shutdown(self):
        self.stopping = True
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
