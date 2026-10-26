import asyncio
import json
from datetime import timedelta

from celery import Celery
from opentelemetry.context import attach, detach
from opentelemetry.propagate import extract, inject
from sqlalchemy import select

from pharma_agent.db import Asset, TaskOutbox, ToolJob, audit, create_database, new_id, now
from pharma_agent.errors import AppError
from pharma_agent.observability import configure, log, span, trace_id
from pharma_agent.security import Actor, require_owned, require_scope
from pharma_agent.db import User
from pharma_agent.settings import Settings
from pharma_agent.storage import LocalStorage


def make_celery(settings):
    app = Celery("pharma_agent", broker=settings.redis_url.get_secret_value())
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_ignore_result=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_time_limit=120,
        task_soft_time_limit=110,
        broker_connection_timeout=3,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "visibility_timeout": settings.segmentation_timeout_seconds + 300,
            "socket_timeout": 3,
            "socket_connect_timeout": 3,
            "retry_on_timeout": False,
        },
    )
    return app


class QueueService:
    def __init__(self, services):
        self.services = services
        self.celery = make_celery(services.settings) if services.settings.enable_queue else None

    async def confirm(self, actor, job_id, automatic=False):
        require_scope(actor, "tasks:draft")
        if self.celery is None:
            raise AppError("QUEUE_UNAVAILABLE", "异步任务功能未启用", 503, True)
        async with self.services.sessions() as session, session.begin():
            # Recheck the live role, including long-running Graph requests.
            user = await session.scalar(select(User).where(User.id == actor.id).with_for_update())
            if not user or user.status != "active" or user.tenant_id != actor.tenant_id:
                raise AppError("PERMISSION_DENIED", "账户已停用", 403)
            require_scope(Actor(user.id, user.tenant_id, user.role, user.department_id), "tasks:draft")
            job = await require_owned(self.services, ToolJob, actor, job_id, session, lock=True)
            if automatic and (
                job.tool_name != "segment_medical_volume"
                or job.input_json.get("execution_mode") != "full_access"
            ):
                raise AppError("PERMISSION_DENIED", "此任务没有自动执行授权", 403)
            if job.status in {"queued", "running", "succeeded"}:
                return job
            if job.status not in {"draft", "failed"} or job.tool_name not in {
                "mock_report",
                "segment_medical_volume",
            }:
                raise AppError("JOB_STATE_CONFLICT", "当前任务状态不允许确认", 409)
            if job.tool_name == "segment_medical_volume":
                job.input_json = {**job.input_json, "attempt": job.input_json.get("attempt", 0) + 1}
                job.result_json = {"phase": "queued", "labels": job.input_json["model"]["labels"]}
            job.finished_at = None
            carrier = {}
            inject(carrier)
            job.status, job.confirmed_at = "queued", now()
            job.trace_carrier, job.error_code = carrier, None
            job.claim_token, job.lease_until = None, None
            outbox = (await session.scalars(select(TaskOutbox).where(TaskOutbox.job_id == job.id))).first()
            if outbox:
                outbox.published, outbox.attempts, outbox.next_attempt_at = False, 0, now()
            else:
                session.add(TaskOutbox(id=new_id("outbox"), job_id=job.id))
            audit(session, actor, "job.auto_confirm" if automatic else "job.confirm", "job", job.id,
                  "queued_pending_publish", trace_id())
            return job

    async def dispatch_once(self):
        if self.celery is None:
            return
        # One row per transaction. Network publish is bounded; failure keeps a durable retry record.
        async with self.services.sessions() as session, session.begin():
            row = (
                await session.scalars(
                    select(TaskOutbox)
                    .where(
                        TaskOutbox.published.is_(False),
                        TaskOutbox.next_attempt_at <= now(),
                        TaskOutbox.attempts < 5,
                    )
                    .order_by(TaskOutbox.created_at)
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
            ).first()
            if row is None:
                return
            job = await session.get(ToolJob, row.job_id)
            if job.status != "queued":
                row.published = True
                return
            try:
                await asyncio.to_thread(
                    self.celery.send_task,
                    "pharma_agent.execute_segmentation"
                    if job.tool_name == "segment_medical_volume"
                    else "pharma_agent.execute_mock_report",
                    args=[row.job_id],
                    task_id=row.job_id,
                    retry=False,
                )
                row.published = True
                log("queue.published", job_id=row.job_id)
            except Exception:
                row.attempts += 1
                row.next_attempt_at = now() + timedelta(seconds=min(30, 2**row.attempts))
                log(
                    "queue.publish_failed",
                    job_id=row.job_id,
                    attempt=row.attempts,
                    error_code="QUEUE_UNAVAILABLE",
                )
                if row.attempts >= 5:
                    job.status, job.error_code, job.finished_at = "failed", "QUEUE_UNAVAILABLE", now()


async def execute_mock_job(settings, job_id):
    """Idempotent CPU worker; produces synthetic JSON, never a medical segmentation mask."""
    engine, sessions = create_database(settings.database_url)
    claim, carrier, asset_key = new_id("claim"), {}, None
    context_token = None
    try:
        async with sessions() as session, session.begin():
            job = await session.get(ToolJob, job_id, with_for_update=True)
            if job is None or job.status in {"draft", "succeeded", "failed", "cancelled"}:
                return
            if job.status == "running" and job.lease_until and job.lease_until > now():
                return  # Duplicate delivery cannot execute an active job twice.
            if not job.confirmed_at or job.tool_name != "mock_report":
                return
            job.status, job.claim_token = "running", claim
            job.lease_until = now() + timedelta(seconds=180)
            carrier = job.trace_carrier
            tenant_id, user_id = job.tenant_id, job.user_id
        context_token = attach(extract(carrier))
        with span("worker.mock_report", job_id=job_id, tool_version="mock-1.1"):
            await asyncio.sleep(1)
            body = json.dumps(
                {
                    "job_id": job_id,
                    "kind": "synthetic_engineering_fixture",
                    "message": "这是异步队列工程演示结果，不包含医学判断、影像检测或分割结果。",
                    "model_id": "mock-report",
                    "model_version": "mock-1.1",
                },
                ensure_ascii=False,
            ).encode()
            asset_id = new_id("asset")
            asset_key = asset_id + ".json"
            storage = LocalStorage(settings.storage_root)
            file_hash = await asyncio.to_thread(storage.write, asset_key, body)
            async with sessions() as session, session.begin():
                job = await session.get(ToolJob, job_id, with_for_update=True)
                if job.claim_token != claim or job.status != "running":
                    storage.remove(asset_key)
                    return
                session.add(
                    Asset(
                        id=asset_id,
                        tenant_id=tenant_id,
                        owner_user_id=user_id,
                        object_key=asset_key,
                        file_hash=file_hash,
                        mime="application/json",
                        size_bytes=len(body),
                    )
                )
                await session.flush()
                job.status, job.finished_at, job.lease_until = "succeeded", now(), None
                job.result_json = {
                    "asset_id": asset_id,
                    "model_id": "mock-report",
                    "model_version": "mock-1.1",
                }
    except Exception as exc:
        log("worker.failed", job_id=job_id, error_code="WORKER_FAILED", exception_type=type(exc).__name__)
        if asset_key:
            LocalStorage(settings.storage_root).remove(asset_key)
        async with sessions() as session, session.begin():
            job = await session.get(ToolJob, job_id, with_for_update=True)
            if job and job.claim_token == claim and job.status == "running":
                job.status, job.error_code, job.finished_at = "failed", "WORKER_FAILED", now()
                job.lease_until = None
        raise RuntimeError("WORKER_FAILED") from None
    finally:
        if context_token is not None:
            detach(context_token)
        await engine.dispose()


def worker_app():
    settings = Settings()
    configure(settings)
    app = make_celery(settings)

    @app.task(name="pharma_agent.execute_mock_report")
    def task(job_id):
        asyncio.run(execute_mock_job(settings, job_id))

    @app.task(
        name="pharma_agent.execute_segmentation",
        time_limit=settings.segmentation_timeout_seconds + 60,
        soft_time_limit=settings.segmentation_timeout_seconds + 30,
    )
    def segmentation_task(job_id):
        from pharma_agent.segmentation import execute_segmentation_job

        asyncio.run(execute_segmentation_job(settings, job_id))

    return app
