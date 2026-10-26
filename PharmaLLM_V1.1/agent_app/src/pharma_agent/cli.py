import argparse
import asyncio
import json
from pathlib import Path

import httpx
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select

from pharma_agent.db import (
    AgentRun,
    Asset,
    AuditEvent,
    Conversation,
    Feedback,
    Message,
    RunEvent,
    User,
    audit,
    create_database,
    new_id,
)
from pharma_agent.models import ModelGateway
from pharma_agent.observability import configure, digest, span, trace_id
from pharma_agent.retrieval import EmbeddingGateway, RetrievalService
from pharma_agent.schemas import DocumentInput
from pharma_agent.settings import Settings


async def run_command(args, settings):
    if args.command == "checkpoint-setup":
        async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_url) as saver:
            await saver.setup()  # Official package migrations, never copied business-table schemas.
        print("LangGraph checkpoint schema initialized.")
        return
    engine, sessions = create_database(settings.database_url)
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            if args.command == "seed-demo":
                retrieval = RetrievalService(sessions, EmbeddingGateway(settings, client))
                item = DocumentInput(
                    title="合成工程演示资料：任务与引用",
                    source="synthetic://pharma-agent/demo",
                    source_version="1.1",
                    license="CC0-1.0 (project-authored synthetic fixture)",
                    published_at="2026-10-21",
                    content="这是一份合成工程演示资料，不是临床指南。"
                    "本系统可以查询本人任务、创建任务草稿、在明确确认后提交演示任务，"
                    "以及返回包含文档编号、分块编号和来源版本的参考资料引用。"
                    "演示知识库不提供疾病诊断、用药建议或医学影像分割证据。",
                )
                print(
                    json.dumps(await retrieval.ingest(item, tenant_id=None, public=True), ensure_ascii=False)
                )
            elif args.command == "ingest":
                manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
                retrieval = RetrievalService(sessions, EmbeddingGateway(settings, client))
                if not args.public and not args.tenant_id:
                    raise ValueError("必须提供 --tenant-id 或 --public")
                if args.public and args.tenant_id:
                    raise ValueError("--public 与 --tenant-id 互斥")
                for item in manifest:
                    source_path = (args.manifest.parent / item.pop("path")).resolve()
                    if (
                        source_path.suffix.lower() not in {".txt", ".md"}
                        or source_path.stat().st_size > 800000
                    ):
                        raise ValueError("导入只支持不超过 800KB 的 UTF-8 TXT/MD 文件")
                    document = DocumentInput(**item, content=source_path.read_text(encoding="utf-8"))
                    print(
                        json.dumps(
                            await retrieval.ingest(document, args.tenant_id, args.public), ensure_ascii=False
                        )
                    )
            elif args.command == "purge-expired-assets":
                from datetime import datetime
                from pharma_agent.db import now
                from pharma_agent.imaging import ImagingStore
                from pharma_agent.storage import LocalStorage

                imaging = ImagingStore(LocalStorage(settings.storage_root), settings)
                scanned, purged = 0, 0
                async with sessions() as session, session.begin():
                    rows = (
                        await session.scalars(
                            select(Asset)
                            .where(Asset.status == "ready")
                            .order_by(Asset.created_at)
                            .with_for_update(skip_locked=True)
                        )
                    ).all()
                    for asset in rows:
                        scanned += 1
                        expiry = imaging.metadata(asset).get("expires_at")
                        if expiry and datetime.fromisoformat(expiry) <= now():
                            imaging.storage.remove(asset.object_key)
                            imaging.purge(asset.id)
                            asset.status = "deleted"
                            purged += 1
                            audit(
                                session,
                                None,
                                "operator.asset_expiry",
                                "asset",
                                asset.id,
                                "content_deleted",
                                trace_id(),
                            )
                print(json.dumps({"scanned": scanned, "purged": purged}))
            elif args.command == "model-health":
                await ModelGateway(settings, client).health()
                print("Configured model aliases are available.")
            elif args.command == "add-user":
                # Trusted operator CLI, never a public API allowing arbitrary tenant enrollment.
                from getpass import getpass
                from pharma_agent.security import hash_password

                from pharma_agent.schemas import PASSWORD_MIN_LENGTH, PASSWORD_MAX_LENGTH

                password = getpass("New user's password (6–12 chars): ").strip()
                if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
                    raise ValueError("新密码需为 6–12 个字符")
                hashed = await hash_password(password)
                async with sessions() as session, session.begin():
                    user = User(
                        id=new_id("user"),
                        tenant_id=args.tenant_id,
                        email=args.email.lower(),
                        password_hash=hashed,
                        role="doctor",
                    )
                    session.add(user)
                    audit(session, None, "operator.add_user", "user", user.id, "succeeded", trace_id())
                print(json.dumps({"user_id": user.id, "tenant_id": user.tenant_id}))
            elif args.command == "export-failmin":
                # CLI requires local DB credentials. Exports metadata/hashes only, no replayable writes.
                async with sessions() as session:
                    run = await session.get(AgentRun, args.run_id)
                    if run is None:
                        raise ValueError("run not found")
                    events = (
                        await session.scalars(
                            select(RunEvent).where(RunEvent.run_id == run.id).order_by(RunEvent.id)
                        )
                    ).all()
                    audits = (
                        await session.scalars(select(AuditEvent).where(AuditEvent.trace_id == run.trace_id))
                    ).all()
                payload = {
                    "schema_version": "failmin-metadata-1",
                    "external_writes": "mock_only",
                    "run_id": run.id,
                    "trace_id": run.trace_id,
                    "graph_version": run.graph_version,
                    "model_snapshot": run.model_snapshot,
                    "status": run.status,
                    "error_code": run.error_code,
                    "request_hash": run.request_hash,
                    "answer_hash": digest(run.final_answer),
                    "events": [
                        {"id": event.id, "kind": event.kind, "payload_hash": digest(event.payload)}
                        for event in events
                    ],
                    "audit": [{"action": event.action, "outcome": event.outcome} for event in audits],
                    "predicate": {"expected_error_code": run.error_code},
                }
                # URLs are deployment config; remove endpoint URLs from exported fixtures.
                for config in payload["model_snapshot"]["models"].values():
                    config.pop("base_url", None)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                args.output.chmod(0o600)
                with span("operator.export", run_id=run.id):
                    async with sessions() as session, session.begin():
                        audit(session, None, "operator.export", "run", run.id, "metadata_only", trace_id())
                print(str(args.output))
            elif args.command == "purge-conversation":
                async with sessions() as session:
                    conversation = (
                        await session.scalars(
                            select(Conversation).where(
                                Conversation.id == args.conversation_id,
                                Conversation.tenant_id == args.tenant_id,
                            )
                        )
                    ).first()
                    if not conversation or not conversation.deleted_at or conversation.active_run_id:
                        raise ValueError("只允许清理指定租户内已软删除、无活动 Run 的会话")
                async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_url) as saver:
                    await saver.adelete_thread(conversation.thread_id)
                async with sessions() as session, session.begin():
                    ids = select(AgentRun.id).where(AgentRun.conversation_id == conversation.id)
                    await session.execute(delete(Feedback).where(Feedback.run_id.in_(ids)))
                    await session.execute(delete(RunEvent).where(RunEvent.run_id.in_(ids)))
                    await session.execute(delete(Message).where(Message.conversation_id == conversation.id))
                    await session.execute(delete(AgentRun).where(AgentRun.conversation_id == conversation.id))
                    await session.execute(delete(Conversation).where(Conversation.id == conversation.id))
                    audit(
                        session,
                        None,
                        "operator.purge_conversation",
                        "conversation",
                        conversation.id,
                        "content_deleted",
                        trace_id(),
                    )
                print("会话内容及官方 Checkpoint 已清理；追加式审计保留无正文元数据。")
    finally:
        await engine.dispose()


async def command(args):
    settings = Settings()
    configure(settings)
    with span("operator.command", command=args.command):
        await run_command(args, settings)


def main():
    parser = argparse.ArgumentParser(description="PharmaLLM Agent operator commands")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("checkpoint-setup", "seed-demo", "model-health", "purge-expired-assets"):
        sub.add_parser(name)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("manifest", type=Path)
    ingest.add_argument("--tenant-id")
    ingest.add_argument("--public", action="store_true")
    add_user = sub.add_parser("add-user")
    add_user.add_argument("--tenant-id", required=True)
    add_user.add_argument("--email", required=True)
    export = sub.add_parser("export-failmin")
    export.add_argument("run_id")
    export.add_argument("--output", type=Path, default=Path("reports/failmin.json"))
    purge = sub.add_parser("purge-conversation")
    purge.add_argument("--tenant-id", required=True)
    purge.add_argument("--conversation-id", required=True)
    args = parser.parse_args()
    asyncio.run(command(args))


if __name__ == "__main__":
    main()
