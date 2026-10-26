from datetime import UTC, datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class Base(DeclarativeBase):
    pass


class Record:
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Tenant(Record, Base):
    __tablename__ = "tenants"
    login_code: Mapped[str] = mapped_column(String(32), unique=True,
                                          default=lambda: "team-" + uuid4().hex[:10])
    name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="active")


class Department(Record, Base):
    __tablename__ = "departments"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(80))


class User(Record, Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email"),)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(254), index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20), default="doctor")
    status: Mapped[str] = mapped_column(String(20), default="active")
    department_id: Mapped[str | None] = mapped_column(ForeignKey("departments.id"), nullable=True)
    display_name: Mapped[str] = mapped_column(String(80), default="")
    execution_mode: Mapped[str] = mapped_column(String(24), default="ask_for_approval")


class Conversation(Record, Base):
    __tablename__ = "conversations"
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    owner_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(120), default="新会话")
    thread_id: Mapped[str] = mapped_column(String(100), unique=True)
    version: Mapped[int] = mapped_column(Integer, default=0)
    active_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentRun(Record, Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", "conversation_id", "idempotency_key"),
        Index(
            "uq_active_conversation_run",
            "conversation_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running', 'cancelling')"),
        ),
    )
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(80))
    request_id: Mapped[str] = mapped_column(String(64))
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    trace_carrier: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), index=True, default="queued")
    model_snapshot: Mapped[dict] = mapped_column(JSON)
    graph_version: Mapped[str] = mapped_column(String(80))
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    usage: Mapped[dict] = mapped_column(JSON, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Message(Record, Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("run_id", "role"),)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)


class RunEvent(Base):
    __tablename__ = "run_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class UserMemory(Record, Base):
    __tablename__ = "user_memories"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id", "kind"),)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    value: Mapped[str] = mapped_column(String(40))
    consent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)


class Asset(Record, Base):
    __tablename__ = "assets"
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    owner_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    object_key: Mapped[str] = mapped_column(String(100), unique=True)
    file_hash: Mapped[str] = mapped_column(String(80))
    mime: Mapped[str] = mapped_column(String(80))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(20), default="ready")


class ToolJob(Record, Base):
    __tablename__ = "tool_jobs"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id", "tool_name", "idempotency_key"),)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    tool_name: Mapped[str] = mapped_column(String(80), default="mock_report")
    tool_version: Mapped[str] = mapped_column(String(40), default="mock-1.1")
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    input_json: Mapped[dict] = mapped_column(JSON)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    trace_carrier: Mapped[dict] = mapped_column(JSON, default=dict)
    claim_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TaskOutbox(Record, Base):
    __tablename__ = "task_outbox"
    job_id: Mapped[str] = mapped_column(ForeignKey("tool_jobs.id"), unique=True)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SegmentationReview(Record, Base):
    __tablename__ = "segmentation_reviews"
    job_id: Mapped[str] = mapped_column(ForeignKey("tool_jobs.id"), unique=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    department_id: Mapped[str] = mapped_column(ForeignKey("departments.id"), index=True)
    submitter_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    source_asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    result_asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"))
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    comment: Mapped[str] = mapped_column(String(2000), default="")
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Document(Record, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "source", "source_version"),
        Index(
            "uq_public_document_version",
            "source",
            "source_version",
            unique=True,
            postgresql_where=text("tenant_id IS NULL"),
        ),
    )
    tenant_id: Mapped[str | None] = mapped_column(ForeignKey("tenants.id"), nullable=True, index=True)
    visibility: Mapped[str] = mapped_column(String(20), default="tenant")
    title: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(500))
    source_version: Mapped[str] = mapped_column(String(100))
    license: Mapped[str] = mapped_column(String(200))
    published_at: Mapped[str] = mapped_column(String(40))
    content_hash: Mapped[str] = mapped_column(String(80))


class DocumentChunk(Record, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", "index_version"),)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    index_version: Mapped[str] = mapped_column(String(100), index=True)


class AuditEvent(Record, Base):
    __tablename__ = "audit_events"
    tenant_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(80))
    resource_kind: Mapped[str] = mapped_column(String(40))
    resource_id: Mapped[str] = mapped_column(String(100))
    outcome: Mapped[str] = mapped_column(String(40))
    trace_id: Mapped[str] = mapped_column(String(32), index=True)


class Feedback(Record, Base):
    __tablename__ = "user_feedback"
    __table_args__ = (UniqueConstraint("user_id", "run_id"),)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(40))


def create_database(url: str):
    engine = create_async_engine(url, pool_pre_ping=True, echo=False, pool_size=12, max_overflow=12)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def audit(session, actor, action: str, kind: str, resource_id: str, outcome: str, trace: str):
    session.add(
        AuditEvent(
            id=new_id("audit"),
            tenant_id=getattr(actor, "tenant_id", None),
            actor_id=getattr(actor, "id", None),
            action=action,
            resource_kind=kind,
            resource_id=resource_id,
            outcome=outcome,
            trace_id=trace,
        )
    )
