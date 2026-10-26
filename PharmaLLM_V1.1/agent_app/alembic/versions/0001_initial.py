"""Frozen initial business schema; LangGraph owns its checkpoint migrations separately."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def col(name, type_, nullable=False, **kwargs):
    return sa.Column(name, type_, nullable=nullable, **kwargs)


def ref(name, target, nullable=False):
    return sa.Column(name, sa.String(64), sa.ForeignKey(target), nullable=nullable)


def record(name, *columns):
    op.create_table(
        name,
        col("id", sa.String(64), primary_key=True),
        col("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        *columns,
    )


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    record("tenants", col("name", sa.String(120)), col("status", sa.String(20)))
    record(
        "users",
        ref("tenant_id", "tenants.id"),
        col("email", sa.String(254)),
        col("password_hash", sa.Text()),
        col("role", sa.String(20)),
        col("status", sa.String(20)),
        sa.UniqueConstraint("tenant_id", "email"),
    )
    record(
        "conversations",
        ref("tenant_id", "tenants.id"),
        ref("owner_user_id", "users.id"),
        col("title", sa.String(120)),
        col("thread_id", sa.String(100), unique=True),
        col("version", sa.Integer()),
        col("active_run_id", sa.String(64), True),
        col("deleted_at", sa.DateTime(timezone=True), True),
    )
    record(
        "agent_runs",
        ref("tenant_id", "tenants.id"),
        ref("user_id", "users.id"),
        ref("conversation_id", "conversations.id"),
        col("idempotency_key", sa.String(128)),
        col("request_hash", sa.String(80)),
        col("request_id", sa.String(64)),
        col("trace_id", sa.String(32)),
        col("trace_carrier", sa.JSON()),
        col("status", sa.String(20)),
        col("model_snapshot", sa.JSON()),
        col("graph_version", sa.String(80)),
        col("final_answer", sa.Text(), True),
        col("citations", sa.JSON()),
        col("usage", sa.JSON()),
        col("error_code", sa.String(80), True),
        col("worker_id", sa.String(64), True),
        col("lease_until", sa.DateTime(timezone=True), True),
        col("started_at", sa.DateTime(timezone=True), True),
        col("finished_at", sa.DateTime(timezone=True), True),
        sa.UniqueConstraint("tenant_id", "user_id", "conversation_id", "idempotency_key"),
    )
    op.create_index(
        "uq_active_conversation_run",
        "agent_runs",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running', 'cancelling')"),
    )
    record(
        "messages",
        ref("conversation_id", "conversations.id"),
        ref("run_id", "agent_runs.id"),
        col("role", sa.String(20)),
        col("content", sa.Text()),
        sa.UniqueConstraint("run_id", "role"),
    )
    op.create_table(
        "run_events",
        col("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        ref("run_id", "agent_runs.id"),
        col("kind", sa.String(40)),
        col("payload", sa.JSON()),
        col("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    record(
        "user_memories",
        ref("tenant_id", "tenants.id"),
        ref("user_id", "users.id"),
        col("kind", sa.String(40)),
        col("value", sa.String(40)),
        col("consent_at", sa.DateTime(timezone=True)),
        col("expires_at", sa.DateTime(timezone=True), True),
        col("version", sa.Integer()),
        sa.UniqueConstraint("tenant_id", "user_id", "kind"),
    )
    record(
        "assets",
        ref("tenant_id", "tenants.id"),
        ref("owner_user_id", "users.id"),
        col("object_key", sa.String(100), unique=True),
        col("file_hash", sa.String(80)),
        col("mime", sa.String(80)),
        col("size_bytes", sa.BigInteger()),
        col("status", sa.String(20)),
    )
    record(
        "tool_jobs",
        ref("tenant_id", "tenants.id"),
        ref("user_id", "users.id"),
        col("tool_name", sa.String(80)),
        col("tool_version", sa.String(40)),
        col("status", sa.String(20)),
        col("idempotency_key", sa.String(128)),
        col("input_json", sa.JSON()),
        col("result_json", sa.JSON()),
        col("error_code", sa.String(80), True),
        col("trace_carrier", sa.JSON()),
        col("claim_token", sa.String(64), True),
        col("lease_until", sa.DateTime(timezone=True), True),
        col("confirmed_at", sa.DateTime(timezone=True), True),
        col("finished_at", sa.DateTime(timezone=True), True),
        sa.UniqueConstraint("tenant_id", "user_id", "tool_name", "idempotency_key"),
    )
    record(
        "task_outbox",
        ref("job_id", "tool_jobs.id"),
        col("published", sa.Boolean()),
        col("attempts", sa.Integer()),
        col("next_attempt_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("job_id"),
    )
    record(
        "documents",
        ref("tenant_id", "tenants.id", True),
        col("visibility", sa.String(20)),
        col("title", sa.String(200)),
        col("source", sa.String(500)),
        col("source_version", sa.String(100)),
        col("license", sa.String(200)),
        col("published_at", sa.String(40)),
        col("content_hash", sa.String(80)),
        sa.UniqueConstraint("tenant_id", "source", "source_version"),
    )
    # NULL tenant IDs also need a public-source uniqueness constraint.
    op.create_index(
        "uq_public_document_version",
        "documents",
        ["source", "source_version"],
        unique=True,
        postgresql_where=sa.text("tenant_id IS NULL"),
    )
    record(
        "document_chunks",
        ref("document_id", "documents.id"),
        col("chunk_index", sa.Integer()),
        col("content", sa.Text()),
        col("embedding", Vector()),
        col("index_version", sa.String(100)),
        sa.UniqueConstraint("document_id", "chunk_index", "index_version"),
    )
    record(
        "audit_events",
        col("tenant_id", sa.String(64), True),
        col("actor_id", sa.String(64), True),
        col("action", sa.String(80)),
        col("resource_kind", sa.String(40)),
        col("resource_id", sa.String(100)),
        col("outcome", sa.String(40)),
        col("trace_id", sa.String(32)),
    )
    record(
        "user_feedback",
        ref("user_id", "users.id"),
        ref("run_id", "agent_runs.id"),
        col("rating", sa.Integer()),
        col("category", sa.String(40)),
        sa.UniqueConstraint("user_id", "run_id"),
    )
    indexes = {
        "users": ["tenant_id", "email"],
        "conversations": ["tenant_id", "owner_user_id"],
        "agent_runs": ["tenant_id", "user_id", "conversation_id", "status", "trace_id"],
        "messages": ["conversation_id", "run_id"],
        "run_events": ["run_id"],
        "user_memories": ["tenant_id", "user_id"],
        "assets": ["tenant_id", "owner_user_id"],
        "tool_jobs": ["tenant_id", "user_id", "status"],
        "documents": ["tenant_id"],
        "document_chunks": ["document_id", "index_version"],
        "audit_events": ["tenant_id", "trace_id"],
        "user_feedback": ["user_id", "run_id"],
    }
    for table, names in indexes.items():
        for name in names:
            op.create_index(f"ix_{table}_{name}", table, [name])
    op.execute("""CREATE FUNCTION reject_audit_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'audit_events is append-only'; END; $$""")
    op.execute(
        "CREATE TRIGGER audit_append_only BEFORE UPDATE OR DELETE ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation()"
    )


def downgrade():
    for table in [
        "user_feedback",
        "audit_events",
        "document_chunks",
        "documents",
        "task_outbox",
        "tool_jobs",
        "assets",
        "user_memories",
        "run_events",
        "messages",
        "agent_runs",
        "conversations",
        "users",
        "tenants",
    ]:
        op.drop_table(table)
    op.execute("DROP FUNCTION IF EXISTS reject_audit_mutation()")
