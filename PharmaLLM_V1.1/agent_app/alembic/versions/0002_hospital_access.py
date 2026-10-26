"""Readable team codes, departments, execution policy and explicit result reviews.

Existing resource IDs and ownership are preserved. Run only via migrate.sh.
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("tenants", sa.Column("login_code", sa.String(32), nullable=True))
    op.execute("""WITH numbered AS (
        SELECT id, row_number() OVER (ORDER BY created_at, id) AS n FROM tenants
    ) UPDATE tenants SET login_code = 'team-' || lpad(numbered.n::text, 6, '0')
      FROM numbered WHERE tenants.id = numbered.id""")
    op.alter_column("tenants", "login_code", nullable=False)
    op.create_unique_constraint("uq_tenants_login_code", "tenants", ["login_code"])
    op.create_table(
        "departments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.UniqueConstraint("tenant_id", "name"),
    )
    op.create_index("ix_departments_tenant_id", "departments", ["tenant_id"])
    op.add_column("users", sa.Column("department_id", sa.String(64), nullable=True))
    op.create_foreign_key("fk_users_department", "users", "departments", ["department_id"], ["id"])
    op.add_column("users", sa.Column("display_name", sa.String(80), nullable=False, server_default=""))
    op.add_column(
        "users", sa.Column("execution_mode", sa.String(24), nullable=False, server_default="ask_for_approval")
    )
    op.execute("UPDATE users SET role = 'doctor' WHERE role = 'member'")
    op.create_table(
        "segmentation_reviews",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("tool_jobs.id"), nullable=False, unique=True),
        sa.Column("tenant_id", sa.String(64), sa.ForeignKey("tenants.id"), nullable=False),
        sa.Column("department_id", sa.String(64), sa.ForeignKey("departments.id"), nullable=False),
        sa.Column("submitter_id", sa.String(64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("source_asset_id", sa.String(64), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("result_asset_id", sa.String(64), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("reviewer_id", sa.String(64), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("comment", sa.String(2000), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    for column in ("tenant_id", "department_id", "status"):
        op.create_index("ix_segmentation_reviews_" + column, "segmentation_reviews", [column])


def downgrade():
    op.drop_table("segmentation_reviews")
    op.drop_column("users", "execution_mode")
    op.drop_column("users", "display_name")
    op.drop_constraint("fk_users_department", "users", type_="foreignkey")
    op.drop_column("users", "department_id")
    op.drop_table("departments")
    op.drop_constraint("uq_tenants_login_code", "tenants", type_="unique")
    op.drop_column("tenants", "login_code")
    op.execute("UPDATE users SET role = 'member' WHERE role IN ('doctor', 'reviewer')")
